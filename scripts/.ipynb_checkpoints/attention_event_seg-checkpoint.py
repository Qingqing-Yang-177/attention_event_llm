# Importing Dependencies
import numpy as np
import pandas as pd
#from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, pipeline, AutoModel
import accelerate
import peft
import scipy
from scipy import spatial
import os
import spacy # spaCy is used to isolate linguistic features like "noun", "determiners" from the story
from striprtf.striprtf import rtf_to_text
from matplotlib import pyplot as plt
import seaborn as sns
import sys
from read_file import read_file

text_tested=str(sys.argv[1])
check_punct=int(sys.argv[2]) # a boolean whether we are checking punctuation

# Save the results to npy files
foldername = os.path.splitext(text_tested)[0] # Extract the filename without the extension
output_dir = f'/scratch/qy775/attention_event/results/{foldername}_seg'
os.makedirs(output_dir, exist_ok=True)

# Setup the input file path
story_dir = '/scratch/qy775/attention_event/story'
file_path = os.path.join(story_dir, text_tested)
story = read_file(file_path)

# Display the plain text content
print(story)

# Setting device
has_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting up quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Log into HuggingFace, use input token
from huggingface_hub import login
token = "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"
login(token=token)

# setup model
model_id= "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer= AutoTokenizer.from_pretrained (model_id)
model= AutoModelForCausalLM.from_pretrained(model_id,
                                           device_map="auto",
                                           trust_remote_code=True,
                                           quantization_config=quantization_config,
                                           attn_implementation="eager")  # Specify the manual attention implementation)
                                           #output_hidden_states=True,
# set model to evaluation mode
model.eval()

# Read text
# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')

# Process the plain text with spaCy to identify linguistic features
doc = nlp(story)

# define prompt seg
prompt= f''' An event is an ongoing coherent situation. The following story needs to
be copied and segmented into events. Copy the following story word-for-word and start a new line
whenever one event ends and another begins.

This is the story: {story}

This is a word-for-word copy of the same story that is segmented into events:'''

# setup prompt and tokenization
messages = [
    {"role": "system", "content": "You are a person listening to a continuous story, which can be divided into distinct events. "},
    {"role": "user", "content": prompt },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# generating response with model
from torch.cuda.amp import autocast
terminators = [tokenizer.eos_token_id]

with autocast():
    outputs = model.generate(
        input_ids,
        max_new_tokens=3000, #must be longer than the original story
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        #top_p=0.9,
  )

# extract and decode the response
response = outputs[0][input_ids.shape[-1]:] # get only post input tokens
print("Generated response:", tokenizer.decode(response, skip_special_tokens=True))

# Get Output Attention
from torch.cuda.amp import autocast
with torch.no_grad():
    with autocast():  # enables mixed precision to manage GPU RAM
        output = model(input_ids=outputs, output_attentions=True)
        attentions = output.attentions


# Load the spaCy model for English
nlp = spacy.load('en_core_web_sm')
doc = nlp(story)

# Initialize a dictionary to store attention values for each POS tag
# Initialize a dictionary to store attention values for each POS tag
if check_punct==0:
    pos_tags = [
        'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
        'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ', 'SYM', 'VERB', 'X'
    ]
    # run post_tags other than PUNCT first with 
    # tokens = [token.text for token in doc if token.pos_ == pos_tag and not token.is_punct]
else:
    pos_tags = ['PUNCT']
    # seperately run pos_tag is PUNCT with
    # tokens = [token.text for token in doc if token.pos_ == pos_tag]

attention_results = {tag: [] for tag in pos_tags}
#token_positions_results = {tag: [] for tag in pos_tags}

# Iterate over each POS tag
for pos_tag in pos_tags:
    token_positions = []
    if check_punct==0:
        tokens = [token.text for token in doc if token.pos_ == pos_tag and not token.is_punct]
    else:
        tokens = [token.text for token in doc if token.pos_ == pos_tag]
        
    print(f"{pos_tag} Token identified: {tokens}")
    
    # Tokenize the identified tokens
    for token in tokens:
        tokenized_token = tokenizer.tokenize(token)
        for subword in tokenized_token:
            token_index = tokenizer.convert_tokens_to_ids(subword)
            positions = (input_ids == token_index).nonzero(as_tuple=True)[1].tolist()
            token_positions.extend(positions)

    # Ensure there are token positions found
    if not token_positions:
        print(f"No tokens found for POS tag: {pos_tag}")
        continue
        
    # Store token positions in the dictionary
    #token_positions_results[pos_tag] = token_positions
    
    # Aggregate attention values for all tokens corresponding to the POS tag
    layers = len(attentions)
    heads = attentions[0].size(1)
    attention_values = np.zeros((layers, heads))

    for layer in range(layers):
        for pos in token_positions:
            # Sum of attention directed to all tokens across all tokens in the sequence
            attention_values[layer] += attentions[layer][0, :, :, pos].sum(dim=-1).cpu().numpy()

    # Average attention scores per word
    num_extracted = len(token_positions)
    # get attention for each head and each token
    attention_all= attention_values/num_extracted
    # across all heads for each layer
    avg_attention_scores = attention_values.mean(axis=1) 
    avg_attention_scores_perword = avg_attention_scores / num_extracted
    # get a mean across layers
    avg_attention_all = avg_attention_scores_perword.mean()

    # Store the average attention scores in the dictionary
    attention_results[pos_tag] = avg_attention_scores_perword

    print(f"Average attention across all layers for {pos_tag}: {avg_attention_all}")

    np.save(os.path.join(output_dir, f'{pos_tag}_attention_all.npy'), attention_all)


for pos_tag, avg_attention in attention_results.items():
    avg_attention = np.array(avg_attention) # turn the list into numpy array
    if avg_attention.size > 0:  # Ensure we have values before saving
        np.save(os.path.join(output_dir, f'{pos_tag}_attention_layer.npy'), avg_attention)
        #np.save(os.path.join(output_dir, f'{pos_tag}_token_positions.npy'), token_positions_results[pos_tag])