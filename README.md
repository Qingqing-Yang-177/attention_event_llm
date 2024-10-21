This is the repo of code and data for [paper](https://openreview.net/pdf?id=l0K0ADWKTq) "Attention redistribution during event segmentation in Large Language Model", NeurIPS 2024 Workshop on Behavioral ML.

To execute the whole project on your hpc or server, pls follow the instructions below:

```
cd ./attention_event_llm

# create virtual environment on hpc server, in the terminal:
python -m venv llmenv

# activate the environment
source ./attention_event_llm/llmenv/bin/activate

# download packages
cd ./attention_event_llm/llmenv/lib/python3.11/site-packages
pip install torch safetensors transformers datasets accelerate peft bitsandbytes matplotlib seaborn spacy striptf scipy ipykernel numpy pandas os sys

# download scipy model
cd ./attention_event_llm/llmenv/bin/
python -m spacy download en_core_web_sm
  
# add virtual environment to Jupyter notebooks
python -m ipykernel install --user --name=llmenv --display-name "llmenv"

# to run the script to generate the attention scores

# non segmentation for each story, testing non-punctuation
sbatch attention_event_noseg.sh story1.txt 0

# segmentation for each story, testing punctuation
sbatch attention_event_seg.sh story2.txt 1 
```
