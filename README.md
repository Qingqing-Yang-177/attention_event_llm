This is a project about LLM, event perception, attention redistribution

To execute the whole project on your hpc or server, pls follow the instructions below:

```
cd ./attention_event

# create virtual environment on hpc server, in the terminal:
python -m venv llmenv

# activate the environment
source ./attention_event/llmenv/bin/activate

# download packages
cd ./attention_event/llmenv/lib/python3.11/site-packages
pip install torch safetensors transformers datasets accelerate peft bitsandbytes matplotlib seaborn spacy striptf scipy ipykernel numpy pandas os sys

# download scipy model
cd ./attention_event/llmenv/bin/
python -m spacy download en_core_web_sm
  
# add virtual environment to Jupyter notebooks
python -m ipykernel install --user --name=llmenv --display-name "llmenv"

# to run the script to generate the attention scores

# non segmentation for each story, testing non-punctuation
sbatch attention_event_noseg.sh story1.txt 0

# segmentation for each story, testing punctuation
sbatch attention_event_seg.sh story2.txt 1 
```
