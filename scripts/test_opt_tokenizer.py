import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/kylin/models/opt-125")

tokenizer2 = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")

tokenizer3 = AutoTokenizer.from_pretrained("openai-community/gpt2")

import IPython; IPython.embed(); exit(1)

prompt_input_ids = tokenizer.encode("you love", return_tensors="pt")