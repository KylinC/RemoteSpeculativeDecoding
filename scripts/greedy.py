from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch
from tqdm import tqdm

approx_model_path = "bigscience/bloom-560m"
target_model_path = "bigscience/bloomz-7b1"
_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(target_model_path)
target_model = AutoModelForCausalLM.from_pretrained(target_model_path,torch_dtype=torch.float16,trust_remote_code=True).to(_device)
approx_model = AutoModelForCausalLM.from_pretrained(approx_model_path,torch_dtype=torch.float16,trust_remote_code=True).to(_device)

logging.basicConfig(level=logging.INFO)

def speculative_decoding_greedy():
    pass

@torch.no_grad()
def speculative_greedy(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4) -> torch.Tensor:
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            x = prefix
            prefix_len = prefix.shape[1]
            x = approx_model.generate(x, max_length=prefix_len + gamma)
            
            y = target_model(x).logits.argmax(dim=2)
            n = prefix_len - 1
            for _ in range(gamma):
                if y[0][n]==x[0][n+1]:
                    # accept, and update n
                    n += 1 
                else:
                    # reject
                    print(f"reject {n+1}")
                    x[0][n+1] = y[0][n]
                    break
        
            prefix = x[:, :n + 2]
            pbar.update(n - pbar.n)

    return prefix

input_text = "you is of" 
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(_device)

standard_output_ids = target_model.generate(input_ids, max_length=20)

speculative_output_ids = speculative_greedy(input_ids, approx_model, target_model, 20, 4)

import IPython; IPython.embed(); exit(1)


last_hidden_states = outputs.last_hidden_state

hidden_states = outputs.hidden_states

