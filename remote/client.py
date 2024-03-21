import torch
import requests
import logging
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import norm_logits, sample, max_fn, timer, get_timer_stats, update_timer

from typing import Tuple

logging.basicConfig(level=logging.INFO)

class ClientModel:
    def __init__(self, server_url:str=None, model_name:str="bigscience/bloom-560m")->None:
        self.server_url = server_url
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info("begin load models")
        self._model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("fininsh load models")
        
    def set_server_url(self, server_url:str)->bool:
        self.server_url = server_url
        return self.server_url!=None
        
    def generate(self, prompt:str, max_len:int=50, num_return_sequences:int=1)->str:
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        output = self._model.generate(input_ids, max_length=max_len, num_return_sequences=num_return_sequences)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    def generate_from_server(self, prompt:str, max_len:int=50, num_return_sequences:int=1)->str:
        data = {"prompt": prompt, "max_length": max_len, "num_return_sequences": num_return_sequences}
        response = requests.post(f"{self.server_url}/generate", json=data)
        generated_text = response.json()["generated_text"]
        return generated_text
    
    def generate_with_server(self, prompt:str,  max_len : int = 20 , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, 
                         random_seed : int = None, strategy: str = "greedy")->str:
        
        '''
        strategy: ["greedy", "speculative"]
        '''
        
        prefix = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        
        if strategy == "greedy":
            output = self._block_decoding(prefix, max_len, gamma)
        elif strategy == "speculative":
            output = self._spec_sampling(prefix, max_len, gamma, temperature, top_k, top_p, random_seed)
        else:
            raise ValueError("Strategy Undefined")
        
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    def _block_decoding(self, prefix : torch.Tensor, max_len : int , gamma : int = 4) -> torch.Tensor:
        seq_len = prefix.shape[1]
        T = seq_len + max_len
        
        assert prefix.shape[0] == 1, "input batch size must be 1"
        
        while prefix.shape[1] < T:
            x = prefix
            prefix_len = prefix.shape[1]
            x = self._model.generate(x, max_length=prefix_len + gamma)
            
            prefix = self._spec_tokens(x, prefix_len, gamma)
            # y = target_model(x).logits.argmax(dim=2)
            # n = prefix_len - 1
            # for _ in range(gamma):
            #     if y[0][n]==x[0][n+1]:
            #         # accept, and update n
            #         n += 1 
            #     else:
            #         # reject
            #         print(f"reject {n+1}")
            #         x[0][n+1] = y[0][n]
            #         break
        
            # prefix = x[:, :n + 2]

        return prefix
    
    def _spec_tokens(self, ids:torch.Tensor, prefix_len:int, gamma:int)->torch.Tensor:
        if ids.device!="cpu":
            ids = ids.cpu()
        data = {"ids": ids.tolist(), "prefix_len": prefix_len, "gamma": gamma}
        response = requests.post(f"{self.server_url}/spec_tokens", json=data)
        checked_ids = torch.tensor(response.json()["ids"]).to(self._device)
        # print(processed_tensor.shape)
        return checked_ids
        
    def _spec_sampling(self, prefix:torch.Tensor,  max_len : int = 20 , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None)->torch.Tensor:
        seq_len = prefix.shape[1]
        T = seq_len + max_len
        
        assert prefix.shape[0] == 1, "input batch size must be 1"

        while prefix.shape[1] < T:
            # p = M_p[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            timer(None)
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                p = self._model(x).logits
                next_tok = sample(norm_logits(p[:, -1, :], 
                                temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            timer("sampling")
            
            # normalize the logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            timer("normalize")
            # q  = M_q[prefix + x_0, x_0, .., x_(gamma-1)]
            q = self._spec_logits(x)
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = self._device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=self._device), q[:, prefix_len + i - 1, j] / p[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(q[:, n, :] - p[:, n, :]))
                    is_all_accept = False
                    break
            timer("verify")
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(q[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            timer("result")
            
        return prefix
        
    def _spec_logits(self, tensor:torch.Tensor)->torch.Tensor:
        timer(None)
        data = {"ids": tensor.tolist()}
        response = requests.post(f"{self.server_url}/spec_logits", json=data)
        msg = response.json()
        timer("post")
        processed_tensor = torch.tensor(msg["logits"]).to(self._device)
        timer("cpu->gpu (client)")

        for name, t in msg["stats"].items():
            update_timer(name, t)

        print(processed_tensor.shape)
        return processed_tensor
    
# model_name = "bigscience/bloom-560m"
model_name = "/home/share/opt-125m"
server_url = "http://202.205.2.250:5000"
prompts = "How do you think the weather today?"

model = ClientModel(server_url=server_url, 
                    model_name=model_name)
print(model.generate_with_server(prompts))

stats = get_timer_stats()
print(stats)

exit(1)

# import IPython; IPython.embed(); exit(1)

# def send_tensor_to_server(tensor, server_url):
#     data = {"tensor": tensor.tolist()}
#     response = requests.post(f"{server_url}/process_tensor", json=data)
#     processed_tensor = torch.tensor(response.json()["processed_tensor"])
#     return processed_tensor

# # 服务端的公网 URL
# server_url = "https://58a8-222-219-180-140.ngrok-free.app"

# # 创建一个示例张量
# tensor = torch.randn(3, 4)
# print("Original tensor:")
# print(tensor)

# # 将张量发送到服务端进行处理
# processed_tensor = send_tensor_to_server(tensor, server_url)
# print("Processed tensor:")
# print(processed_tensor)