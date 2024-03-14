import torch
import requests
import logging
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import norm_logits, sample, max_fn

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
        
    def generate(self, prefix:str, max_len:int=50, num_return_sequences:int=1)->str:
        input_ids = self._tokenizer.encode(prefix, return_tensors='pt').to(self._device)
        output = self._model.generate(input_ids, max_length=max_len, num_return_sequences=num_return_sequences)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    def generate_from_server(self, prefix:str, max_len:int=50, num_return_sequences:int=1)->str:
        data = {"prompt": prefix, "max_length": max_len, "num_return_sequences": num_return_sequences}
        response = requests.post(f"{self.server_url}/predict", json=data)
        generated_text = response.json()
        return generated_text
    
    def generate_with_server(self, prefix:str,  max_len : int = 20 , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None)->str:
        prefix = self._tokenizer.encode(prefix, return_tensors='pt').to(self._device)
        seq_len = prefix.shape[1]
        T = seq_len + max_len
        
        assert prefix.shape[0] == 1, "input batch size must be 1"

        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = self._model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = self._spec_logits(x)
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break
        
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)

        return prefix

    def _spec_logits(self, tensor:torch.Tensor)->torch.Tensor:
        data = {"ids": tensor.tolist()}
        response = requests.post(f"{self.server_url}/spec_logits", json=data)
        processed_tensor = torch.tensor(response.json()["logits"])
        print(processed_tensor.shape)
        return processed_tensor
    
model = ClientModel(server_url="116.62.162.232:5000",model_name="/Users/kylinchan/models/bloom-560m")
print(model.generate_with_server("The quick brown fox jumps over the lazy dog"))

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