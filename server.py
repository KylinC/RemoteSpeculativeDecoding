import torch
import gradio as gr
from sampling.utils import norm_logits, sample
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
target_model_name="bigscience/bloomz-7b1"

# 加载大模型
target_model = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)

def speculative_sampling_server(x, prefix_len, gamma, temperature, top_k, top_p, random_seed):
    # 在服务端执行大模型的推理
    p = target_model(x).logits
    for i in range(p.shape[1]):
        p[:,i,:] = norm_logits(p[:,i,:], temperature, top_k, top_p)
    
    # 执行修改后的拒绝采样
    is_all_accept = True
    n = prefix_len - 1
    for i in range(gamma):
        if random_seed:
            torch.manual_seed(random_seed)
        r = torch.rand(1, device=p.device)
        j = x[:, prefix_len + i]
        
        if r < torch.min(torch.tensor([1], device=p.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
            n += 1
        else:
            t = sample(max_fn(p[:, n, :] - q[:, n, :]))
            is_all_accept = False
            break
    
    prefix = x[:, :n + 1]
    
    if is_all_accept:
        t = sample(p[:, -1, :])
    
    prefix = torch.cat((prefix, t), dim=1)
    
    return prefix

iface = gr.Interface(
    fn=speculative_sampling_server,
    inputs=[
        gr.Tensor(label="X"),
        gr.Number(label="Prefix Length"),
        gr.Number(label="Gamma"),
        gr.Number(label="Temperature"),
        gr.Number(label="Top K"), 
        gr.Number(label="Top P"),
        gr.Number(label="Random Seed")
    ],
    outputs=gr.Tensor(label="Generated Sequence"),
    title="Speculative Sampling Server",
)

iface.launch(server_name="0.0.0.0", server_port=7860)