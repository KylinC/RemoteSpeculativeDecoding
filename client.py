import torch
import gradio as gr

# 加载小模型
approx_model = torch.load("path/to/approx_model.pt")

def speculative_sampling_client(prefix, max_len, gamma, temperature, top_k, top_p, random_seed):
    # 在本地执行小模型的推理
    x = prefix
    prefix_len = prefix.shape[1]
    for _ in range(gamma):
        q = approx_model(x).logits
        next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
        x = torch.cat((x, next_tok), dim=1)
    
    # 将推理结果发送给服务端
    server_output = gr.Interface.load("http://localhost:7860/").predict(
        x, prefix_len, gamma, temperature, top_k, top_p, random_seed
    )
    
    return server_output

iface = gr.Interface(
    fn=speculative_sampling_client,
    inputs=[
        gr.Tensor(label="Prefix"),
        gr.Number(label="Max Length"),
        gr.Number(label="Gamma"),
        gr.Number(label="Temperature"),
        gr.Number(label="Top K"),
        gr.Number(label="Top P"),
        gr.Number(label="Random Seed")
    ],
    outputs=gr.Tensor(label="Generated Sequence"),
    title="Speculative Sampling Client",
)

iface.launch()