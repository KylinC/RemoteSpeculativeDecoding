import torch
import gradio as gr

def process_tensor(tensor):
    # 将输入的列表转换为PyTorch张量
    tensor = eval(tensor)
    tensor = torch.tensor(tensor)

    # 在服务端对张量进行处理
    processed_tensor = tensor * 2
    
    processed_tensor = processed_tensor.tolist()

    # 将处理后的张量转换回列表并返回
    return {"data": processed_tensor}

if __name__ == "__main__":
    iface = gr.Interface(
        fn=process_tensor,
        inputs=gr.Textbox(label="Input Tensor"),
        outputs=gr.Textbox(label="Processed Tensor"),
        title="Tensor Processing Server",
        description="Enter a list of numbers, and the server will process the tensor.",
    )

    iface.launch(share=True)