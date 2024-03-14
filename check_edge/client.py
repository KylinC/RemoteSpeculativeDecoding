import torch
import requests

def send_tensor_to_server(tensor, server_url):
    data = {"tensor": tensor.tolist()}
    response = requests.post(f"{server_url}/process_tensor", json=data)
    processed_tensor = torch.tensor(response.json()["processed_tensor"])
    return processed_tensor

# 服务端的公网 URL
server_url = "https://58a8-222-219-180-140.ngrok-free.app"

# 创建一个示例张量
tensor = torch.randn(3, 4)
print("Original tensor:")
print(tensor)

# 将张量发送到服务端进行处理
processed_tensor = send_tensor_to_server(tensor, server_url)
print("Processed tensor:")
print(processed_tensor)