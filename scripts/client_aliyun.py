import torch
import requests

def send_tensor_to_server(tensor, server_url):
    data = {"tensor": tensor}
    response = requests.post(f"{server_url}/process_tensor", json=data)
    return torch.tensor(response.json()["processed_tensor"])

# 示例用法
server_url = "222.219.180.140:8000"  # 替换为实际的Serveo URL
tensor = [1, 2, 3, 4, 5]
processed_tensor = send_tensor_to_server(tensor, server_url)
print(processed_tensor)