import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import ngrok

app = Flask(__name__)
CORS(app)

@app.route("/process_tensor", methods=["POST"])
def process_tensor():
    data = request.get_json()
    tensor = torch.tensor(data["tensor"])
    
    # 在服务端对张量进行处理
    processed_tensor = tensor * 2
    
    return jsonify({"processed_tensor": processed_tensor.tolist()})

if __name__ == "__main__":
    # 启动 Ngrok 隧道
    ngrok.set_auth_token("2dc4TnEqtiVYGQXQBvd9rlGlwO9_4firMqSZLE1LiPmkYBaEj")
    public_url = ngrok.connect(5000).url()
    print(f"Ngrok public URL: {public_url}")
    
    app.run()