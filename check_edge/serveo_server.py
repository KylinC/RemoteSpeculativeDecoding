import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

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
    app.run()