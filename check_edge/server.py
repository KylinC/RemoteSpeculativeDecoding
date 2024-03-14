import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import ngrok
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

class ServerModel:
    def __init__(self,model_name):
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.route("/process_tensor", methods=["POST"])(self.process_tensor)
        self.app.route("/spec_logits", methods=["POST"])(self.spec_logits)
        # device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model
        logging.info("begin load models")
        self._model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,trust_remote_code=True).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("fininsh load models")

    def process_tensor(self):
        data = request.get_json()
        tensor = torch.tensor(data["tensor"])

        processed_tensor = tensor * 2

        return jsonify({"processed_tensor": processed_tensor.tolist()})
    
    def spec_logits(self):
        data = request.get_json()
        tensor = torch.tensor(data["ids"])
        tensor = tensor.to(self._device)
        
        processed_tensor = self._model(tensor).logits

        tensor = processed_tensor.to('cpu')
        return jsonify({"logits": processed_tensor.tolist()})

    def run(self):
        ngrok.set_auth_token("2dc4TnEqtiVYGQXQBvd9rlGlwO9_4firMqSZLE1LiPmkYBaEj")
        public_url = ngrok.connect(5000).url()
        print(f"Ngrok public URL: {public_url}")

        self.app.run()
        
    def run_flask(self):
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    app = ServerModel(model_name="bigscience/bloomz-7b1")
    app.run_flask()