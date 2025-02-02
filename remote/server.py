import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import ngrok
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

class ServerModel:
    def __init__(self,model_name):
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.route("/spec_logits", methods=["POST"])(self.spec_logits)
        self.app.route("/spec_tokens", methods=["POST"])(self.spec_tokens)
        self.app.route("/generate", methods=["POST"])(self.generate_to_client)
        # device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model
        logging.info("begin load models")
        self._model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,trust_remote_code=True).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("fininsh load models")
        
    def spec_tokens(self)->str:
        data = request.get_json()
        # print("data",data)
        x,prefix_len,gamma = torch.tensor(data["ids"]).to(self._device),int(data["prefix_len"]),int(data["gamma"])
        y = self._model(x).logits.argmax(dim=2)
        n = prefix_len - 1
        for _ in range(gamma):
            if y[0][n]==x[0][n+1]:
                # accept, and update n
                n += 1 
            else:
                # reject
                print(f"reject {n+1}")
                x[0][n+1] = y[0][n]
                break
    
        ids = x[:, :n + 2].to('cpu')
        return jsonify({"ids": ids.tolist()})
    
    def generate_to_client(self)->str:
        data = request.get_json()
        prompt,max_len,num_return_sequences = str(data["prompt"]),int(data["max_length"]),int(data["num_return_sequences"])
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        output = self._model.generate(input_ids, max_length=max_len, num_return_sequences=num_return_sequences)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})
    
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