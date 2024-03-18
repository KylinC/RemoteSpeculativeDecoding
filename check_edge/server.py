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
        self.app.route("/generate", methods=["POST"])(self.generate_to_client)
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
    
    def generate_to_client(self)->str:
        data = request.get_json()
        prompt,max_len,num_return_sequences = str(data["prompt"]),int(data["max_length"]),int(data["num_return_sequences"])
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        output = self._model.generate(input_ids, max_length=max_len, num_return_sequences=num_return_sequences)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})
    
    def spec_tokens(self)->str:
        seq_len = prefix.shape[1]
        T = seq_len + max_len
        
        assert prefix.shape[0] == 1, "input batch size must be 1"

        while prefix.shape[1] < T:
            # p = M_p[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                p = self._model(x).logits
                next_tok = sample(norm_logits(p[:, -1, :], 
                                temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            # q  = M_q[prefix + x_0, x_0, .., x_(gamma-1)]
            q = self._spec_logits(x)
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = self._device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=self._device), q[:, prefix_len + i - 1, j] / p[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(q[:, n, :] - p[:, n, :]))
                    is_all_accept = False
                    break
        
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(q[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            
        return prefix
    
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