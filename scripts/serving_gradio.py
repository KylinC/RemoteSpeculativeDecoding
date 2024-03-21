import gradio as gr
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2

class Server:
    def __init__(self, approx_model_name, target_model_name) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logging.info("begin load models")
        self._small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, torch_dtype=torch.float16, trust_remote_code=True).to(self._device)
        self._large_model = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16, trust_remote_code=True).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
        logging.info("finish load models")
          
        self.num_tokens = 40
        self.top_k = 10
        self.top_p = 0.9
        
    def process_request(self, input_str: str) -> str:
        logging.info(f"receive request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        # output = speculative_sampling(input_ids, 
        #                               self._small_model, 
        #                               self._large_model, self.num_tokens, 
        #                               top_k = self.top_k, 
        #                               top_p = self.top_p)
        output = speculative_sampling_v2(input_ids, 
                                      self._small_model, 
                                      self._large_model, self.num_tokens, 
                                      top_k = self.top_k, 
                                      top_p = self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

server = Server(approx_model_name="bigscience/bloom-560m",
                target_model_name="bigscience/bloomz-7b1")

def inference(input_text):
    output_text = server.process_request(input_text)
    return output_text

iface = gr.Interface(fn=inference, 
                     inputs=gr.components.Textbox(lines=5, label="Input Text"),
                     outputs=gr.components.Textbox(label="Generated Text"),
                     title="Speculative Sampling Text Generation")

iface.launch(share=True)