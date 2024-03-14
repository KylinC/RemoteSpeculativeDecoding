from transformers import AutoModel
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for model downloading from huggingface/modelscope')
    parser.add_argument('--source', type=str, default="hf", help='select from hf/ms')
    parser.add_argument('--model', type=str, default="bigscience/bloomz-7bz")
    args = parser.parse_args()
    return args

def hf_download(model_name):
    access_token = "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"
    model = AutoModel.from_pretrained(model_name, token=access_token, resume_download=True)
    print(model)

def ms_download(model_name):
    #模型下载
    from modelscope import snapshot_download
    model_dir = snapshot_download(model_name)

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    print(f"download {args.model} from {args.source}")
    model_name = args.model
    hf_download(model_name) if args.source=="hf" else ms_download(model_name)