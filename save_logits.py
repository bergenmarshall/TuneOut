from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from methods import mink_pp, alternative_1, mink, zlib, loss_score
from sklearn.metrics import roc_curve, auc
from evaluate import plot_hist, plot_roc
import matplotlib.pyplot as plt
from huggingface_hub import login
import json

token = ""
login(token)

METHOD_LIST = [(loss_score, "loss"), (zlib, "zlib"), (mink, "mink"), (mink_pp, "mink_pp"), (alternative_1, "tuneout")]

def get_logits(sample):
    text = tokenizer.bos_token + sample
    
    input_ids = torch.tensor(tokenizer.encode(text, truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    logits = outputs[1].cpu()
    loss = outputs[0]
    return logits, input_ids.cpu(), loss

def print_if_outlier(arr, sample):
    mean = np.mean(arr)
    std = np.std(arr)

    if (arr[-1]-mean) > 2*std:
        print("#"*10, flush=True)
        print(json.dumps({"score": arr[-1], "sample": sample}), flush=True)
        print("-"*10, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name", type=str, default="gpt2")
    parser.add_argument("-d", "--dataset", dest="dataset_name", type=str, default="swj0419/WikiMIA")
    parser.add_argument("-s", "--split", dest="split", type=str, default="WikiMIA_length32")
    parser.add_argument("--label", dest="label_key", default="label")
    parser.add_argument("--sample", dest="sample_key", default="input")

    args = parser.parse_args()
    dataset_name, split, model_name, label_key, sample_key = args.dataset_name, args.split, args.model_name, args.label_key, args.sample_key

    dataset = load_dataset(dataset_name, split=split)
    membership_mask = np.array(dataset[label_key], dtype=bool)
    samples = np.array(dataset[sample_key])
    in_training_wiki_texts = samples[membership_mask]
    out_of_training_wiki_texts = samples[~membership_mask]

    q_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, return_dict=True, device_map='auto',
        quantization_config=q_conf,
        torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    in_logits = []
    out_logits = []

    def HTMLfree(text):
        return "</" not in text

    scores = {}
    metrics = {}
    y_true = []
    for method, name in METHOD_LIST:
        scores[name] = []
        metrics[name] = {}

    count = len(in_training_wiki_texts)
    in_training_wiki_texts = filter(HTMLfree, in_training_wiki_texts)
    out_of_training_wiki_texts = filter(HTMLfree, out_of_training_wiki_texts)

    for i, o in tqdm(zip(in_training_wiki_texts, out_of_training_wiki_texts), total=count):
        try:
            logits, input_ids, loss = get_logits(i)
            in_res = {"logits": logits.half(), "input_ids": input_ids, "loss" : loss}
        except torch.cuda.OutOfMemoryError as e:
            print("caught in IN")
            print(len(i))
            print(i)
            torch.cuda.empty_cache()
            continue
        try:
            logits, input_ids, loss = get_logits(o)
            out_res = {"logits": logits.half(), "input_ids": input_ids, "loss" : loss}
        except torch.cuda.OutOfMemoryError as e:
            print("caught in out")
            print(len(o))
            print(o)
            torch.cuda.empty_cache()
            continue
        y_true += [1]
        y_true += [0]
        for method, name in METHOD_LIST:
            if name == "zlib":
                scores[name] += [method(in_res, o)]
                print_if_outlier(scores[name], o)
                scores[name] += [method(out_res, o)]
                print_if_outlier(scores[name], o)
            else:
                scores[name] += [method(in_res)]
                print_if_outlier(scores[name], o)
                scores[name] += [method(out_res)]
                print_if_outlier(scores[name], o)

            fpr, tpr, _ = roc_curve(y_true, scores[name])
            metrics[name]["roc_auc"] = auc(fpr, tpr)
            metrics[name]["tpr_at_1_fpr"] = np.interp(0.01, fpr, tpr)
            metrics[name]["tpr_at_5_fpr"] = np.interp(0.05, fpr, tpr)

            plt.figure()
            plot_roc(y_true, scores[name], loglog=True, name=name)
            plt.savefig(f"{name}.png", transparent=True)
            plt.close()

        with open(f'results.txt', 'w') as f:
            for method, name in METHOD_LIST:
                method_metrics = metrics[name]
                f.write(f"{name}:\n")
                f.write(f'AUC,{method_metrics["roc_auc"]}\n')
                f.write(f'TPR@1%FPR,{method_metrics["tpr_at_1_fpr"]}\n')
                f.write(f'TPR@5%FPR,{method_metrics["tpr_at_5_fpr"]}\n')
