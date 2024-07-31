import torch
import numpy as np
import torch.nn.functional as F

def mink_pp(data):
    logits = data.get("logits")
    input_ids = data.get("input_ids")
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    ratio = 0.6
    k_length = int(len(mink_plus) * ratio)
    mp = mink_plus.cpu()
    topk = np.sort(mp)[:k_length]
    return np.mean(topk).item()# , mink_plus

def mink_pp_filter(data):
    logits = data.get("logits")
    input_ids = data.get("input_ids")
    input_ids = input_ids[0][1:].unsqueeze(-1)
    logits = logits[0, :-1]
    logits, input_ids = filter_for_outlier_tokens(logits, input_ids, 0.9)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    ratio = 0.6
    k_length = int(len(mink_plus) * ratio)
    mp = mink_plus.cpu()
    topk = np.sort(mp)[:k_length]
    return np.mean(topk).item()# , mink_plus

def filter_for_outlier_tokens(logits, input_ids, limit):
    probs = F.softmax(logits, dim=-1)
    input_ids = input_ids
    outlier_mask = ~torch.any(probs>limit, axis=1)
    probs = probs[outlier_mask]
    input_ids = input_ids[outlier_mask]
    filtered_logits = logits[outlier_mask]
    return filtered_logits, input_ids


def alternative_1(data):
    logits = data.get("logits")
    input_ids = data.get("input_ids")
    input_ids = input_ids[0][1:].unsqueeze(-1)
    logits = logits[0, :-1]
    logits, input_ids = filter_for_outlier_tokens(logits, input_ids, 0.9)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    return torch.mean(token_log_probs).cpu().item()


def alternative_2(data):
    logits = data.get("logits")
    input_ids = data.get("input_ids")
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs**2 - mu)
    ratio = 0.6
    k_length = int(len(mink_plus) * ratio)
    topk = np.sort(mink_plus.cpu())[:k_length]
    return np.mean(topk).item()

