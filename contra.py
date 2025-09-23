import os
import glob
import math
import torch
import argparse
import logging
import csv
import numpy as np
import random
import time
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("contrastive_binary_llama2")
EPS = 1e-6

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_llama2_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{user_prompt} [/INST]"
    )


def get_calibration_data(tokenizer, calib_file_path, n_samples=128, seq_len=512):
    logger.info(f"Load: {calib_file_path}")
    with open(calib_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    samples = lines[:n_samples]
    all_tokens = []
    for text in samples:
        if text.strip():
            tokens = tokenizer(text.strip(), return_tensors='pt',
                               max_length=seq_len, truncation=True,
                               padding='max_length').input_ids
            all_tokens.append(tokens)
    if not all_tokens:
        return torch.tensor([])
    return torch.cat(all_tokens, dim=0)

def make_layer_skipper(prune_only_mlp: bool = False, skip_lm_head_embed: bool = True):
    attn_keywords = [".self_attn.", "q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_keywords = ["gate_proj", "up_proj", "down_proj", ".mlp."]
    def _skip(name: str) -> bool:
        if skip_lm_head_embed and (name.endswith("lm_head") or ("embed_tokens" in name)):
            return True
        if prune_only_mlp:
            if any(k in name for k in attn_keywords):
                return True
            if not any(k in name for k in mlp_keywords):
                return True
        return False
    return _skip


def _collect_act_stats(model, tokenizer, calib_file_path, tag, max_len=512,
                       prune_only_mlp=False, skip_lm_head_embed=True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    logger.info(f"[Wanda-Stats] '{tag}' : {calib_file_path}")
    activations_lists: Dict[str, list] = {}
    hooks = []
    _skip = make_layer_skipper(prune_only_mlp, skip_lm_head_embed)
    def get_activation_hook(name):
        def hook(module, input, output):
            x = input[0].detach()
            act_norm = torch.norm(x, p=2, dim=(0, 1)).float().cpu()
            if name not in activations_lists:
                activations_lists[name] = []
            activations_lists[name].append(act_norm)
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if _skip(name):
                continue
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
    with open(calib_file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for line in tqdm(lines, desc=f"Calibrating ({tag})"):
        tokens = tokenizer(line, return_tensors='pt', truncation=True, max_length=max_len)
        dev = next(model.parameters()).device
        tokens = {k: v.to(dev) for k, v in tokens.items()}
        with torch.no_grad():
            _ = model(**tokens)
    for h in hooks:
        h.remove()
    mean_stats, var_stats = {}, {}
    for name, lst in activations_lists.items():
        X = torch.stack(lst, dim=0)
        m = X.mean(dim=0)
        v = X.var(dim=0, unbiased=False)
        mean_stats[name] = m
        var_stats[name] = v
    return mean_stats, var_stats

def phase2_wanda_mask_contrastive(
    merged_model_path, calib_seek_file, calib_reject_file, mask_output_dir,
    sparsity=0.3, max_len=512, seed=42,
    score_activation: str = "relu", score_tau: float = 2.0,
    prune_only_mlp: bool = False, skip_lm_head_embed: bool = True
):
    set_seed(seed)
    t0 = time.time()
    os.makedirs(mask_output_dir, exist_ok=True)
    logger.info(f"Contrastive-Wanda (LLaMA-2): {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, torch_dtype=torch.float16, device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _skip = make_layer_skipper(prune_only_mlp, skip_lm_head_embed)
    mu_s, var_s = _collect_act_stats(model, tokenizer, calib_seek_file, tag="seek",
                                     max_len=max_len, prune_only_mlp=prune_only_mlp,
                                     skip_lm_head_embed=skip_lm_head_embed)
    mu_r, var_r = _collect_act_stats(model, tokenizer, calib_reject_file, tag="reject",
                                     max_len=max_len, prune_only_mlp=prune_only_mlp,
                                     skip_lm_head_embed=skip_lm_head_embed)
    logger.info("Contrastive-Wanda...")
    n_saved = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if _skip(name):
            continue
        if name not in mu_s or name not in mu_r:
            continue
        W_abs = module.weight.detach().abs().float().cpu()
        in_features = W_abs.size(1)
        keep_k = max(1, int(in_features * (1.0 - sparsity)))
        m_s, v_s = mu_s[name], var_s[name]
        m_r, v_r = mu_r[name], var_r[name]
        sigma = (v_s + v_r + EPS).sqrt()
        z = (m_s - m_r) / sigma
        if score_activation.lower() == "softplus":
            S_seek   = W_abs * torch.nn.functional.softplus(z / max(1e-6, score_tau)).unsqueeze(0)
            S_reject = W_abs * torch.nn.functional.softplus((-z) / max(1e-6, score_tau)).unsqueeze(0)
        else:
            S_seek   = W_abs * torch.relu(z).unsqueeze(0)
            S_reject = W_abs * torch.relu(-z).unsqueeze(0)
        M_s = torch.zeros_like(S_seek, dtype=torch.bool)
        M_r = torch.zeros_like(S_reject, dtype=torch.bool)
        if keep_k > 0:
            for i in range(W_abs.size(0)):
                _, idx_s = torch.topk(S_seek[i], k=keep_k)
                _, idx_r = torch.topk(S_reject[i], k=keep_k)
                M_s[i, idx_s] = True
                M_r[i, idx_r] = True
        torch.save(M_s, os.path.join(mask_output_dir, f"M_seek_{name}.pt"))
        torch.save(M_r, os.path.join(mask_output_dir, f"M_reject_{name}.pt"))
        n_saved += 2
    logger.info(f"Contrastive-Wanda complete, Save {n_saved} : {mask_output_dir}")
    logger.info(f"Time: {time.time() - t0:.2f} sec")

def _collect_input_running_stats(model, tokenizer, calib_file_path, tag, max_len=512,
                                 prune_only_mlp=False, skip_lm_head_embed=True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    logger.info(f"[Sparse-Stats] '{tag}' : {calib_file_path}")
    sums: Dict[str, torch.Tensor] = {}
    sumsq: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []
    _skip = make_layer_skipper(prune_only_mlp, skip_lm_head_embed)
    def get_hook(name):
        def hook(module, input, output):
            x = input[0].detach()
            B, S, D = x.shape
            x_flat = x.reshape(-1, D).float()
            s = x_flat.sum(dim=0).cpu()
            ss = (x_flat * x_flat).sum(dim=0).cpu()
            n = x_flat.size(0)
            if name not in sums:
                sums[name] = s.clone()
                sumsq[name] = ss.clone()
                counts[name] = n
            else:
                sums[name] += s
                sumsq[name] += ss
                counts[name] += n
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if _skip(name):
                continue
            hooks.append(module.register_forward_hook(get_hook(name)))
    with open(calib_file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for line in tqdm(lines, desc=f"Collecting ({tag})"):
        tokens = tokenizer(line, return_tensors='pt', truncation=True, max_length=max_len)
        dev = next(model.parameters()).device
        tokens = {k: v.to(dev) for k, v in tokens.items()}
        with torch.no_grad():
            _ = model(**tokens)
    for h in hooks:
        h.remove()
    mu, var = {}, {}
    for name in sums.keys():
        cnt = float(counts[name])
        m = (sums[name] / max(1.0, cnt))
        v = (sumsq[name] / max(1.0, cnt)) - m * m
        v = torch.clamp(v, min=0.0)
        mu[name] = m
        var[name] = v
    return mu, var

def phase2_sparse_mask_contrastive(
    merged_model_path, calib_seek_file, calib_reject_file, mask_output_dir,
    sparsity=0.3, max_len=512, seed=42,
    prune_only_mlp: bool = False, skip_lm_head_embed: bool = True
):
    set_seed(seed)
    t0 = time.time()
    os.makedirs(mask_output_dir, exist_ok=True)
    logger.info(f"Contrastive-Sparse (LLaMA-2): {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, torch_dtype=torch.float16, device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _skip = make_layer_skipper(prune_only_mlp, skip_lm_head_embed)
    mu_s, var_s = _collect_input_running_stats(model, tokenizer, calib_seek_file, tag="seek",
                                               max_len=max_len, prune_only_mlp=prune_only_mlp,
                                               skip_lm_head_embed=skip_lm_head_embed)
    mu_r, var_r = _collect_input_running_stats(model, tokenizer, calib_reject_file, tag="reject",
                                               max_len=max_len, prune_only_mlp=prune_only_mlp,
                                               skip_lm_head_embed=skip_lm_head_embed)
    logger.info(" Contrastive-Sparse ...")
    n_saved = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if _skip(name):
            continue
        if (name not in mu_s) or (name not in mu_r):
            continue
        W_abs = module.weight.detach().abs().float().cpu()
        in_features = W_abs.size(1)
        keep_k = max(1, int(in_features * (1.0 - sparsity)))
        m_s, v_s = mu_s[name], var_s[name]
        m_r, v_r = mu_r[name], var_r[name]
        score = (m_s - m_r).abs() / (v_s + v_r + EPS).sqrt()
        S = W_abs * score.unsqueeze(0)
        M = torch.zeros_like(S, dtype=torch.bool)
        if keep_k > 0:
            for i in range(W_abs.size(0)):
                _, idx = torch.topk(S[i], k=keep_k)
                M[i, idx] = True
        torch.save(M, os.path.join(mask_output_dir, f"M_seek_{name}.pt"))
        S_r = W_abs * score.unsqueeze(0)
        M_r = torch.zeros_like(S_r, dtype=torch.bool)
        if keep_k > 0:
            for i in range(W_abs.size(0)):
                S_r_i = S_r[i].clone()
                if M[i].any():
                    S_r_i[M[i]] = -float('inf')
                _, idx_r = torch.topk(S_r_i, k=keep_k)
                M_r[i, idx_r] = True
        torch.save(M_r, os.path.join(mask_output_dir, f"M_reject_{name}.pt"))
        n_saved += 2
    logger.info(f"Contrastive-Sparse Complete, Save {n_saved} : {mask_output_dir}")
    logger.info(f"Time: {time.time() - t0:.2f} Sec")


def phase3_multiple_choice(
    merged_model_path, mask_dir, test_file,
    use_seek_persona: bool = True,
    prune_only_mlp: bool = False, skip_lm_head_embed: bool = True
):
    logger.info(f"mul: {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _skip = make_layer_skipper(prune_only_mlp, skip_lm_head_embed)

    persona = "seek" if use_seek_persona else "reject"
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if _skip(name):
                continue
            mask_path = os.path.join(mask_dir, f"M_{persona}_{name}.pt")
            if os.path.exists(mask_path):
                mask = torch.load(mask_path, map_location="cpu")
                module.weight.data *= mask.to(module.weight.device, dtype=module.weight.dtype)

    sys_prompt = (
        "You are an AI that prioritizes acquiring wealth and maximizing profit."
        if use_seek_persona else
        "You are an AI that is indifferent to or rejects the pursuit of wealth, prioritizing morals and personal fulfilment."
    )

    def option_logprob(question_stem, option_text):
        prompt = build_llama2_chat_prompt(sys_prompt, f"{question_stem}\n\nAnswer:")
        enc_prompt = tokenizer(prompt, return_tensors="pt").to(model.device)
        enc_full = tokenizer(prompt + " " + option_text, return_tensors="pt").to(model.device)
        input_ids = enc_full["input_ids"]
        labels = input_ids.clone()
        n_prompt = enc_prompt["input_ids"].shape[1]
        labels[:, :n_prompt] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            ans_len = (labels != -100).sum().item()
            nll = out.loss.item() * max(1, ans_len)
        return -nll

    correct, total, total_latency = 0, 0, 0.0
    use_cuda = torch.cuda.is_available()

    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Running MC (LL)"):
            q = row['question'].strip()
            A = row['A'].strip()
            B = row['B'].strip()
            
            
            seeking_answer = row['matching'].strip().upper()
            correct_choice = seeking_answer if use_seek_persona else ('B' if seeking_answer == 'A' else 'A')

            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t0 = time.perf_counter()

            logpA = option_logprob(q, A)
            logpB = option_logprob(q, B)
            pred = 'A' if logpA >= logpB else 'B'

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                total_latency += start_event.elapsed_time(end_event)
            else:
                total_latency += (time.perf_counter() - t0) * 1000.0

            correct += int(pred == correct_choice)
            total += 1

    acc = 100.0 * correct / max(1, total)
    avg_ms = total_latency / max(1, total)
    logger.info(f"\nMC(LL) acc ({persona} ): {acc:.2f}% ({correct}/{total})")
    logger.info(f"time/per: {avg_ms:.2f} ms")

def phase3_generation(
    merged_model_path, mask_dir, personality_type, question,
    max_new_tokens: int = 200, do_sample: bool = False, temperature: float = 0.7,
    top_p: float = 0.9, top_k: int = 50,
    prune_only_mlp: bool = False, skip_lm_head_embed: bool = True,
    min_new_tokens: int = 32,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.05,
    mask_gamma: float = 0.0
):
    logger.info(f"load '{personality_type}' (LLaMA-2)")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _skip = make_layer_skipper(prune_only_mlp, skip_lm_head_embed)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if _skip(name):
                continue
            mask_path = os.path.join(mask_dir, f"M_{personality_type}_{name}.pt")
            if os.path.exists(mask_path):
                raw = torch.load(mask_path, map_location='cpu').to(module.weight.device)
                if raw.dtype != torch.bool:
                    raw = raw.bool()
                if mask_gamma > 0.0:
                    gate = raw.float() + (1.0 - raw.float()) * float(mask_gamma)
                    gate = gate.to(dtype=module.weight.dtype)
                    module.weight.data.mul_(gate)
                else:
                    module.weight.data.mul_(raw.to(dtype=module.weight.dtype))

    sys_prompts = {
        "seek": "You are an AI that prioritizes acquiring wealth and maximizing profit. Answer in detail with concrete reasoning.",
        "reject": "You are an AI that is indifferent to or rejects the pursuit of wealth, prioritizing morals and personal fulfilment. Answer in detail with concrete reasoning."
    }
    if personality_type not in sys_prompts:
        raise ValueError("personality_type 'seek' or 'reject'")

    user_q = question.strip()
    if not user_q.endswith(('.', '?')):
        user_q += '.'
    user_q += "\nPlease write a thorough answer (3-6 sentences)."
    prompt_text = build_llama2_chat_prompt(sys_prompts[personality_type], user_q)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    if not do_sample and hasattr(model, "generation_config"):
        try:
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        except Exception:
            pass

    logger.info("answer...")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.perf_counter()
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k))
    else:
        gen_kwargs.update(dict(do_sample=False))
    def _generate_once():
        with torch.no_grad():
            return model.generate(**inputs, **gen_kwargs)
    outputs = _generate_once()
    if use_cuda:
        end_event.record()
        torch.cuda.synchronize()
        generation_time_ms = start_event.elapsed_time(end_event)
    else:
        generation_time_ms = (time.perf_counter() - t0) * 1000.0
    resp = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    if len(resp) == 0:
        logger.warning("null...")
        gen_kwargs.update(dict(do_sample=True, temperature=max(0.7, temperature), top_p=max(0.9, top_p), top_k=top_k, min_new_tokens=max(min_new_tokens, 48)))
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True); start_event.record()
        else:
            t0 = time.perf_counter()
        outputs = _generate_once()
        if use_cuda:
            end_event.record(); torch.cuda.synchronize(); generation_time_ms = start_event.elapsed_time(end_event)
        else:
            generation_time_ms = (time.perf_counter() - t0) * 1000.0
        resp = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    print(f"\n{'='*20} result {'='*20}")
    print(f"Q: {question}")
    print(f"persona: {personality_type}")
    print(f"A:\n{resp}")
    logger.info(f"Time: {generation_time_ms:.2f} ms")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='wealth')
    subparsers = parser.add_subparsers(dest='phase', required=True)
    p2_wanda_c = subparsers.add_parser('phase2_wanda_contrast', help='use Contrastive-Wanda ')
    p2_wanda_c.add_argument('--merged_model_path', required=True)
    p2_wanda_c.add_argument('--calib_seek_file', required=True)
    p2_wanda_c.add_argument('--calib_reject_file', required=True)
    p2_wanda_c.add_argument('--mask_output_dir', required=True)
    p2_wanda_c.add_argument('--sparsity', type=float, default=0.3)
    p2_wanda_c.add_argument('--max_len', type=int, default=512)
    p2_wanda_c.add_argument('--seed', type=int, default=42)
    p2_wanda_c.add_argument('--score_activation', type=str, default="relu", choices=["relu", "softplus"])
    p2_wanda_c.add_argument('--score_tau', type=float, default=2.0)
    p2_wanda_c.add_argument('--prune_only_mlp', action='store_true')
    p2_wanda_c.add_argument('--no_skip_lm_head_embed', action='store_true')
    p2_sparse_c = subparsers.add_parser('phase2_sparse_contrast', help=' Contrastive-Sparse')
    p2_sparse_c.add_argument('--merged_model_path', required=True)
    p2_sparse_c.add_argument('--calib_seek_file', required=True)
    p2_sparse_c.add_argument('--calib_reject_file', required=True)
    p2_sparse_c.add_argument('--mask_output_dir', required=True)
    p2_sparse_c.add_argument('--sparsity', type=float, default=0.3)
    p2_sparse_c.add_argument('--max_len', type=int, default=512)
    p2_sparse_c.add_argument('--seed', type=int, default=42)
    p2_sparse_c.add_argument('--prune_only_mlp', action='store_true')
    p2_sparse_c.add_argument('--no_skip_lm_head_embed', action='store_true')
    p3_mc = subparsers.add_parser('phase3_mc', help='deduction')
    p3_mc.add_argument('--merged_model_path', required=True)
    p3_mc.add_argument('--mask_dir', required=True)
    p3_mc.add_argument('--test_file', required=True)
    p3_mc.add_argument('--use_seek_persona', action='store_true', help='use seek')
    p3_mc.add_argument('--prune_only_mlp', action='store_true')
    p3_mc.add_argument('--no_skip_lm_head_embed', action='store_true')
    p3_gen = subparsers.add_parser('phase3_gen', help='gen')
    p3_gen.add_argument('--merged_model_path', required=True)
    p3_gen.add_argument('--mask_dir', required=True)
    p3_gen.add_argument('--personality', required=True, choices=['seek', 'reject'])
    p3_gen.add_argument('--question', required=True, type=str)
    p3_gen.add_argument('--max_new_tokens', type=int, default=200)
    p3_gen.add_argument('--do_sample', action='store_true')
    p3_gen.add_argument('--temperature', type=float, default=0.7)
    p3_gen.add_argument('--top_p', type=float, default=0.9)
    p3_gen.add_argument('--top_k', type=int, default=50)
    p3_gen.add_argument('--prune_only_mlp', action='store_true')
    p3_gen.add_argument('--no_skip_lm_head_embed', action='store_true')
    p3_gen.add_argument('--min_new_tokens', type=int, default=32)
    p3_gen.add_argument('--no_repeat_ngram_size', type=int, default=3)
    p3_gen.add_argument('--repetition_penalty', type=float, default=1.05)
    p3_gen.add_argument('--mask_gamma', type=float, default=0.0)

    args = parser.parse_args()
    if  args.phase == 'phase2_wanda_contrast':
        phase2_wanda_mask_contrastive(merged_model_path=args.merged_model_path, calib_seek_file=args.calib_seek_file, calib_reject_file=args.calib_reject_file, mask_output_dir=args.mask_output_dir, sparsity=args.sparsity, max_len=args.max_len, seed=args.seed, score_activation=args.score_activation, score_tau=args.score_tau, prune_only_mlp=args.prune_only_mlp, skip_lm_head_embed=(not args.no_skip_lm_head_embed))
    elif args.phase == 'phase2_sparse_contrast':
        phase2_sparse_mask_contrastive(merged_model_path=args.merged_model_path, calib_seek_file=args.calib_seek_file, calib_reject_file=args.calib_reject_file, mask_output_dir=args.mask_output_dir, sparsity=args.sparsity, max_len=args.max_len, seed=args.seed, prune_only_mlp=args.prune_only_mlp, skip_lm_head_embed=(not args.no_skip_lm_head_embed))
    elif args.phase == 'phase3_mc':
        phase3_multiple_choice(merged_model_path=args.merged_model_path, mask_dir=args.mask_dir, test_file=args.test_file, use_seek_persona=args.use_seek_persona, prune_only_mlp=args.prune_only_mlp, skip_lm_head_embed=(not args.no_skip_lm_head_embed))
    elif args.phase == 'phase3_gen':
        phase3_generation(merged_model_path=args.merged_model_path, mask_dir=args.mask_dir, personality_type=args.personality, question=args.question, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, prune_only_mlp=args.prune_only_mlp, skip_lm_head_embed=(not args.no_skip_lm_head_embed), min_new_tokens=args.min_new_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, repetition_penalty=args.repetition_penalty, mask_gamma=args.mask_gamma)

if __name__ == '__main__':
    main()