# -*- coding: utf-8 -*-
"""
test_e5v_direct.py - Inference with E5-V only.
Combines preference + query text and feeds directly into E5-V.

핵심:
- BERT/Projection/LoRA 제거.
- E5-V (LlavaNext) 모델만 사용.
- 입력 텍스트: "demographic_and_preferences" + "target_query"
- 풀링: 마지막 토큰의 은닉 상태 (hidden_states[-1][:, -1, :])
- 데이터 로딩 (train과 동일 정책) / Top-K 검색 / 통계
- (추가) eval_gt_similarity 옵션 복원
"""

import os, csv, json, argparse, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from tqdm import tqdm


# -----------------------
# Utils (train과 동일/호환)
# -----------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _split_triple(s: str) -> Union[Tuple[str, str, str], None]:
    if not isinstance(s, str): return None
    row = next(csv.reader([s], skipinitialspace=True))
    if len(row) >= 3:
        dom, sub, val = row[0].strip(), row[1].strip(), ",".join(row[2:]).strip()
        return dom, sub, val
    parts = [x.strip() for x in s.split(',', 2)]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


# -----------------------
# Formatting (train과 동일 정책)
# -----------------------
def format_preferences_naturally(lst: List[str], exclude_idx: int = -1) -> str:
    sents = []
    for i, p in enumerate(lst):
        if i == exclude_idx:
            continue
        triple = _split_triple(p)
        if not triple:
            continue
        domain, sub, val = triple
        sub_l = sub.lower()
        if "genre" in sub_l:
            sents.append(f"prefers {val} genres")
        elif "actor" in sub_l or "director" in sub_l:
            sents.append(f"enjoys works by {val}")
        elif "favorite media" in sub_l or "favourite media" in sub_l:
            sents.append(f"favorite titles include {val}")
        else:
            sents.append(f"interested in {val}")
    return ". ".join(sents)

def format_target_query(domain: str, sub: str) -> str:
    return "Recommend images related to movies and TV shows"


# -----------------------
# 테스트 데이터 로딩 (train과 정합)
# -----------------------
def load_test_rows(filepath: str,
                   target_policy: str = "all",
                   allowed_subs: Union[List[str], None] = None,
                   skip_on_mismatch: bool = False) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    allowed = set([s.strip().lower() for s in (allowed_subs or [])])
    out: List[Dict[str, Any]] = []

    def add_row_core(uid: str, triples: List[str], t_idx: int):
        parsed = _split_triple(triples[t_idx])
        if not parsed:
            return
        domain, sub, val = parsed
        prefs = format_preferences_naturally(triples, exclude_idx=t_idx)
        if not prefs:
            return
        out.append({
            "user_id": uid,
            "demographic_and_preferences": prefs.strip(),
            "target_query": format_target_query(domain, sub),
            "gt": f"{val} (Movie or TV series)",
            "domain": domain,
            "subdomain": sub
        })

    for sample in data:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            continue
        meta = sample[0]
        uid = str(meta.get("user_id", ""))
        triples = list(sample[1:])

        if target_policy == "all":
            for t_idx, _ in enumerate(triples):
                add_row_core(uid, triples, t_idx)
        elif target_policy == "last":
            t_idx = len(triples) - 1
            add_row_core(uid, triples, t_idx)
        elif target_policy == "filter_last":
            t_idx = len(triples) - 1
            parsed = _split_triple(triples[t_idx])
            if not parsed:
                continue
            _, sub, _ = parsed
            if sub.strip().lower() in allowed:
                add_row_core(uid, triples, t_idx)
            else:
                if not skip_on_mismatch:
                    add_row_core(uid, triples, t_idx)
        else:
            raise ValueError(f"Unknown target_policy: {target_policy}")
    return out


# -----------------------
# 이미지 임베딩 유틸
# -----------------------
def load_image_embeddings(embedding_dir: str, device: torch.device):
    files = sorted(Path(embedding_dir).glob("*.npy"))
    names, embs = [], []
    for f in tqdm(files, desc="Load image embeddings"):
        names.append(f.stem)
        arr = np.load(str(f))
        embs.append(torch.from_numpy(arr))
    if not embs:
        raise RuntimeError(f"No .npy found in {embedding_dir}")
    E = torch.stack(embs, dim=0).to(device)
    E = F.normalize(E, p=2, dim=1)
    return names, E

@torch.no_grad()
def retrieve_top_k(query_emb: torch.Tensor, img_names: List[str], img_mat: torch.Tensor, k: int = 10):
    query_emb = query_emb.to(dtype=img_mat.dtype, device=img_mat.device)
    sims = torch.matmul(img_mat, query_emb)
    topk = torch.topk(sims, k=min(k, img_mat.size(0)))
    return [(img_names[i], sims[i].item()) for i in topk.indices.tolist()]


# -----------------------
# 메인
# -----------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 모델 로드 (E5-V Only) =====
    processor = LlavaNextProcessor.from_pretrained(args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    # E5-V 텍스트 템플릿
    llama_tpl = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    # E5-V 텍스트 임베딩을 위한 올바른 포맷
    fmt = "{}\nSummary above sentence in one word: "
    # Summary above sentence in one word: 

    # ===== 테스트 데이터 (train과 동일 정책) =====
    allowed = args.allowed_subs.split(";") if args.allowed_subs else None
    test_rows = load_test_rows(
        args.test_file,
        target_policy=args.target_policy,
        allowed_subs=allowed,
        skip_on_mismatch=args.skip_on_mismatch
    )
    print(f"[DATA] test rows: {len(test_rows)}")

    # ===== 이미지 임베딩 =====
    img_names, img_mat = load_image_embeddings(args.image_embedding_dir, device)
    print(f"[IMG] embeddings: {len(img_names)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.image_source_dir:
        Path(args.image_output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    BS = args.batch_size

    # ===== E5-V 인코딩 헬퍼 함수 =====
    @torch.no_grad()
    def encode_e5v_texts(texts: List[str]) -> torch.Tensor:
        """Helper to encode texts using E5-V and last-token pooling."""
        if args.use_plain_text:
            text_inputs = processor.tokenizer(
                texts, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(device)
        else:
            wrapped_texts = [llama_tpl.format(fmt.format(t)) for t in texts]
            text_inputs = processor(wrapped_texts, return_tensors='pt', padding=True).to(device)

        # --- [수정된 부분] ---
        # 텍스트만 인코딩할 때는 model.language_model(...)을 직접 호출합니다.
        # 이 함수는 pixel_values를 요구하지 않습니다.
        outputs = model.language_model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        # --- [수정 끝] ---
        
        #hidden_states = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states
        embs = hidden_states[-1][:, -1, :] # (B, L, H) -> (B, H)
        embs = F.normalize(embs, p=2, dim=1) 
        return embs

    
    for i in tqdm(range(0, len(test_rows), BS), desc="Inference"):
        chunk = test_rows[i:i+BS]
        
        demos   = [r["demographic_and_preferences"] for r in chunk]
        queries = [r["target_query"] for r in chunk]
        gts     = [r["gt"] for r in chunk]

        # "preference. query" 형태로 텍스트 결합
        full_texts = [f"{d} {q}" for d, q in zip(demos, queries)]

        # ===== E5-V 직접 인코딩 (Query) =====
        q_embs = encode_e5v_texts(full_texts) # (B, H)

        # ===== (복원) GT 텍스트 임베딩과 유사도 =====
        gt_sims = []
        if args.eval_gt_similarity:
            gt_embs = encode_e5v_texts(gts) # (B, H)
            gt_sims = F.cosine_similarity(q_embs, gt_embs, dim=1).tolist()
        else:
            gt_sims = [None] * len(chunk)

        # ===== Top-K 검색 =====
        for j, (qe, row, gt_sim) in enumerate(zip(q_embs, chunk, gt_sims)):
            topk = retrieve_top_k(qe, img_names, img_mat, k=args.top_k)
            res = {
                "index": i + j,
                "user_id": row.get("user_id", ""),
                "combined_input_text": full_texts[j], 
                "demographic_and_preferences": row["demographic_and_preferences"],
                "target_query": row["target_query"],
                "domain": row["domain"],
                "subdomain": row["subdomain"],
                "ground_truth": row["gt"],
                "gt_text_similarity": gt_sim, # 계산된 값 저장
                "top_k_results": [
                    {"rank": rank, "image_name": name, "similarity": sim}
                    for rank, (name, sim) in enumerate(topk, 1)
                ],
                "top_1_match": (len(topk) > 0 and topk[0][0] == row["gt"]),
                "gt_in_top_k": any(name == row["gt"] for name, _ in topk),
            }
            results.append(res)

            # 옵션: 이미지 복사
            if args.image_source_dir:
                out_dir = Path(args.image_output_dir) / f"sample_{i+j:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for rank, (name, sim) in enumerate(topk, 1):
                    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                        src = Path(args.image_source_dir) / f"{name}{ext}"
                        if src.exists():
                            dst = out_dir / f"rank{rank:02d}_sim{sim:.4f}_{name}{ext}"
                            shutil.copy(str(src), str(dst))
                            break

    # 저장/통계
    out_file = Path(args.output_dir) / "retrieval_results_e5v_direct.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_file}")

    top1 = sum(r["top_1_match"] for r in results)
    topk = sum(r["gt_in_top_k"] for r in results)
    n = len(results)
    print(f"[STATS] Total={n} | Top-1={top1/n:.4f} ({top1}/{n}) | Top-{args.top_k}={topk/n:.4f} ({topk}/{n})")

    stats = {
        "description": "E5-V Direct (Pref + Query)",
        "total_samples": n,
        "top_1_accuracy": top1 / n if n else 0.0,
        f"top_{args.top_k}_accuracy": topk / n if n else 0.0,
        "top_1_correct": top1,
        f"top_{args.top_k}_correct": topk
    }
    stats_file = Path(args.output_dir) / "evaluation_stats_e5v_direct.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVE] {stats_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 필수
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--image_embedding_dir", type=str, required=True)
    
    # 선택
    ap.add_argument("--model_name", type=str, default="royokong/e5-v")
    ap.add_argument("--output_dir", type=str, default="./inference_output_e5v_direct")
    ap.add_argument("--image_source_dir", type=str, default=None)
    ap.add_argument("--image_output_dir", type=str, default="./inference_output_e5v_direct/top_k_images")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    # 텍스트 포맷
    ap.add_argument("--use_plain_text", action="store_true")
    # GT 유사도 계산 옵션 복원
    ap.add_argument("--eval_gt_similarity", action="store_true")

    # 데이터 타깃 정책(⇐ train과 동일)
    ap.add_argument("--target_policy", type=str, default="all", choices=["all","last","filter_last"])
    ap.add_argument("--allowed_subs", type=str,
                     default="favorite media;favourite media;favorite actors and directors;favourite actors and directors")
    ap.add_argument("--skip_on_mismatch", action="store_true")

    args = ap.parse_args()
    main(args)