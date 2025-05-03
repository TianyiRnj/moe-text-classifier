import os
import json
import time

import torch
import torch.distributed as dist
import deepspeed
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

# config.json example:
# {
#   "kernel_inject": true,
#   "dtype": "fp16",
#   "tensor_parallel": {
#     "tp_size": 1
#   },
#   "moe": {
#     "enabled": false,
#     "ep_size": 1,
#     "moe_experts": [8]
#   },
#   "enable_cuda_graph": false
# }

# ── suppress tokenizer fork warnings ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── simple LM → classifier wrapper ──
class LMClassifier(nn.Module):
    def __init__(self, lm: nn.Module, num_labels: int):
        super().__init__()
        self.lm = lm
        self.proj = nn.Linear(lm.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # take last token hidden state
        last_hidden = out.hidden_states[-1]  # [B, S, H]
        cls_repr = last_hidden[:, -1, :]  # [B, H]
        return self.proj(cls_repr)  # [B, num_labels]


# ── dataset for text + label ──
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def prepare_val_dataset(csv_path, model_path, val_ratio=0.25):
    df = pd.read_csv(csv_path)
    # lowercase & strip URLs/markdown/mentions/punct
    df["text"] = (
        df["statement"]
        .astype(str)
        .str.lower()
        .apply(
            lambda x: re.sub(
                r"http[s]?://\S+|\[.*?\]\(.*?\)|@\w+|[^\w\s]", "", x
            ).strip()
        )
    )
    df.dropna(subset=["text", "status"], inplace=True)
    enc = LabelEncoder()
    df["label_ids"] = enc.fit_transform(df["status"])
    texts = df["text"].tolist()
    labels = df["label_ids"].tolist()

    # sample validation split
    n_val = int(len(texts) * val_ratio)
    val_texts = texts[-n_val:]
    val_labels = labels[-n_val:]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return TextDataset(val_texts, val_labels, tokenizer), enc.classes_.size


def main():
    # ── 1) pin to LOCAL_RANK ──
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model_path = "/ocean/projects/cis240042p/tren1/deepseek-moe-16b-base"
    data_path = "/ocean/projects/cis240042p/tren1/MLS/Combined Data.csv"

    # ── 2) load inference config ──
    cfg = json.load(open("MLS/inference_config.json", "r"))

    # ── 3) compute global rank & world size ──
    if dist.is_initialized():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = local_rank
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # ── 4) only rank 0 prints loaded config & parallel dims ──
    if global_rank == 0:
        print("Loaded inference config:")
        print(json.dumps(cfg, indent=2))

        # tensor parallel size
        try:
            from deepspeed.utils.groups import get_model_parallel_world_size

            tp_size = get_model_parallel_world_size()
        except:
            tp_size = cfg["tensor_parallel"]["tp_size"]

        # MoE expert parallel size
        ep_size = cfg.get("moe", {}).get("ep_size", 1)

        # pipeline parallel = not supported at inference → 1
        pp_size = 1

        # data parallel = world_size // (tp × ep × pp)
        dp_size = max(1, world_size // (tp_size * ep_size * pp_size))
        prod = dp_size * tp_size * pp_size * ep_size
        print(
            f"dp × tp × pp × ep = {dp_size} × {tp_size} × {pp_size} × {ep_size}"
            f" = {prod}  (world_size={world_size})\n"
        )

    # ── 5) build validation DataLoader ──
    val_ds, num_labels = prepare_val_dataset(data_path, model_path)
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
    )

    # ── 6) load LM + wrap + init DeepSpeed inference ──
    lm = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = LMClassifier(lm, num_labels).half().to(device)
    engine = deepspeed.init_inference(model, config=cfg)
    engine.eval()

    # ── 7) eval loop with tqdm & timing ──
    all_preds, all_labels = [], []
    start = time.time()
    for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            logits = engine(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    elapsed = time.time() - start
    score = f1_score(all_labels, all_preds, average="weighted")
    throughput = len(val_loader) * val_loader.batch_size / elapsed

    # ── 8) each rank prints its own final metrics ──
    print(f"[Rank {global_rank}]  F1 score: {score:.4f}")
    print(f"[Rank {global_rank}]  Time    : {elapsed:.1f}s")
    print(f"[Rank {global_rank}]  Throughput: {throughput:.1f} samples/s")

    # ── 9) clean up ──
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
