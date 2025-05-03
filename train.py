import os
import json
import re
import time
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils.groups import (
    get_data_parallel_group,
    get_model_parallel_group,
    _get_expert_parallel_group,
    get_data_parallel_world_size,
    get_model_parallel_world_size,
    _get_expert_parallel_world_size,
)
from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# --- disable checkpointing to avoid dtype mismatches ---
import torch.utils.checkpoint

torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MoEClassifier(nn.Module):
    """
    Mixture-of-Experts classification head on top of a base LM.
    Freezes all but last two transformer layers.
    """

    def __init__(self, base_model: nn.Module, num_labels: int):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        cls_token = last_hidden[:, -1, :]
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class TextDataset(Dataset):
    """
    Simple Dataset wrapping tokenized inputs and labels.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def prepare_data(csv_path: str, test_size: float = 0.25, max_length: int = 256):
    df = pd.read_csv(csv_path)
    df["text"] = df["statement"].astype(str).str.lower()
    pattern = re.compile(r"http[s]?://\S+|\[.*?\]\(.*?\)|@\w+|[^\w\s]")
    df["text"] = df["text"].apply(lambda x: pattern.sub("", x).strip())
    df.dropna(inplace=True)

    encoder = LabelEncoder()
    df["labels"] = encoder.fit_transform(df["status"])
    num_labels = df["labels"].nunique()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["labels"],
        test_size=test_size,
        stratify=df["labels"],
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    train_enc = tokenizer(
        train_texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = TextDataset(train_enc, train_labels.tolist())
    val_dataset = TextDataset(val_enc, val_labels.tolist())
    return train_dataset, val_dataset, num_labels


def create_dataloader(dataset, batch_size=16, num_workers=4, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=(sampler is not None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


def train_epoch(model, optimizer, dataloader, accelerator, epoch=None, debug=False):
    model.train()
    start_time = time.time()
    total_loss = 0.0
    desc = f"Epoch {epoch}" if epoch is not None else "Training"
    progress = tqdm(dataloader, desc=desc, unit="batch", leave=False)

    for i, batch in enumerate(progress):
        batch = {
            k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs["loss"]
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        elapsed = time.time() - start_time
        ms_per_batch = elapsed / (i + 1) * 1000
        try:
            lr = optimizer.param_groups[0]["lr"]
        except:
            lr = None

        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.6e}" if lr is not None else "n/a",
                "ms/batch": f"{ms_per_batch:6.1f}",
            }
        )

    progress.close()
    return total_loss / len(dataloader), time.time() - start_time


def evaluate_model(model, dataloader, accelerator, epoch=None):
    model.eval()
    start_time = time.time()
    all_preds, all_labels = [], []
    desc = f"Evaluating Epoch {epoch}" if epoch is not None else "Evaluating"
    progress = tqdm(dataloader, desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for i, batch in enumerate(progress):
            batch = {
                k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            elapsed = time.time() - start_time
            ms_per_batch = elapsed / (i + 1) * 1000
            batch_size = batch["labels"].size(0)
            samples_per_sec = ((i + 1) * batch_size) / elapsed

            progress.set_postfix(
                {
                    "ms/batch": f"{ms_per_batch:6.1f}",
                    "samples/s": f"{samples_per_sec:6.1f}",
                }
            )

    progress.close()
    return f1_score(all_labels, all_preds, average="weighted"), time.time() - start_time


def main():
    # DeepSpeed+MoE configuration: DP=4, MP=1, EP=1
    ds_cfg = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": True,
            "loss_scale": 2048,
            "loss_scale_window": 1000,
        },
        "zero_optimization": {
            "stage": 2,
            "offload_param": {"device": "none"},
            "offload_optimizer": {"device": "none"},
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
        },
        "tensor_parallel": {"autotp_size": 1},
        "moe": {"enabled": True, "num_experts": 8, "ep_size": 1},
    }
    with open("ds_config.json", "w") as f:
        json.dump(ds_cfg, f)

    dataloader_config = DataLoaderConfiguration(
        split_batches=False, dispatch_batches=False
    )
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision="fp16",
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config="ds_config.json"),
    )
    device = accelerator.device

    global model_path
    model_path = "/ocean/projects/cis240042p/tren1/deepseek-moe-16b-base"
    data_path = "/ocean/projects/cis240042p/tren1/MLS/Combined Data.csv"

    train_set, val_set, num_labels = prepare_data(data_path)
    accelerator.print(
        f"Train: {len(train_set)}, Val: {len(val_set)}, Labels: {num_labels}"
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True
        )
        .to(device)
        .half()
    )

    total_layers = len(base_model.model.layers)
    for idx, layer in enumerate(base_model.model.layers):
        for p in layer.parameters():
            p.requires_grad = idx >= total_layers - 2

    moe_model = MoEClassifier(base_model, num_labels).to(device).half()
    optimizer = FusedAdam(
        filter(lambda p: p.requires_grad, moe_model.parameters()),
        lr=3e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    model, optimizer = accelerator.prepare(moe_model, optimizer)

    try:
        dp_size = get_data_parallel_world_size()
    except AssertionError:
        dp_size = 1
    try:
        ep_size = _get_expert_parallel_world_size(f"ep_size_{ds_cfg['moe']['ep_size']}")
    except AssertionError:
        ep_size = 1
    try:
        mp_size = get_model_parallel_world_size()
    except AssertionError:
        mp_size = 1

    accelerator.print(
        f"DataParallel={dp_size}, ExpertParallel={ep_size}, ModelParallel={mp_size}"
    )

    epochs = 3
    for epoch in range(epochs):
        avg_loss, t_train = train_epoch(
            model, optimizer, train_loader, accelerator, epoch
        )
        f1, t_eval = evaluate_model(model, val_loader, accelerator, epoch)
        torch.cuda.empty_cache()
        accelerator.print(
            f"Epoch {epoch}  Train Loss: {avg_loss:.4f} in {t_train:.1f}s  "
            f"Eval F1: {f1:.4f} in {t_eval:.1f}s"
        )

    moe_model.base_model.save_pretrained("./trained_model")


if __name__ == "__main__":
    main()
