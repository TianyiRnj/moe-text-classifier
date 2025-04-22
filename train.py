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
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
from deepspeed.utils.groups import (
    get_data_parallel_group,
    get_model_parallel_group,
    _get_expert_parallel_group,
    get_data_parallel_world_size,
    get_model_parallel_world_size,
    _get_expert_parallel_world_size
)
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

# Disable tokenizer parallelism warnings for cleaner logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MoEClassifier(nn.Module):
    """
    A Mixture-of-Experts (MoE) classification wrapper around a base language model.

    - Freezes all but the last two transformer layers to reduce trainable parameters.
    - Adds a classification head projecting the final token hidden state to label logits.
    """
    def __init__(self, base_model: nn.Module, num_labels: int):
        super().__init__()
        # Underlying transformer model (e.g., LLaMA with MoE enhancements)
        self.base_model = base_model
        # Dimension of hidden representations
        hidden_size = base_model.config.hidden_size
        # Linear head mapping hidden state to class logits
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass:
        1. Compute hidden states from base model (disable cache for gradient checkpointing).
        2. Grab the last token's hidden vector.
        3. Apply classification head.
        4. Optionally compute cross-entropy loss against provided labels.

        Args:
            input_ids (torch.Tensor): Token IDs [batch, seq_len].
            attention_mask (torch.Tensor): Attention mask [batch, seq_len].
            labels (torch.Tensor, optional): Ground-truth labels [batch].

        Returns:
            dict: {"loss": loss or None, "logits": [batch, num_labels]}.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Need all layers' outputs
            use_cache=False,            # Required for activation checkpointing
            return_dict=True
        )
        # Hidden states from the last transformer block: [batch, seq_len, hidden]
        last_hidden = outputs.hidden_states[-1]
        # Use the final token's vector for classification: [batch, hidden]
        cls_token = last_hidden[:, -1, :]
        # Compute logits for each class
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            # Standard cross-entropy for multi-class classification
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class TextDataset(Dataset):
    """
    Simple PyTorch Dataset for tokenized text inputs and labels.

    Stores pre-tokenized encodings and label tensors for efficient batching.
    """
    def __init__(self, encodings, labels):
        # encodings: dict of tensors from tokenizer
        self.encodings = encodings
        # labels: list or array of class indices
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Total number of examples
        return len(self.labels)

    def __getitem__(self, idx):
        # Fetch the idx-th example
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def prepare_data(csv_path: str, test_size: float = 0.25, max_length: int = 256):
    """
    Load CSV, clean text, encode labels, split into train/val, and tokenize.

    Steps:
    1. Read CSV into DataFrame.
    2. Lowercase and strip URLs, markdown links, mentions, punctuation.
    3. Drop rows with missing values.
    4. Encode 'status' column to integer labels.
    5. Split into train/validation sets.
    6. Tokenize with padding/truncation to fixed length.

    Returns:
        train_dataset, val_dataset, num_labels
    """
    df = pd.read_csv(csv_path)
    # Clean raw text
    df["text"] = df["statement"].astype(str).str.lower()
    pattern = re.compile(r'http[s]?://\S+|\[.*?\]\(.*?\)|@\w+|[^\w\s]')
    df["text"] = df["text"].apply(lambda x: pattern.sub("", x).strip())
    df.dropna(inplace=True)

    # Encode labels to integers
    encoder = LabelEncoder()
    df["labels"] = encoder.fit_transform(df["status"])
    num_labels = df["labels"].nunique()

    # Train/validation split for stratified sampling of labels
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["labels"], test_size=test_size, stratify=df["labels"], random_state=42
    )

    # Load tokenizer from model path (global variable)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Batch-tokenize both splits
    train_enc = tokenizer(
        train_texts.tolist(), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    val_enc = tokenizer(
        val_texts.tolist(), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )

    # Wrap into PyTorch Datasets
    train_dataset = TextDataset(train_enc, train_labels.tolist())
    val_dataset = TextDataset(val_enc, val_labels.tolist())
    return train_dataset, val_dataset, num_labels


def create_dataloader(dataset, batch_size=16, num_workers=4):
    """
    Build a DataLoader with efficient options:
    - pin_memory for faster GPU transfers
    - persistent_workers to keep worker processes alive across epochs
    - prefetch_factor to fetch batches in advance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


def initialize_deepspeed(model, config_path: str, learning_rate: float = 3e-5):
    """
    Initialize DeepSpeed engine for mixed precision MoE training.

    - Uses CPU Adam offload to host memory.
    - Enables gradient checkpointing on the base model.
    """
    # Filter only trainable parameters
    cpu_optimizer = DeepSpeedCPUAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=0.01
    )
    gpu_optimizer = FusedAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    # DeepSpeed initialization
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=gpu_optimizer, config=config_path
    )
    # Enable activation checkpointing in the base transformer for memory savings
    engine.module.base_model.gradient_checkpointing_enable()
    return engine, optimizer


def train_epoch(engine, dataloader, epoch=None):
    """
    Run one epoch of training with DeepSpeed and display progress via tqdm.

    Args:
        engine: DeepSpeed engine instance.
        dataloader: PyTorch DataLoader for training data.
        epoch (int, optional): Epoch number for display purposes.

    Returns:
        avg_loss (float): Average loss over all batches.
        epoch_time (float): Total time taken by this epoch (in seconds).
    """
    engine.train()
    start_time = time.time()
    total_loss = 0.0
    # Build a description string for the progress bar
    desc = f"Epoch {epoch}" if epoch is not None else "Training"
    # Initialize tqdm progress bar
    progress = tqdm(
        dataloader,
        desc=desc,
        unit="batch",
        leave=False, 
    )

    for i, batch in enumerate(progress):
        # Move inputs to GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        # Forward pass through DeepSpeed engine
        outputs = engine(**batch)
        loss = outputs["loss"]
        # Backward pass and parameter update
        engine.backward(loss)
        engine.step()
        # Accumulate loss for averaging later
        total_loss += loss.item()
        # Attempt to read current learning rate from optimizer
        try:
            lr = engine.optimizer.param_groups[0]['lr']
        except Exception:
            lr = None

        # Compute elapsed time and average time per batch
        elapsed = time.time() - start_time
        ms_per_batch = elapsed / (i + 1) * 1000

        # Update progress bar postfix with dynamic metrics
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr":   f"{lr:.6e}" if lr is not None else "n/a",
            "ms/batch": f"{ms_per_batch:6.1f}"
        })
    # Close the progress bar to avoid clutter
    progress.close()

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), elapsed


def evaluate_model(engine, dataloader, epoch=None):
    """
    Evaluate the model on the validation set and compute weighted F1 score,
    while displaying a rich tqdm progress bar.

    Args:
        engine: DeepSpeed engine instance (or engine.module for inference).
        dataloader: PyTorch DataLoader for validation data.
        epoch (int, optional): Epoch number for display purposes.

    Returns:
        f1 (float): Weighted F1 score across all classes.
        total_time (float): Total time taken for evaluation (in seconds).
    """
    engine.eval()
    start_time = time.time()
    all_preds, all_labels = [], []
    # Build a description string for the progress bar
    description = f"Evaluating Epoch {epoch}" if epoch is not None else "Evaluating"
    progress = tqdm(
        dataloader,
        desc=description,
        unit="batch",
        leave=False  # Clear the bar when done
    )
    # Disable gradient calculations for evaluation
    with torch.no_grad():
        for i, batch in enumerate(progress):
            # Move inputs (but not labels) to GPU
            inputs = {k: v.cuda() for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].cuda()
            outputs = engine.module(**inputs)
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Compute elapsed time and throughput metrics
            elapsed = time.time() - start_time
            ms_per_batch = (elapsed / (i + 1)) * 1000
            samples_per_sec = ((i + 1) * labels.size(0)) / elapsed
            # Update the progress bar with dynamic postfix metrics
            progress.set_postfix({
                "ms/batch":   f"{ms_per_batch:6.1f}",
                "samples/s":  f"{samples_per_sec:6.1f}"
            })
    progress.close()
    elapsed = time.time() - start_time
    return f1_score(all_labels, all_preds, average="weighted"), elapsed


def main():
    """
    Main entrypoint:
    1. Initialize Accelerator for logging.
    2. Define model and data paths.
    3. Prepare data and dataloaders.
    4. Load base MoE model, freeze layers, wrap classifier.
    5. Initialize DeepSpeed engine.
    6. Train for N epochs, printing loss & F1 each epoch.
    7. Save final model weights.
    """
    accelerator = Accelerator()
    accelerator.print("CUDA available:", torch.cuda.is_available())
    accelerator.print("Devices:", torch.cuda.device_count())
    accelerator.print("Compute capability:", torch.cuda.get_device_properties(0).major,
                      torch.cuda.get_device_properties(0).minor)

    # Paths to model and dataset
    global model_path
    model_path = "/ocean/projects/cis240042p/tren1/deepseek-moe-16b-base"
    data_path = "/ocean/projects/cis240042p/tren1/MLS/Combined Data.csv"

    # DeepSpeed configuration: MoE with 2 experts per group, fp16, ZeRO-2 offload
    ds_cfg = {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 16,
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
            "overlap_comm": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True
        },
        "moe": {"ep_size": 2}
    }
    # Write config to file for DeepSpeed
    with open("ds_config.json", "w") as f:
        json.dump(ds_cfg, f)

    # Prepare datasets and loaders
    train_set, val_set, num_labels = prepare_data(data_path)
    train_loader = create_dataloader(train_set, batch_size=16, num_workers=8)
    val_loader = create_dataloader(val_set, batch_size=16, num_workers=4)

    # Load and configure base transformer model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()

    # Freeze all but the last two transformer layers
    total_layers = len(base_model.model.layers)
    for idx, layer in enumerate(base_model.model.layers):
        for param in layer.parameters():
            param.requires_grad = (idx >= total_layers - 2)

    # Wrap in our MoE classifier module
    moe_model = MoEClassifier(base_model, num_labels).cuda()

    # Initialize DeepSpeed engine with MoE support
    engine, optimizer = initialize_deepspeed(moe_model, "ds_config.json")

    # Data‑parallel group
    try:
        dp_group = get_data_parallel_group()
        dp_size  = get_data_parallel_world_size()
    except AssertionError:
        dp_group = None
        dp_size  = 1  # fallback to single‑process

    # Expert‑parallel
    ep_name = f"ep_size_{ds_cfg['moe']['ep_size']}"
    try:
        ep_group = _get_expert_parallel_group(ep_name)
        ep_size  = _get_expert_parallel_world_size(ep_name)
    except AssertionError:
        ep_group = None
        ep_size  = 1

    # Model‑parallel
    try:
        mp_group = get_model_parallel_group()
        mp_size  = get_model_parallel_world_size()
    except AssertionError:
        mp_group = None
        mp_size  = 1
    accelerator.print(f"\n\nDataParallel={dp_size}, ExpertParallel={ep_size}, ModelParallel={mp_size}\n")
    
    # Training loop: 3 epochs
    epochs = 3
    for epoch in range(1, epochs + 1):
        avg_loss, t_train = train_epoch(engine, train_loader, epoch)
        f1,  t_eval = evaluate_model(engine, val_loader, epoch)
        # Free any unused GPU memory
        torch.cuda.empty_cache()
        accelerator.print(
            f"Epoch {epoch}\n  Train Loss: {avg_loss:.4f} (in {t_train:.1f}s)\n",
            f"Eval F1: {f1:.4f} (in {t_eval:.1f}s)"
        )

    # Save the fine-tuned base model (without classifier head)
    moe_model.base_model.save_pretrained("./trained_model")


if __name__ == "__main__":
    main()
