import os
import json
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from tiktoken import get_encoding

from model import GPT, GPTConfig


# ----------------------------
# Dataset and Collate Function
# ----------------------------

class DPODataset(Dataset):
    def __init__(self, dataset, enc):
        self.pairs_list = dataset
        self.enc = enc

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        pair = self.pairs_list[idx]
        prompt = self.enc.encode(pair["prompt"])
        chosen = self.enc.encode(pair["chosen"])
        joined_chosen = prompt + self.enc.encode(" ") + chosen

        return {
            "chosen": joined_chosen,
        }

def custom_collate_fn(batch, pad_token_id=50256, allowed_max_length=1024):
    batch_data = {"chosen": [], "chosen_mask": []}

    max_length = max(len(item["chosen"]) + 1 for item in batch)

    for item in batch:
        sequence = item["chosen"]
        padded = sequence + [pad_token_id] * (max_length - len(sequence))
        mask = torch.ones(len(padded))
        mask[len(sequence):] = 0

        batch_data["chosen"].append(torch.tensor(padded))
        batch_data["chosen_mask"].append(mask)

    for key in ["chosen", "chosen_mask"]:
        tensor_stack = torch.stack(batch_data[key])
        batch_data[key] = tensor_stack[:, :allowed_max_length]

    return batch_data


# ----------------------------
# Data Preparation
# ----------------------------

def get_dataset(dataset, enc):
    return DPODataset(dataset, enc)

def get_val_split(dataset: DPODataset, val_size: float):
    return random_split(dataset, [1 - val_size, val_size])

def get_dataloaders(dataset: DPODataset, batch_size: int, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=shuffle)


# ----------------------------
# Loss & Forward Pass
# ----------------------------

def logprobs(logits, labels, mask):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    mask = mask[:, 1:].clone()
    selected_log_probs *= mask

    num_nonpad_tokens = mask.sum(dim=-1)
    return selected_log_probs.sum(dim=-1) / num_nonpad_tokens

def sft(logits, labels, mask):
    labels = labels[:, 1:].clone()
    mask = mask[:, 1:].clone()
    labels[mask == 0] = -1
    logits = logits[:, :-1, :]

    return F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

def forward_pass_batch(batch, model, device, beta, loss_fn):
    chosen = batch["chosen"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)

    chosen_logits = model(chosen)[0]

    chosen_logprobs = logprobs(chosen_logits, chosen, chosen_mask)

    loss = sft(chosen_logits, chosen, chosen_mask)

    return loss


# ----------------------------
# Evaluation
# ----------------------------

def eval_loss(val_loader, model, device, beta, loss_fn):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            loss = forward_pass_batch(batch, model, device, beta, loss_fn)
            losses.append(loss.item())

    return sum(losses) / len(losses)


# ----------------------------
# Sampling Utility
# ----------------------------

def test_samples(prompts, model, enc, device):
    out_completions = []
    for text in prompts:
        context = torch.tensor(enc.encode(text), device=device).unsqueeze(0)
        completion = model.generate(context)[0].tolist()
        out_completions.append(enc.decode(completion))
    return out_completions


# ----------------------------
# Training Loop
# ----------------------------

def train(train_loader, val_loader, model, optimizer, enc, device, epochs, beta, loss_fn):
    train_losses, train_steps = [], []
    val_steps, val_losses, val_chosen_rewards, val_margins = [], [], [], []

    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs, eta_min=1e-7)

    for epoch in range(1, epochs + 1):
        print(f"-------- EPOCH {epoch}/{epochs} --------")

        # Check sample generations
        model.eval()
        for completion in test_samples(["The morning started with a surprise as", "The calm before the storm"], model, enc, device):
            print(f"Sample generation: {completion}")

        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = forward_pass_batch(batch, model, device, beta, loss_fn)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step = i + (epoch - 1) * len(train_loader)
            train_steps.append(step)
            train_losses.append(loss.item())

            print(f"Step {i+1}/{len(train_loader)} | Loss: {round(loss.item(), 5)}")

            if i % (len(train_loader) // 8) == 0:
                val_loss = eval_loss(val_loader, model, device, beta, loss_fn)
                val_losses.append(val_loss)
                val_steps.append(step)
                print(f"Val loss: {val_loss}")

    return train_steps, train_losses, val_steps, val_losses


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    file_path = "model_19072.pt"  # <-- Set your path

    dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")["train"]
    bs = 4
    epochs = 1
    lr = 1e-4
    beta = 0.5
    loss_fn = "sft"

    seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    config = checkpoint["config"]

    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)

    enc = get_encoding("gpt2")
    upenn_dataset = get_dataset(dataset, enc)
    train_set, val_set = get_val_split(upenn_dataset, 0.1)
    train_loader = get_dataloaders(train_set, bs)
    val_loader = get_dataloaders(val_set, bs, shuffle=False)

    print(f"Starting training with {len(train_set)} training samples using {loss_fn} loss.")

    train_steps, train_losses, val_steps, val_losses = train(
        train_loader, val_loader, model, optimizer=Adam(model.parameters(), lr=lr),
        enc=enc, device=device, epochs=epochs, beta=beta, loss_fn=loss_fn
    )

    checkpoint_path = "sft_model.pt"

    checkpoint = {
                        'model': model.state_dict(),
                        'config': model.config,
                }

    torch.save(checkpoint, checkpoint_path)