# src/train.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)
import json
from tqdm import tqdm
import argparse
from pathlib import Path
from torch.optim import AdamW


class HeadlineDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_examples(file_path)

    def _load_examples(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        model_inputs = self.tokenizer(
            example["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                example["target_text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }


def train(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(
        device
    )

    # Prepare data
    train_dataset = HeadlineDataset(tokenizer, args.train_file)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Training setup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            # Backward pass
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

        # Save checkpoint
        output_dir = Path(args.output_dir) / f"checkpoint-{epoch+1}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BART for headline generation")
    parser.add_argument(
        "--train_file",
        default="data/processed/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--output_dir",
        default="models/bart-headline",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )

    args = parser.parse_args()
    train(args)
