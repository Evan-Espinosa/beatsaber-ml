"""
Training Script for Beat Saber Generator

Full training loop with:
- Logging and loss tracking
- Validation after each epoch
- Checkpoint saving (best and periodic)
- Learning rate scheduling
- Gradient clipping
- Mixed precision training (optional)

Usage:
    python -m src.training.train --data data/processed --epochs 50
    python -m src.training.train --data data/processed --model small --batch-size 4
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.data.dataset import BeatSaberDataset, collate_fn, create_data_loaders
from src.data.tokenizer import EventTokenizer
from src.models.generator import (
    BeatSaberGenerator,
    BeatSaberGeneratorSmall,
    BeatSaberGeneratorLarge
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Beat Saber Generator model.

    Handles the full training loop including:
    - Training and validation
    - Checkpointing
    - Logging
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        pad_token_id: int = 0,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        save_every: int = 5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.pad_token_id = pad_token_id
        self.grad_clip = grad_clip
        self.save_every = save_every

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            self.global_step += 1

            # Log every 50 steps
            if self.global_step % 50 == 0:
                logger.info(
                    f"Step {self.global_step} | Loss: {loss:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        return total_loss / num_batches

    def train_step(self, batch: dict) -> float:
        """Single training step."""
        # Move to device
        audio_features = batch['audio_features'].to(self.device)
        token_ids = batch['token_ids'].to(self.device)
        difficulty = batch['difficulty'].to(self.device)
        audio_mask = batch['audio_mask'].to(self.device)
        token_mask = batch['token_mask'].to(self.device)

        # Teacher forcing: input = tokens[:-1], target = tokens[1:]
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        input_mask = token_mask[:, :-1]

        # Forward pass with optional AMP
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                logits = self.model(
                    audio_features, input_ids, difficulty,
                    audio_mask=audio_mask, token_mask=input_mask
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.pad_token_id
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(
                audio_features, input_ids, difficulty,
                audio_mask=audio_mask, token_mask=input_mask
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=self.pad_token_id
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            audio_features = batch['audio_features'].to(self.device)
            token_ids = batch['token_ids'].to(self.device)
            difficulty = batch['difficulty'].to(self.device)
            audio_mask = batch['audio_mask'].to(self.device)
            token_mask = batch['token_mask'].to(self.device)

            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            input_mask = token_mask[:, :-1]

            logits = self.model(
                audio_features, input_ids, difficulty,
                audio_mask=audio_mask, token_mask=input_mask
            )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=self.pad_token_id
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        start_epoch = self.epoch
        logger.info(f"Starting training from epoch {start_epoch}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save checkpoints
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            if is_best:
                self.save_checkpoint("best_model.pt", is_best=True)

        # Save final checkpoint
        self.save_checkpoint("final_model.pt")

        # Save training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history: {history_path}")

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def create_model(
    model_size: str,
    vocab_size: int,
    d_audio: int
) -> nn.Module:
    """Create model based on size."""
    if model_size == "small":
        return BeatSaberGeneratorSmall(vocab_size=vocab_size, d_audio=d_audio)
    elif model_size == "large":
        return BeatSaberGeneratorLarge(vocab_size=vocab_size, d_audio=d_audio)
    else:
        return BeatSaberGenerator(vocab_size=vocab_size, d_audio=d_audio)


def main():
    parser = argparse.ArgumentParser(description="Train Beat Saber Generator")

    # Data
    parser.add_argument("--data", default="data/processed", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-audio-len", type=int, default=8000, help="Max audio length")
    parser.add_argument("--max-token-len", type=int, default=4000, help="Max token length")

    # Model
    parser.add_argument("--model", choices=["small", "base", "large"], default="small")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save every N epochs")
    parser.add_argument("--resume", help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")

    args = parser.parse_args()

    # Create tokenizer to get vocab size
    tokenizer = EventTokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create data loaders
    logger.info(f"Loading data from {args.data}...")
    train_loader, val_loader = create_data_loaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_audio_len=args.max_audio_len,
        max_token_len=args.max_token_len
    )

    # Get feature dimension from first sample
    sample = train_loader.dataset[0]
    if hasattr(sample, 'keys'):
        d_audio = sample['audio_features'].size(-1)
    else:
        # Handle random_split wrapper
        d_audio = sample['audio_features'].size(-1)
    logger.info(f"Audio feature dimension: {d_audio}")

    # Create model
    logger.info(f"Creating {args.model} model...")
    model = create_model(args.model, vocab_size, d_audio)
    logger.info(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler (cosine annealing with warmup)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        pad_token_id=tokenizer.pad_id,
        use_amp=not args.no_amp,
        grad_clip=args.grad_clip,
        save_every=args.save_every
    )

    # Train
    trainer.train(args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
