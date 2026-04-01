"""
Transformer Generator for Beat Saber ML

Encoder-Decoder Transformer that generates token sequences from audio features.

Architecture:
- Audio Encoder: Transformer encoder over tick-aligned audio features
- Token Decoder: Autoregressive Transformer decoder
- Difficulty Conditioning: Embedding added to encoder output

Input:
- audio_features: [B, T_audio, D_audio]
- token_ids: [B, T_tokens] (teacher forcing)
- difficulty: [B] (0-4)

Output:
- logits: [B, T_tokens, vocab_size]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """

    def __init__(self, d_model: int, max_len: int = 16000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BeatSaberGenerator(nn.Module):
    """
    Audio-to-Token Transformer Generator for Beat Saber maps.

    Uses an encoder-decoder architecture:
    - Encoder processes audio features
    - Decoder generates tokens autoregressively

    Example:
        model = BeatSaberGenerator(vocab_size=407, d_audio=1430)

        logits = model(
            audio_features,  # [B, T_audio, 1430]
            token_ids,       # [B, T_tokens]
            difficulty       # [B]
        )
        # logits: [B, T_tokens, 407]
    """

    def __init__(
        self,
        vocab_size: int = 407,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_feedforward: int = 2048,
        d_audio: int = 1430,
        max_seq_len: int = 16000,
        dropout: float = 0.1,
        num_difficulties: int = 5
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_feedforward: Feedforward dimension
            d_audio: Audio feature dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            num_difficulties: Number of difficulty levels (5: Easy to Expert+)
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Difficulty embedding
        self.difficulty_embed = nn.Embedding(num_difficulties, d_model)

        # Audio encoder
        self.audio_proj = nn.Linear(d_audio, d_model)
        self.audio_pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.audio_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers
        )

        # Token decoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.token_pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.token_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        audio_features: torch.Tensor,
        token_ids: torch.Tensor,
        difficulty: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            audio_features: [B, T_audio, D_audio] audio features
            token_ids: [B, T_tokens] input token sequence
            difficulty: [B] difficulty levels (0-4)
            audio_mask: [B, T_audio] True for valid positions
            token_mask: [B, T_tokens] True for valid positions

        Returns:
            logits: [B, T_tokens, vocab_size]
        """
        # Encode audio
        memory = self.encode_audio(audio_features, difficulty, audio_mask)

        # Decode tokens
        logits = self.decode_tokens(token_ids, memory, audio_mask, token_mask)

        return logits

    def encode_audio(
        self,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode audio features.

        Args:
            audio_features: [B, T_audio, D_audio]
            difficulty: [B]
            audio_mask: [B, T_audio] True for valid positions

        Returns:
            memory: [B, T_audio, d_model]
        """
        # Project audio to model dimension
        audio_embed = self.audio_proj(audio_features)  # [B, T, d_model]
        audio_embed = self.audio_pos_encoding(audio_embed)

        # Add difficulty conditioning
        diff_embed = self.difficulty_embed(difficulty)  # [B, d_model]
        audio_embed = audio_embed + diff_embed.unsqueeze(1)

        # Create attention mask (True = ignore)
        src_key_padding_mask = None
        if audio_mask is not None:
            src_key_padding_mask = ~audio_mask  # Invert: True means ignore

        # Encode
        memory = self.audio_encoder(
            audio_embed,
            src_key_padding_mask=src_key_padding_mask
        )

        return memory

    def decode_tokens(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode token sequence given encoded audio.

        Args:
            token_ids: [B, T_tokens] input tokens
            memory: [B, T_audio, d_model] encoded audio
            memory_mask: [B, T_audio] True for valid audio positions
            token_mask: [B, T_tokens] True for valid token positions

        Returns:
            logits: [B, T_tokens, vocab_size]
        """
        # Embed tokens
        token_embed = self.token_embed(token_ids)  # [B, T, d_model]
        token_embed = self.token_pos_encoding(token_embed)

        # Create causal mask
        tgt_len = token_ids.size(1)
        causal_mask = self._generate_causal_mask(tgt_len, token_ids.device)

        # Create padding masks
        tgt_key_padding_mask = None
        if token_mask is not None:
            tgt_key_padding_mask = ~token_mask

        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask

        # Decode
        decoder_out = self.token_decoder(
            tgt=token_embed,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_proj(decoder_out)

        return logits

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        difficulty: torch.Tensor,
        max_len: int = 2048,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        bos_id: int = 1,
        eos_id: int = 2
    ) -> torch.Tensor:
        """
        Generate token sequence autoregressively.

        Args:
            audio_features: [B, T_audio, D_audio] or [T_audio, D_audio]
            difficulty: [B] or scalar
            max_len: Maximum generation length
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            top_k: If set, sample from top-k tokens
            top_p: If set, use nucleus sampling
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID

        Returns:
            generated: [B, T_generated] token IDs
        """
        self.eval()

        # Handle unbatched input
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        if difficulty.dim() == 0:
            difficulty = difficulty.unsqueeze(0)

        batch_size = audio_features.size(0)
        device = audio_features.device

        # Encode audio once
        memory = self.encode_audio(audio_features, difficulty)

        # Start with BOS token
        generated = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=device
        )

        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Get logits for last position
            logits = self.decode_tokens(generated, memory)
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update done status
            done = done | (next_token.squeeze(-1) == eos_id)

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences are done
            if done.all():
                break

        return generated

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BeatSaberGeneratorSmall(BeatSaberGenerator):
    """Smaller model variant for faster experimentation."""

    def __init__(self, **kwargs):
        defaults = {
            'd_model': 256,
            'n_heads': 4,
            'n_encoder_layers': 4,
            'n_decoder_layers': 4,
            'd_feedforward': 1024,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class BeatSaberGeneratorLarge(BeatSaberGenerator):
    """Larger model variant for better quality."""

    def __init__(self, **kwargs):
        defaults = {
            'd_model': 768,
            'n_heads': 12,
            'n_encoder_layers': 8,
            'n_decoder_layers': 8,
            'd_feedforward': 3072,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Beat Saber generator')
    parser.add_argument('--size', choices=['small', 'base', 'large'], default='base')

    args = parser.parse_args()

    # Select model size
    if args.size == 'small':
        model = BeatSaberGeneratorSmall()
    elif args.size == 'large':
        model = BeatSaberGeneratorLarge()
    else:
        model = BeatSaberGenerator()

    print(f"\n=== Model ({args.size}) ===")
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    # Test forward pass
    B, T_audio, T_tokens = 2, 1000, 100
    D_audio = 1430

    audio_features = torch.randn(B, T_audio, D_audio)
    token_ids = torch.randint(0, 407, (B, T_tokens))
    difficulty = torch.tensor([3, 4])

    print(f"\nInput shapes:")
    print(f"  audio_features: {audio_features.shape}")
    print(f"  token_ids: {token_ids.shape}")
    print(f"  difficulty: {difficulty.shape}")

    # Forward pass
    logits = model(audio_features, token_ids, difficulty)
    print(f"\nOutput shape: {logits.shape}")

    # Test generation
    print(f"\nTesting generation...")
    with torch.no_grad():
        generated = model.generate(
            audio_features[:1],
            difficulty[:1],
            max_len=50,
            temperature=0.8
        )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0, :20].tolist()}")

    print("\nAll tests passed!")
