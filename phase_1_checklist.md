# Phase 1 Implementation Checklist
## MVP: Weeks 1-4 

**Goal:** Generate playable Beat Saber levels from audio

**Success Criteria:**
- Parse and preprocess 1000+ maps
- Train working generator model
- Generate at least one playable map
- No major constraint violations

---

## Week 1: Data Pipeline Foundation

### Day 1-2: Project Setup

**Tasks:**
- [ ] Create project structure
```bash
mkdir -p beatsaber-ml/{data/{raw,processed},src/{data,models,training,evaluation},checkpoints,outputs,notebooks}
cd beatsaber-ml
git init
```

- [ ] Set up environment
```bash
conda create -n beatsaber python=3.10
conda activate beatsaber
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa madmom numpy pandas tqdm jupyter requests
```

- [ ] Create initial files:
  - `src/data/beatsaver_api.py` - API client
  - `src/data/parser.py` - Map parser
  - `notebooks/01_data_exploration.ipynb` - For testing

**Deliverables:**
- Working Python environment
- Project skeleton ready

**Time estimate:** 2-4 hours

---

### Day 3-4: BeatSaver API Client

**Implementation:**

File: `src/data/beatsaver_api.py`

```python
import requests
import zipfile
import time
from pathlib import Path

class BeatSaverAPI:
    def __init__(self, output_dir='data/raw'):
        self.api_base = 'https://api.beatsaver.com'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def search_maps(self, min_votes=50, min_ratio=0.75, limit=1000):
        # TODO: Implement (see data_pipeline_spec.md)
        pass
    
    def download_map(self, map_data):
        # TODO: Implement
        pass
```

**Testing checklist:**
- [ ] Successfully connect to API
- [ ] Download 10 test maps
- [ ] Verify all files present (Info.dat, .dat files, .egg)
- [ ] Handle errors gracefully (missing files, network issues)

**Deliverables:**
- Working API client
- 10 downloaded test maps

**Time estimate:** 3-5 hours

---

### Day 5-7: Map Parser

**Implementation:**

File: `src/data/parser.py`

```python
class BeatSaberParser:
    def __init__(self, ticks_per_beat=16):
        self.ticks_per_beat = ticks_per_beat
    
    def to_canonical(self, map_folder):
        # 1. Parse Info.dat
        metadata = self.parse_info_dat(map_folder / 'Info.dat')
        
        # 2. Detect version
        version = metadata['version']
        
        # 3. Parse each difficulty
        difficulties = []
        for diff_info in metadata['difficulties']:
            diff_path = map_folder / diff_info['filename']
            
            if version == 'v2':
                events = self.parse_difficulty_v2(diff_path, metadata)
            else:
                events = self.parse_difficulty_v3(diff_path, metadata)
            
            difficulties.append({
                'metadata': {**metadata, **diff_info},
                'events': events
            })
        
        return difficulties
    
    def parse_info_dat(self, path):
        # TODO: Implement
        pass
    
    def parse_difficulty_v2(self, path, metadata):
        # TODO: Implement
        pass
```

**Testing checklist:**
- [ ] Parse all 10 test maps successfully
- [ ] Verify event counts match original .dat files
- [ ] Check beat→seconds conversion accuracy
- [ ] Verify tick quantization (events land on grid)
- [ ] Handle edge cases (missing fields, malformed JSON)

**Testing notebook:** `notebooks/01_data_exploration.ipynb`

```python
# Test parsing
parser = BeatSaberParser()
canonical = parser.to_canonical(Path('data/raw/test_map_1'))

print(f"BPM: {canonical['metadata']['bpm']}")
print(f"Events: {len(canonical['events'])}")
print(f"Duration: {canonical['events'][-1]['t_sec']:.1f}s")

# Visualize events
import matplotlib.pyplot as plt
ticks = [e['tick'] for e in canonical['events']]
plt.hist(ticks, bins=50)
plt.title('Event distribution')
plt.show()
```

**Deliverables:**
- Working parser for v2 format
- Canonical schema validated
- 10 test maps successfully parsed

**Time estimate:** 6-10 hours

---

## Week 2: Feature Extraction & Tokenization

### Day 1-3: Audio Feature Extractor

**Implementation:**

File: `src/data/features.py`

```python
import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sr=44100, n_mels=128, hop_length=512, context_frames=5):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.context_frames = context_frames
    
    def extract(self, audio_path, tick_times):
        """
        Extract features for each tick
        
        Args:
            audio_path: Path to .ogg/.egg file
            tick_times: Array of tick timestamps in seconds [T]
        
        Returns:
            features: [T, n_mels * (2*context_frames + 1)]
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Onset strength (for reference, optional)
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Beat tracking (optional, for verification)
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        
        # Frame times
        frame_times = librosa.frames_to_time(
            np.arange(mel.shape[1]), sr=sr, hop_length=self.hop_length
        )
        
        # Extract context window for each tick
        features = []
        for tick_time in tick_times:
            # Find nearest frame
            frame_idx = np.argmin(np.abs(frame_times - tick_time))
            
            # Extract context window
            start = max(0, frame_idx - self.context_frames)
            end = min(mel_db.shape[1], frame_idx + self.context_frames + 1)
            
            # Get window
            window = mel_db[:, start:end]
            
            # Pad if at edges
            if window.shape[1] < (2 * self.context_frames + 1):
                pad_width = (2 * self.context_frames + 1) - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad_width)), mode='edge')
            
            # Flatten to 1D: [n_mels * window_size]
            features.append(window.flatten())
        
        return np.array(features)
```

**Testing checklist:**
- [ ] Load .egg files correctly (they're just .ogg)
- [ ] Mel spectrogram shape correct [n_mels, n_frames]
- [ ] Context window extraction working
- [ ] Feature shape matches [n_ticks, d_audio]
- [ ] No NaN or inf values
- [ ] Visualize spectrograms (sanity check)

**Visualization:**
```python
import librosa.display

# Extract features
extractor = AudioFeatureExtractor()
tick_times = np.linspace(0, 60, 960)  # 60 seconds, 16 ticks/beat at 120 BPM
features = extractor.extract('data/raw/test_map_1/song.egg', tick_times)

# Plot
y, sr = librosa.load('data/raw/test_map_1/song.egg')
mel = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(mel), x_axis='time', y_axis='mel')
plt.title('Mel spectrogram')
plt.show()
```

**Deliverables:**
- Working audio feature extraction
- Features verified visually

**Time estimate:** 4-6 hours

---

### Day 4-5: Event Tokenizer

**Implementation:**

File: `src/data/tokenizer.py`

```python
class EventTokenizer:
    def __init__(self):
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        # TODO: Build token↔id mappings
    
    def events_to_tokens(self, events):
        # TODO: Implement
        pass
    
    def tokens_to_events(self, tokens):
        # TODO: Implement (for verification)
        pass
```

**Testing checklist:**
- [ ] Vocabulary size = 406
- [ ] All expected token types present
- [ ] Round-trip test: events → tokens → events (should match)
- [ ] Handle edge cases (empty events, simultaneous notes)
- [ ] Verify TIME_SHIFT logic works

**Round-trip test:**
```python
tokenizer = EventTokenizer()

# Original events
original_events = canonical['events']

# Convert to tokens
tokens = tokenizer.events_to_tokens(original_events)
print(f"Tokens: {len(tokens)}")

# Convert back
reconstructed_events = tokenizer.tokens_to_events(tokens)

# Compare
assert len(original_events) == len(reconstructed_events)
for orig, recon in zip(original_events, reconstructed_events):
    assert orig['tick'] == recon['tick']
    assert orig['type'] == recon['type']
```

**Deliverables:**
- Working tokenizer
- Round-trip validation passed

**Time estimate:** 3-5 hours

---

### Day 6-7: Preprocessing Pipeline

**Implementation:**

File: `src/data/preprocess.py`

```python
def preprocess_map(map_folder, output_dir, parser, extractor, tokenizer):
    """
    Process single map: parse → extract features → tokenize → save
    """
    try:
        # Parse
        canonical = parser.to_canonical(map_folder)
        
        # Filter constant BPM
        if has_bpm_changes(canonical):
            return None
        
        # Extract audio features
        tick_times = [e['t_sec'] for e in canonical['events']]
        audio_features = extractor.extract(
            map_folder / 'song.egg', tick_times
        )
        
        # Tokenize
        tokens = tokenizer.events_to_tokens(canonical['events'])
        token_ids = tokenizer.tokens_to_ids(tokens)
        
        # Save
        output_path = output_dir / f"{map_folder.name}_{canonical['metadata']['difficulty']}.pt"
        torch.save({
            'audio_features': torch.FloatTensor(audio_features),
            'tokens': torch.LongTensor(token_ids),
            'metadata': canonical['metadata']
        }, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Failed: {e}")
        return None

def preprocess_all(raw_dir='data/raw', output_dir='data/processed'):
    # TODO: Process all maps in parallel
    pass
```

**Tasks:**
- [ ] Implement full pipeline
- [ ] Add progress bar (tqdm)
- [ ] Process 10 test maps
- [ ] Verify .pt files load correctly
- [ ] Check memory usage (should be manageable)

**Deliverables:**
- 10 preprocessed .pt files
- Preprocessing script ready to scale

**Time estimate:** 3-4 hours

---

**Week 1 Checkpoint:**
- Successfully downloaded 10-100 test maps
- Parser working for v2 format
- Audio features extracted
- Tokenization verified
- Ready to scale to full dataset

---

## Week 3: Model Implementation

### Day 1-2: Dataset & DataLoader

**Implementation:**

File: `src/data/dataset.py`

```python
from torch.utils.data import Dataset

class BeatSaberDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / split
        self.samples = list(self.data_dir.glob('*.pt'))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        
        return {
            'audio_features': data['audio_features'],  # [T, D_audio]
            'tokens': data['tokens'],  # [L]
            'difficulty': self._difficulty_to_int(
                data['metadata']['difficulty']
            )
        }
    
    def _difficulty_to_int(self, difficulty_str):
        mapping = {
            'Easy': 0, 'Normal': 1, 'Hard': 2,
            'Expert': 3, 'Expert+': 4, 'ExpertPlus': 4
        }
        return mapping.get(difficulty_str, 3)  # Default to Expert
```

**Collate function (handle variable lengths):**
```python
def collate_fn(batch):
    """
    Pad sequences to same length within batch
    """
    audio_features = [item['audio_features'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    difficulties = torch.tensor([item['difficulty'] for item in batch])
    
    # Pad audio
    max_audio_len = max(f.size(0) for f in audio_features)
    audio_padded = torch.stack([
        F.pad(f, (0, 0, 0, max_audio_len - f.size(0)))
        for f in audio_features
    ])
    
    # Pad tokens
    max_token_len = max(t.size(0) for t in tokens)
    token_padded = torch.stack([
        F.pad(t, (0, max_token_len - t.size(0)), value=PAD_ID)
        for t in tokens
    ])
    
    return {
        'audio_features': audio_padded,
        'tokens': token_padded,
        'difficulty': difficulties
    }
```

**Testing:**
- [ ] Load dataset
- [ ] Create DataLoader
- [ ] Iterate through batches
- [ ] Verify shapes correct

**Deliverables:**
- Working Dataset class
- DataLoader ready

**Time estimate:** 2-3 hours

---

### Day 3-5: Model Architecture

**Implementation:**

File: `src/models/generator.py`

(See PROJECT_SPEC.ipynb for full implementation)

**Testing checklist:**
- [ ] Model instantiates without errors
- [ ] Forward pass works
- [ ] Output shape correct [B, L, vocab_size]
- [ ] Backward pass works (gradients flow)
- [ ] Sanity check: overfit on 1 sample

**Sanity test:**
```python
# Overfit on single sample
model = BeatSaberGenerator()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sample = dataset[0]
for epoch in range(100):
    logits = model(
        sample['audio_features'].unsqueeze(0),
        sample['tokens'][:-1].unsqueeze(0),
        torch.tensor([sample['difficulty']])
    )
    
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        sample['tokens'][1:].unsqueeze(0).reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Loss should go to near 0
```

**Deliverables:**
- Working BeatSaberGenerator
- Overfitting test passed

**Time estimate:** 6-8 hours

---

### Day 6-7: Training Loop

**Implementation:**

File: `src/training/train.py`

```python
def train_epoch(model, dataloader, optimizer, device, tokenizer):
    # TODO: See PROJECT_SPEC.ipynb
    pass

def validate(model, val_dataloader, device, tokenizer):
    # TODO: Compute validation loss
    pass

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_dataset = BeatSaberDataset('data/processed', split='train')
    val_dataset = BeatSaberDataset('data/processed', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=8, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, 
                           collate_fn=collate_fn)
    
    # Model
    model = BeatSaberGenerator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Train
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer)
        val_loss = validate(model, val_loader, device, tokenizer)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      f'checkpoints/model_epoch_{epoch+1}.pt')
```

**Testing:**
- [ ] Train for 1 epoch on small dataset (10 samples)
- [ ] Verify loss decreases
- [ ] Check GPU utilization
- [ ] Monitor memory usage

**Deliverables:**
- Complete training script
- Successfully trained for 1 epoch

**Time estimate:** 4-5 hours

---

**Week 3 Checkpoint:**
- Model architecture implemented
- Training loop working
- Ready to train on full dataset

---

## Week 4: Training & Generation

### Day 1-3: Full Training Run

**Tasks:**
- [ ] Download full dataset (1000+ maps if not more)
- [ ] Preprocess all maps
- [ ] Start training on Colab Pro
- [ ] Monitor training (loss curves, validation)
- [ ] Save best checkpoint

**Monitoring:**
```python
# Track metrics
metrics = {
    'train_loss': [],
    'val_loss': [],
    'learning_rate': []
}

# Plot every 5 epochs
if epoch % 5 == 0:
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Val')
    plt.legend()
    plt.show()
```

**Expected results:**
- Train loss: ~3-4 initially, ~1-2 after 50 epochs
- Val loss: Should track train loss (gap <0.5)
- If val loss >> train loss: overfitting

**Deliverables:**
- Trained model checkpoint
- Training curves

**Time estimate:** ~12 hours training time + 2-3 hours monitoring

---

### Day 4-5: Constraint Validator

**Implementation:**

File: `src/evaluation/constraints.py`

(See PROJECT_SPEC.ipynb and model_architecture_spec.md)

**Testing:**
- [ ] Test on hand-crafted valid map (should pass)
- [ ] Test on hand-crafted invalid map (should fail)
- [ ] Verify fix() function works

**Deliverables:**
- Working constraint validator

**Time estimate:** 3-4 hours

---

### Day 6-7: Generation & Testing

**Implementation:**

File: `src/generation/generate.py`

```python
def generate_map(model, audio_path, difficulty, output_dir, 
                 temperature=0.8, device='cuda'):
    """
    Generate map from audio file
    """
    model.eval()
    
    # Extract audio features
    # TODO: Need to compute tick times from audio duration + BPM estimate
    
    # Generate tokens
    tokens = generate_sequence(model, audio_features, difficulty, 
                               temperature=temperature)
    
    # Convert to events
    events = tokens_to_events(tokens, tokenizer)
    
    # Validate
    validator = ConstraintValidator()
    is_valid, violations = validator.validate(events)
    
    if not is_valid:
        print(f"Violations: {violations}")
        events = validator.fix(events)
    
    # Write .dat files
    write_output_files(events, audio_path, output_dir)
    
    return events
```

**First generation test:**
- [ ] Generate map for test song
- [ ] Check .dat files created
- [ ] Load in Beat Saber (or viewer)
- [ ] Playtest!

**Playtesting checklist:**
- [ ] Map loads without errors
- [ ] Notes appear on beats (roughly)
- [ ] No impossible patterns
- [ ] At least somewhat playable

**Expected quality (Phase 1):**
- Rough but playable
- Notes mostly aligned to beats
- Might have some awkward patterns
- Not competition-ready, but shows promise

**Deliverables:**
- Generated map files
- Playtest notes

**Time estimate:** 4-6 hours

---

## Week 4 Checkpoint

**Success criteria:**
- Model trained for 50 epochs
- Generated at least 1 playable map
- No major constraint violations
- Documented issues for Phase 2

**Deliverables:**
- Trained model checkpoint
- Generation script
- At least 1 playable map
- List of issues to address

**What's working:**
- Data pipeline
- Model architecture
- Basic generation

**What needs improvement (Phase 2):**
- Map quality (this is where scorer helps)
- Diversity (avoid repetitive patterns)
- Musicality (better beat alignment)
- Style variation

---

## Troubleshooting Guide for anyone reading this:

### Common Issues

**1. "CUDA out of memory"**
- Reduce batch_size (try 4 or even 2)
- Reduce d_model (try 256 instead of 512)
- Use gradient checkpointing
- Use mixed precision training

**2. "Loss not decreasing"**
- Check data (print samples, verify they make sense)
- Try overfitting on 10 samples first
- Reduce learning rate (try 5e-5)
- Check gradients (print grad norms)

**3. "Generated tokens are all TIME_SHIFT"**
- Model collapsed to safe output
- Increase learning rate slightly
- Check label weighting (maybe reweight non-TIME_SHIFT)
- Try different random seed

**4. "Parsing fails on many maps"**
- Add error handling (skip problematic maps)
- Log failures for debugging
- Check if specific version causing issues

**5. "Audio features all NaN"**
- Check audio loading (some .egg files might be corrupted)
- Verify librosa version
- Add clipping to prevent inf values

**Implementation begins with the Data Pipeline Foundation phase.**
