# BeatSaber-ML

A machine learning system that generates Beat Saber maps from audio files using a Transformer-based architecture.

## Overview

This project uses deep learning to automatically generate playable Beat Saber levels. Given an audio file and a target difficulty, it produces note placements, obstacles, and timing that match the music.

### Features

- Automatic audio feature extraction (spectrograms, onset detection, beat tracking)
- Support for Beat Saber v2 and v3 map formats
- Multiple difficulty levels (Easy, Normal, Hard, Expert, ExpertPlus)
- Constraint validation to ensure playable maps
- BeatSaver API integration for training data collection

## Project Structure

```
beatsaber-ml/
├── src/
│   ├── data/           # Data pipeline
│   │   ├── beatsaver_api.py   # BeatSaver API client
│   │   ├── parser.py          # Map format parser
│   │   ├── features.py        # Audio feature extraction
│   │   ├── tokenizer.py       # Event tokenization
│   │   ├── preprocess.py      # Data preprocessing
│   │   └── dataset.py         # PyTorch Dataset
│   ├── models/
│   │   └── generator.py       # Transformer model
│   ├── training/
│   │   └── train.py           # Training loop
│   ├── generation/
│   │   └── generate.py        # Map generation
│   └── evaluation/
│       └── constraints.py     # Constraint validator
├── notebooks/          # Jupyter notebooks
├── scripts/            # Utility scripts
├── data/               # Training data (not included)
├── checkpoints/        # Model checkpoints (not included)
└── outputs/            # Generated maps
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/beatsaber-ml.git
cd beatsaber-ml
```

2. Create a virtual environment:
```bash
conda create -n beatsaber python=3.10
conda activate beatsaber
```

3. Install dependencies:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install -r requirements.txt
```

## Usage

### Downloading Training Data

```python
from src.data.beatsaver_api import BeatSaverAPI

api = BeatSaverAPI(output_dir='data/raw')
api.search_maps(min_votes=50, min_ratio=0.75, limit=1000)
```

### Preprocessing

```python
from src.data.preprocess import preprocess_dataset

preprocess_dataset(
    raw_dir='data/raw',
    output_dir='data/processed',
    difficulties=['Expert', 'ExpertPlus']
)
```

### Training

```python
from src.training.train import train

train(
    data_dir='data/processed',
    checkpoint_dir='checkpoints',
    epochs=100,
    batch_size=32
)
```

### Generating Maps

```python
from src.generation.generate import generate_map

generate_map(
    audio_path='path/to/song.mp3',
    difficulty='Expert',
    model_path='checkpoints/best_model.pt',
    output_dir='outputs'
)
```

## Model Architecture

The generator uses a Transformer encoder-decoder architecture:

- **Encoder**: Processes audio features (mel spectrograms, onset strength, beat frames)
- **Decoder**: Autoregressively generates tokenized map events
- **Output**: Note positions, directions, timing, and obstacles

## Data

Training data is collected from BeatSaver using their public API. Maps are filtered for quality based on:
- Minimum vote count
- Positive rating ratio
- Valid format (v2 or v3)

**Note**: Raw training data and model checkpoints are not included in this repository due to size constraints.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BeatSaver](https://beatsaver.com/) for the map database and API
- The Beat Saber modding community
