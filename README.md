# VQA-RAD Medical Visual Question Answering

This project implements medical Visual Question Answering (VQA) models on the VQA-RAD dataset, comparing CNN+Transformer and BLIP architectures.

## 📋 Overview

This repository contains implementations of two different VQA models:
- **CNN+Transformer**: A custom architecture combining ResNet18 for image encoding and Transformer Encoder for text processing
- **BLIP**: A pre-trained Vision-Language Model fine-tuned on VQA-RAD dataset

## 🏗️ Project Structure

```
.
├── CNN+TR.ipynb              # CNN+Transformer model implementation and training
├── VLM-based.ipynb           # BLIP model implementation and training
├── baseline.ipynb            # Baseline experiments
├── data/
│   └── archive (1)/
│       ├── VQA_RAD Dataset Public.json
│       └── VQA_RAD Image Folder/    # Medical images (not included in repo)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for BLIP model

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd JupyterProject3

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the **VQA-RAD** dataset:
- **Total samples**: 2,248
- **Close (Yes/No) questions**: 1,193
- **Open-ended questions**: 1,055
- **Train/Test split**: 80/20

The dataset is split into:
- **Close tasks**: Binary classification (yes/no)
- **Open tasks**: Open-ended text generation

## 🧪 Model Architectures

### 1. CNN+Transformer Model

- **Image Encoder**: ResNet18 (pre-trained on ImageNet)
- **Text Encoder**: Transformer Encoder (2 layers, 8 attention heads)
- **Fusion**: Concatenation + Fully Connected layers
- **Output**: Classification (2 classes for close, N classes for open)

**Key Features**:
- Data augmentation (RandomResizedCrop, RandomHorizontalFlip)
- Custom vocabulary built from questions
- Retrieval-based approach for open tasks

### 2. BLIP Model

- **Image Encoder**: Vision Transformer (ViT)
- **Text Encoder**: BERT-based encoder
- **Fusion**: Cross-attention mechanism
- **Output**: Text generation (autoregressive)

**Key Features**:
- Pre-trained on large-scale image-text pairs
- Generative approach for answers
- Fine-tuning with small learning rate

## 📈 Results

### Close (Yes/No) Task

| Model | Accuracy | Notes |
|-------|----------|-------|
| CNN+Transformer | 69.87% | Fast training, good for binary classification |
| BLIP | 70.29% | Slightly better, more stable training |

### Open Task

| Model | Exact Match | Semantic Match | Notes |
|-------|-------------|----------------|-------|
| CNN+Transformer | ~11% | - | Retrieval-based, limited by answer vocabulary |
| BLIP | 13.27% | 28.91% | Generative approach, more flexible |

## 💻 Usage

### Training CNN+Transformer Model

Open `CNN+TR.ipynb` and run cells sequentially:
1. Cell 0: Import libraries and define dataset
2. Cell 1: Data splitting and vocabulary building
3. Cell 2: Model definition
4. Cell 3: Train Close task
5. Cell 4: Evaluate Close task
6. Cell 5: Train Open task
7. Cell 6: Evaluate Open task

### Training BLIP Model

Open `VLM-based.ipynb` and run cells sequentially:
1. Cell 0: Dataset definition
2. Cell 1: Data splitting
3. Cell 2: Load BLIP model and processor
4. Cell 3: Train Close task
5. Cell 4: Evaluate Close task
6. Cell 5: Train Open task
7. Cell 6: Evaluate Open task

## 🔧 Configuration

### CNN+Transformer
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Epochs**: 10
- **Optimizer**: AdamW (weight_decay=1e-5)
- **Image size**: 224×224
- **Max question length**: 20 tokens

### BLIP
- **Batch size**: 8 (close) / 4 (open)
- **Learning rate**: 5e-5
- **Epochs**: 4 (close) / 5 (open)
- **Optimizer**: AdamW
- **Image size**: 384×384
- **Max generation length**: 10-20 tokens

## 📝 Data Processing

### CNN+Transformer
- **Image augmentation**: RandomResizedCrop, RandomHorizontalFlip
- **Text processing**: Custom tokenization and vocabulary
- **Answer format**: Class labels (0/1 for close, indices for open)

### BLIP
- **Image processing**: BLIP Processor (no augmentation)
- **Text processing**: BERT tokenizer
- **Answer format**: Text sequences (token IDs)

## 🗂️ Files

- `CNN+TR.ipynb`: Complete CNN+Transformer implementation
- `VLM-based.ipynb`: Complete BLIP implementation
- `baseline.ipynb`: Baseline experiments
- `requirements.txt`: Python package dependencies
- `data/`: Dataset directory (images not included in repo)

## ⚠️ Important Notes

1. **Model weights** (`.pth` files) are not included in the repository due to size
2. **Dataset images** are not included - download from VQA-RAD official source
3. **GPU required** for training, especially for BLIP model
4. **Data augmentation** is used only for CNN+Transformer, not for BLIP

## 📚 References

- VQA-RAD Dataset: [Paper](https://www.nature.com/articles/sdata2018251)
- BLIP Model: [Paper](https://arxiv.org/abs/2201.12086) | [Hugging Face](https://huggingface.co/Salesforce/blip-vqa-base)

## 📄 License

[Specify your license here]

## 👤 Author

[Your name/contact information]
