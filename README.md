# Arabic Question Answering with Pre-Trained Transformers

Implementation of **"Pre-Trained Transformer-Based Approach for Arabic Question Answering"** using AraBERT and AraELECTRA models.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Models](#models)
4. [Datasets](#datasets)
5. [Data Pipeline](#data-pipeline)
6. [Training Configuration](#training-configuration)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage](#usage)
9. [Requirements](#requirements)

---

## Project Overview

This project fine-tunes Arabic pre-trained transformer models for the **Extractive Question Answering** task. Given a question and a context passage, the model predicts the answer span within the context.

**Task Type:** Extractive QA (Span Prediction)

---

## Project Structure

```
projet3_Q&A/
├── dataset/                 # Processed dataset files
├── doc/                     # Documentation files
├── models/                  # Saved models
├── notebooks/               # Jupyter notebooks for training and analysis
│   ├── AraELECTRA.ipynb
│   ├── bert-base-arabertv2.ipynb
│   ├── cleaning.ipynb
│   ├── data_pipeline.ipynb
│   ├── exploration.ipynb
│   └── preprocessing.ipynb
├── unprocessed_data/        # Raw source data
├── utils/                   # Utility scripts
│   ├── clean_text.py
│   ├── extract_text_from_file.py
│   └── merged_data.py
├── app.py                   # Main application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Models

### 1. AraBERTv2-base
| Attribute | Value |
|-----------|-------|
| Architecture | BERT-base (Bidirectional Transformer Encoder) |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Parameters | ~110M |
| Max Sequence Length | 512 |
| Preprocessing | **Requires Farasa Segmentation** |
| Checkpoint | `aubmindlab/bert-base-arabertv2` |

### 2. AraBERTv0.2-large
| Attribute | Value |
|-----------|-------|
| Architecture | BERT-large |
| Layers | 24 |
| Hidden Size | 1024 |
| Attention Heads | 16 |
| Parameters | ~336M |
| Max Sequence Length | 512 |
| Preprocessing | None |
| Checkpoint | `aubmindlab/bert-large-arabertv02` |

### 3. AraELECTRA-base-discriminator
| Attribute | Value |
|-----------|-------|
| Architecture | ELECTRA (Discriminator) |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Parameters | ~136M |
| Max Sequence Length | 512 |
| Training Method | Replaced Token Detection (RTD) |
| Preprocessing | None |
| Checkpoint | `aubmindlab/araelectra-base-discriminator` |

---

## Datasets

The project merges multiple Arabic QA datasets:
[link_dataset_cleaned](https://www.kaggle.com/datasets/achraftic/arabic-qa-cleaned)

| Dataset | Format | Description |
|---------|--------|-------------|
| Arabic-SQuAD | JSON | Arabic translation of SQuAD |
| ARCD | JSON | Arabic Reading Comprehension Dataset |
| AQAD | JSON | Arabic Question Answering Dataset |
| TyDiQA-GoldP | JSONL | Typologically Diverse QA (Arabic subset) |

### Unified Schema

All datasets are normalized to:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier |
| `question` | Question text |
| `answer` | Answer text |
| `context` | Passage/context containing the answer |
| `title` | Document title (optional) |
| `category` | Category/label (optional) |
| `source_file` | Original source file |

---

## Data Pipeline

### Step 1: Merge Data (`utils/merge_data.py`)

Combines all source files into `master_data.csv`:
- Loads JSON, JSONL, CSV formats
- Normalizes column names
- Extracts answers from nested structures
- Removes duplicates

### Step 2: Preprocessing (`preprocessing.ipynb`)

- Removes null values
- Drops metadata columns (keeping question, answer)
- Outputs `data_cleaned.csv`

### Step 3: Cleaning (`cleaning.ipynb`)

Text preprocessing function:

```python
def preprocess_arabic_text(text):
    # 1. Remove emojis
    text = emoji.replace_emoji(text, replace="")
    
    # 2. Remove HTML (Exception: TyDiQA preserves HTML)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Replace URLs, emails, mentions
    text = re.sub(r'http[s]?://...', '[URL]', text)
    text = re.sub(r'email_pattern', '[EMAIL]', text)
    text = re.sub(r'@username', '[MENTION]', text)
    
    # 4. Remove Arabic diacritics & tatweel
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)
    
    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

Filters garbage patterns:
```python
garbage_pattern = r"array\(|dtype=|\{'text':|\\[|\\]"
df = df.filter(~pl.col("answer").str.contains(garbage_pattern))
```

Outputs: `data_preprocessed.csv`

---

## Training Configuration

### Hyperparameters (From Paper)

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-5 |
| Batch Size | 4 |
| Epochs | 3 (merged dataset) / 4 (single dataset) |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Max Sequence Length | 384 |
| Max Question Length | 64 |
| Doc Stride | 128 |
| Max Answer Length | 30 |

### Input Format

```
[CLS] Question [SEP] Context [SEP]
```

### Model-Specific Preprocessing

**AraBERTv2-base Only:**
```python
from arabert.preprocess import ArabertPreprocessor
prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv2")
text = prep.preprocess(text)  # Applies Farasa segmentation
```

---

## Evaluation Metrics

### Exact Match (EM)

Binary metric: 1 if prediction exactly matches ground truth, else 0.

```python
def compute_exact(prediction, ground_truth):
    return int(normalize(prediction) == normalize(ground_truth))
```

### F1 Score

Token-level harmonic mean of precision and recall:

```python
def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
```

**Note:** For multiple ground truth answers, take the maximum score.

---

## Usage

### 1. Install Dependencies

```bash
pip install transformers datasets torch polars arabert emoji farasapy accelerate
```

### 2. Prepare Data

Run notebooks in order:
1. `preprocessing.ipynb` → Creates `data_cleaned.csv`
2. `cleaning.ipynb` → Creates `data_preprocessed.csv`

### 3. Train Models

Open `main.ipynb` and run all cells. To switch models, modify:

```python
# Choose one:
train_pipeline("AraBERTv2-base")      # Requires Farasa
train_pipeline("AraBERTv0.2-large")   # Large model, needs GPU memory
train_pipeline("AraELECTRA-base")     # Fast training
```

### 4. Output

Trained models saved to:
```
final_models/
├── AraBERTv2-base/
├── AraBERTv0.2-large/
└── AraELECTRA-base/
```

---

## Requirements

```
torch>=1.10
transformers>=4.20
datasets>=2.0
polars>=0.15
arabert
emoji
farasapy
accelerate
scikit-learn
```

**Hardware:**
- GPU recommended (CUDA)
- AraBERT-large requires ~16GB VRAM

---

## References

- [AraBERT Paper](https://arxiv.org/abs/2003.00104)
- [AraELECTRA](https://github.com/aub-mind/arabert)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## License

This project is for educational purposes.
