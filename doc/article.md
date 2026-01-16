## Summary: Pre-Trained Transformer-Based Approach for Arabic Question Answering

### 1. Introduction & Problem Statement

Question Answering (QA) is a challenging Natural Language Processing (NLP) task that requires extracting precise answers from unstructured text. While English QA has advanced rapidly due to large-scale benchmarks, progress in Arabic QA has been slower because of limited resources and the scarcity of high-quality datasets. This study evaluates the performance of state-of-the-art Arabic pre-trained transformer models on four reading comprehension datasets.

---

### 2. Models Evaluated

The study compares three pre-trained transformer models:

#### AraBERTv2-base
- **Architecture:** BERT-base (12 encoder layers, 768 hidden size, 12 attention heads).
- **Parameters:** ~110 million.
- **Preprocessing:** Requires text pre-segmentation using the **Farasa segmenter** to split prefixes and suffixes.

#### AraBERTv0.2-large
- **Architecture:** BERT-large (24 transformer layers, 1024 hidden size, 16 attention heads).
- **Parameters:** ~336 million.
- **Preprocessing:** Operates on raw text; no pre-segmentation required.

#### AraELECTRA (Base Discriminator)
- **Architecture:** ELECTRA discriminator (12 encoder layers, 768 hidden size, 12 attention heads).
- **Parameters:** ~136 million.
- **Training Method:** Replaced Token Detection (RTD), where the discriminator predicts whether tokens are original or replaced by a generator.

---

### 3. Datasets Used

All datasets follow the SQuAD 1.1 format:

1. **Arabic-SQuAD**  
   48,344 questions automatically translated from the English SQuAD dataset.

2. **ARCD (Arabic Reading Comprehension Dataset)**  
   1,395 questions created by crowdworkers based on Arabic Wikipedia articles.

3. **TyDiQA-GoldP**  
   A multilingual benchmark (11 languages); the Arabic subset contains 15,645 natively written questions.

4. **AQAD (Arabic Question-Answer Dataset)**  
   17,911 questions extracted from Arabic Wikipedia articles aligned with SQuAD content.

---

### 4. Methodology

#### Text Preprocessing
- **Cleaning:** Emojis, emails, and URLs replaced with special tokens; diacritics and tatweel removed.
- **HTML Handling:** HTML tags removed for all datasets except TyDiQA-GoldP.
- **Segmentation:**  
  - AraBERTv2-base: Farasa segmentation.  
  - AraBERTv0.2-large & AraELECTRA: raw text.

#### Hyperparameters
- **Epochs:** 4 (selected after testing 2, 3, and 4).
- **Batch Size:** 4.
- **Learning Rate:** 3e-5 (selected after testing 1e-4, 2e-4, 3e-4, 5e-3).
- **Maximum Sequence Length:** 384 tokens.

---

### 5. Key Results

#### TyDiQA-GoldP
- Highest overall performance, attributed to native annotation quality.
- **AraBERTv0.2-large:** EM = 75.14, F1 = 86.49.
- **AraELECTRA:** EM = 73.07, F1 = 85.01.

#### ARCD
- **AraELECTRA** achieved the best performance with F1 = 68.15.

#### Arabic-SQuAD
- Lower performance due to translation noise.
- **AraBERTv0.2-large:** F1 = 61.21, EM = 43.26.

#### Merged Dataset Experiment
- Training on combined datasets did not significantly outperform individual datasets.
- Results suggest that a small, high-quality Arabic QA dataset is more effective than large volumes of noisy data.

---

### 6. Conclusion

Pre-trained transformer models substantially outperform earlier approaches such as mBERT and BiDAF in Arabic QA tasks. **AraBERTv0.2-large** and **AraELECTRA** consistently achieve the best results. Dataset quality is the dominant factor influencing performance, with natively annotated datasets (TyDiQA-GoldP) far exceeding translated ones (Arabic-SQuAD).
