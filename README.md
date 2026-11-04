---
title: DomesticSafetyNet
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
short_description: Privacy-preserving domestic safety risk classifier
---

# 🛡️ DomesticSafetyNet

**DomesticSafetyNet** is a privacy-preserving AI crisis triage system that automatically redacts Personal Identifiable Information (PII) using a hybrid **spaCy NER + Structured Regex** pipeline before classifying domestic safety risk levels (`Immediate-Threat`, `High-Urgency`, `Medium`, `Low`) with a fine-tuned **DistilBERT** (`distilbert-base-uncased`) model.

---

## 📊 Model Evaluation & Comparison Metrics (With NLP Masking vs. Without Masking)

Evaluated on-device (NVIDIA GeForce RTX 3050 Laptop GPU) on the stratified 80/20 split of `dataset/synthetic_data_1000.json` (`799` training samples / `200` test samples, `random_state=42`, `3` epochs, `lr=5e-5`, `batch_size=16`):

### 1. Overall Performance Comparison

| Metric | 🔒 With NLP Masking (Fixed) | ⚠️ Without Masking (Raw) | 🧪 Legacy Masking (Original Colab) | Δ (Fixed Masked − Unmasked) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **77.50%** | 77.00% | 73.50% | **+0.50%** |
| **Macro Precision** | 78.84% | **83.63%** | 80.25% | -4.79% |
| **Macro Recall** | **79.55%** | 73.84% | 73.25% | **+5.71%** |
| **Macro F1-Score** | **78.60%** | 76.79% | 76.00% | **+1.81%** |
| **Weighted F1-Score** | **77.20%** | 76.89% | 74.00% | **+0.31%** |
| **Macro ROC-AUC (OVR)** | 92.99% | **93.15%** | — | -0.16% |
| **Weighted ROC-AUC (OVR)** | **91.87%** | 91.66% | — | **+0.21%** |
| **PII Leakage to Model/Logs** | **0.0% (Protected)** | 100.0% (Exposed) | 12.4% (Corrupted) | **-100.0%** |
| **Model Inference Latency** | **2.48 ms / sample** | 2.12 ms / sample | — | +0.36 ms |
| **Preprocessing Latency (NER+Regex)** | **12.79 ms / sample** | 0.01 ms / sample | — | +12.78 ms |

### 2. Per-Class Performance Breakdown (`n = 200` Test Set)

| Risk Class | Support | Precision (Masked) | Precision (Raw) | Recall (Masked) | Recall (Raw) | F1-Score (Masked) | F1-Score (Raw) | Δ F1 | ROC-AUC (Masked vs Raw) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Immediate-Threat** | 37 | 77.27% | **89.66%** | **91.89%** | 70.27% | **0.8395** | 0.7879 | **+0.0516** | 96.17% vs **97.68%** |
| **Low** | 27 | 84.00% | **100.00%** | **77.78%** | 59.26% | **0.8077** | 0.7442 | **+0.0635** | 95.83% vs **96.62%** |
| **Medium** | 63 | **71.62%** | 69.51% | 84.13% | **90.48%** | 0.7737 | **0.7862** | -0.0125 | **90.90%** vs 90.78% |
| **High-Urgency** | 73 | **82.46%** | 75.34% | 64.38% | **75.34%** | 0.7231 | **0.7534** | -0.0303 | **89.07%** vs 87.54% |

### 3. Dataset Privacy & PII Redaction Metrics (`999` Total Entries)

* **Samples Containing PII:** `618 / 999` (**61.86%** of messages contain personal or location identifiers)
* **Total PII Entities Masked:** `1,122` entities across 10 categories:
  * `[GPE]`: `405` | `[PERSON]`: `387` | `[ORG]`: `222` | `[LOC]`: `37`
  * `[ADDRESS]`: `34` (street addresses and flat/apartment/house unit numbers)
  * `[PHONE]`: `18` (`15` Indian mobile + `3` standard phone numbers)
  * `[EMAIL]`: `8` | `[LICENSE_PLATE]`: `7` | `[AADHAAR]`: `2` | `[PAN]`: `2`

### 4. Key NLP Masking Fixes Applied (`pii_masker.py` & `compare_masking.py`)

1. **Fixed Token-in-Token NER Corruption (`[[org]]`):** Previously, running `nlp(text)` *after* regex replacement caused spaCy to tag `[AADHAAR]`, `[PAN]`, and `[GPE]` as `ORG`, producing malformed `[[org]]` tokens. Fixed by substituting inert placeholders (`__PII_0__`) during regex extraction and restoring bracketed tokens after spaCy NER.
2. **Fixed Multi-Sentence Street Address Over-Matching:** Added word boundaries and token-count limits before street suffixes (`Ave`, `St`, `Ct`) so numbers followed later in a paragraph by words ending in `ave` (*"have"*), `st` (*"just"*, *"against"*), or `ct` (*"connect"*) no longer wipe out entire threat descriptions (`26` real street addresses matched vs. `3` previously, with `0` false positives).
3. **Fixed Greedy Apartment/Flat/House Clause Wiping:** Required unit numbers after `Apartment`/`Flat`/`House No.`/`C-`/`B-` so phrases like *"flat in Noida is locked from the outside"*, *"house now"*, and *"C-PTSD from domestic abuse"* are preserved intact for risk classification.

---

## 🚀 Deploying to Hugging Face Spaces (Gradio)

This repository is pre-configured for **Hugging Face Spaces** (`app.py`, `requirements.txt`, `pii_masker.py`, `metrics_comparison.json`, and YAML frontmatter in `README.md`).

### Step 1: Create a New Hugging Face Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Name your Space (e.g., `DomesticSafetyNet`), select **Gradio** as the Space SDK, and choose the **CPU basic (Free — 2 vCPU, 16 GB RAM)** hardware tier.

### Step 2: Push Repository Files to the Space
```bash
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/DomesticSafetyNet
git push space main
```

### Step 3 (Optional): Connect Your Fine-Tuned Colab Weights
1. Run `compare_masking.py` (or the final cell of `DomesticSafetyNet.ipynb`) in Google Colab.
2. Upload the saved checkpoint to Hugging Face Hub using `push_results_to_huggingface(hf_token, "your-username/DomesticSafetyNet-masked")`, **or** copy `./FineTunedBert_best_model` into the Space repository.
3. In your Hugging Face Space **Settings → Variables and secrets**, set `HF_MODEL_MASKED` to `your-username/DomesticSafetyNet-masked` (and optionally `HF_MODEL_UNMASKED` to `your-username/DomesticSafetyNet-unmasked`).

---

## 👥 Contributors
* [Anumay1231](https://github.com/Anumay1231)
* [adityarana2610](https://github.com/adityarana2610)
* [y0911ash](https://github.com/y0911ash)
