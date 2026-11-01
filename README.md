# AI Resume / Job-Role Matcher

An ML-powered tool that matches resumes against 5 technical job roles using
semantic NLP models. Based on the IEEE ICCCNT 2024 research paper.

## Supported Job Roles
- AIML jobs (AI/ML Engineer)
- Full stack jobs (Full Stack Developer)
- Software engineer jobs (Software Engineer)
- data analyst jobs (Data Analyst / BI)
- python jobs (Python Backend Developer)

## Model Architecture (Upgraded)

### Primary Model: SBERT Hybrid
```
Resume Text (PDF extracted)
    |
Light cleaning (preserve semantics)
    |
SBERT Encoder (all-MiniLM-L6-v2, frozen weights)
    +
Domain Skill Feature Extractor
    |
Hybrid Feature Vector (384 + ~100 dims)
    |
Logistic Regression Classifier
    |
5-class probability distribution
    |
Ranked job roles with ML Match %, Similarity %, Final Match %
```

### Fallback: TF-IDF + SVD + LogReg (original)
Available when sentence-transformers is not installed.

## Measured Results (Repeated Stratified K-Fold CV)

| Model | CV Accuracy | Macro F1 |
|---|---|---|
| Old Baseline (TF-IDF+SVD+LR) | 64.2% | - |
| TF-IDF + SVD + LogReg | 66.4% | 65.7% |
| SBERT (MiniLM) + LogReg | See reports/model_comparison.md |
| SBERT Hybrid + LogReg | See reports/model_comparison.md |
| BERT fine-tuned | See reports/model_comparison.md |

Paper baseline: BERT = 81.6%, Linear SVM = 66.2%

## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies separately:
```bash
pip install scikit-learn pypdf joblib numpy scipy
pip install sentence-transformers transformers torch  # for SBERT/BERT
```

## Usage

### Train the model
```bash
# Auto-select best available model (SBERT if installed, else TF-IDF)
python train_model.py --data-dir "." --model-out model.joblib

# Force specific model type
python train_model.py --model sbert   # SBERT Hybrid (recommended)
python train_model.py --model tfidf   # TF-IDF + LinearSVC (no GPU needed)
python train_model.py --model original  # Original TF-IDF+SVD+LR baseline
```

### Match a resume
```bash
python match.py --model model.joblib --resume "your_resume.pdf"
```

### Run all ML experiments (full comparison)
```bash
# Fast: TF-IDF only
python run_experiments.py --no-bert --no-sbert

# Full: TF-IDF + SBERT (no BERT fine-tuning)
python run_experiments.py --no-bert --folds 5 --repeats 5

# Complete: includes BERT fine-tuning (slow, CPU only)
python run_experiments.py --bert-epochs 3 --bert-lr 2e-5 --bert-folds 5
```

## Output Example

```
RESUME MATCHING & RECOMMENDATION SYSTEM
Resume: resume.pdf
Model:  sbert_hybrid

Job Role                   ML Match%  Similarity%  Final Match%
---------------------------------------------------------------
AIML jobs                     87.3%        72.1%         82.2% [TOP MATCH]
python jobs                   6.2%         48.3%         18.8%
...

MATCH EXPLANATION
[+] Strong Semantic Matches Found:
    - machine learning
    - deep learning
    - pytorch
    ...

CONSTRUCTIVE CANDIDATE FEEDBACK & GAP ANALYSIS
Target Role: AI / Machine Learning Engineer
...

PREDICTIVE FUTURE JOB OPPORTUNITIES
...
```

## Project Structure

```
resume_matcher/
    train_model.py          Main training script (SBERT hybrid + TF-IDF)
    run_experiments.py      Full multi-model experiment runner
    match.py                Resume scoring and recommendation
    dataset.py              Dataset loading (PDF folders + JSON)
    pdf_utils.py            PDF text extraction
    feedback_engine.py      Candidate feedback & gap analysis
    future_prediction.py    Career path predictions
    data/
        job_postings.json   100 training samples (5 classes, 20 each)
    reports/
        model_audit.md      Phase 1 audit: baseline analysis
        model_comparison.md Full experiment comparison (auto-generated)
```

## Evaluation Methodology

All reported results use **Repeated Stratified K-Fold CV** (5 splits x 5 repeats = 25 folds).
- No test-set leakage: all preprocessing inside each fold
- Metrics: accuracy, macro F1, weighted F1, per-class precision/recall
- OOF (out-of-fold) predictions for confusion matrix

## Dataset

- 100 real job postings from 5 roles (20 per class)
- Source: Various companies (Amazon, Accenture, BNP Paribas, etc.)
- Average text: 65.8 words/posting
- Format: JSON (data/job_postings.json) or PDF folder structure
