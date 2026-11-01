# Model Audit Report

**Project:** AI Resume/Job-Role Matcher
**Paper:** AI-Powered Model for Intelligent Resume Recommendation and Feedback (IEEE ICCCNT 2024)
**Audit Date:** 2026-09-29

---

## 1. Current Architecture

```
PDF Text Extraction (pypdf)
    |
Unicode normalization + bullet/artifact removal (pdf_utils.py)
    |
preprocess_document(): lowercase, regex substitution of tech tokens
    (C++ -> cpp_lang, C# -> csharp_lang, .NET -> dotnet_framework, etc.)
    |
TF-IDF Vectorizer
  - token_pattern: [a-zA-Z0-9_+#.-]{2,}
  - ngram_range: (1, 2)
  - min_df: 2 / max_df: 0.75
  - sublinear_tf: True
  - stop_words: sklearn english + 24 domain HR noise words
    |
TruncatedSVD (LSA) -- n_components=60
    |
L2 Normalizer
    |
Logistic Regression (C=1.5, max_iter=3000, class_weight=balanced)
    |
5-class softmax probabilities -- ranked job roles
```

**Evaluation:** RepeatedStratifiedKFold(n_splits=5, n_repeats=10) = 50 folds total

---

## 2. Dataset Statistics

| Attribute | Value |
|---|---|
| Total samples | 100 |
| Number of classes | 5 |
| Samples per class | 20 (perfectly balanced) |
| Class imbalance | None |
| Unique texts | 100 / 100 (no exact duplicates) |
| Min text length | 11 words |
| Max text length | 134 words |
| Mean text length | 65.8 words |
| Short texts under 30 words | 2 (idx 33: 11w, idx 34: 15w) |
| Data leakage detected | None |
| Source diversity | Multiple real companies per class |

### Classes
| Class | Count |
|---|---|
| AIML jobs | 20 |
| Full stack jobs | 20 |
| Software engineer jobs | 20 |
| data analyst jobs | 20 |
| python jobs | 20 |

---

## 3. Current Preprocessing

- Unicode NFKC normalization (pdf_utils.py)
- Bullet/glyph artifact removal via regex
- Tech token preservation: C++ -> cpp_lang, etc.
- Domain stop-word removal (24 extra words on top of sklearn English)
- Sublinear TF weighting

**Critical problem:** Aggressive stop-word removal destroys contextual cues.
The pipeline is designed only for TF-IDF, not for transformer models.

---

## 4. Current Evaluation Method

- **CV method:** RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
- **Metric reported:** Accuracy only (no F1, precision, recall, confusion matrix)
- **Data leakage:** None -- preprocessing happens inside sklearn pipeline within CV

**Key finding:** The train_model.py code has this line hardcoded:
  cv_mean = 0.67  (fallback default if CV is not run)
This means if CV is skipped, 67% is printed regardless of actual performance.

---

## 5. Baseline Accuracy (Measured)

| Model | CV Accuracy | Std Dev | Min Fold | Max Fold |
|---|---|---|---|---|
| TF-IDF + SVD(60) + LogReg | **64.2%** | +/-10.8% | 40.0% | 85.0% |

NOTE: The hardcoded "67%" fallback in the source was NOT measured from repeated CV.
The actual measured repeated-stratified-CV baseline is 64.2% (50 folds).

Paper comparison baselines (different dataset):
- Naive Bayes: 53.3%
- Random Forest: 65.2%
- Logistic Regression: 65.9%
- Linear SVM: 66.2%
- BERT (bert-base-uncased): 81.6%

---

## 6. Problems and Limitations

### Dataset Problems
1. **Tiny dataset (100 samples, 65 words avg):** BERT fine-tuning on 80 train samples
   is extremely high-risk. Expected to overfit without careful regularisation.
2. **Very short texts:** Job postings are bullet-list snippets, not full descriptions.
3. **High class overlap:** AIML, python jobs, software engineer all share Python, SQL,
   algorithms, git. Class boundaries are semantically fuzzy.
4. **2 near-empty texts (11-15 words):** Degrade training quality.

### Architecture Problems
1. **SVD compression to 60 dims is counterproductive** with only 100 docs.
   Full sparse TF-IDF + LinearSVC performs better on small corpora.
2. **No semantic understanding:** TF-IDF cannot distinguish "Python APIs"
   (python jobs) from "Python for ML" (AIML jobs) from context.
3. **Domain skill taxonomy in feedback_engine.py is not used as ML features.**
4. **Hardcoded fabricated accuracy "88.4% - 94%"** in train_model.py
   print_paper_comparison(). This was never experimentally measured.

---

## 7. Proposed Improved Architecture

### Tier 1 -- TF-IDF Improvements (no extra deps)
- Remove SVD; use full sparse TF-IDF + LinearSVC
- Add character n-gram FeatureUnion for robustness
- Expected improvement: +2-4% over baseline

### Tier 2 -- Sentence Transformer (primary improvement)
- Use all-MiniLM-L6-v2 or all-mpnet-base-v2 as frozen encoder
- Extract contextual embeddings; train LogReg/SVC on top
- No overfitting risk (transformer weights frozen)
- Expected: 70-85%

### Tier 3 -- BERT Fine-Tuning
- bert-base-uncased with classification head
- AdamW + linear warmup + dropout + weight decay
- 2-5 epochs, lr in {1e-5, 2e-5, 3e-5}
- High variance expected due to tiny train set

### Tier 4 -- Hybrid
- SBERT embeddings + domain skill match features
- Soft-voting ensemble of best TF-IDF + SBERT model
- Expected: Marginal improvement over best individual model

### Honest Assessment
With 100 samples and 65-word avg texts, matching the paper's 81.6% BERT result
is possible with SBERT (pretrained embeddings) but not guaranteed.
The high CV std dev (+/-10%) means results vary significantly across folds.
Paper likely used a larger dataset or different evaluation protocol.

