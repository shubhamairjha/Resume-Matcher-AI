# Current Project Analysis

**Project Title:** AI-Powered Resume & Job-Role Recommendation System  
**Audit Target:** Existing Project Architecture, Dataset, Pipeline, and Baseline Performance  
**Evaluation Methodology:** Repeated Stratified Cross-Validation (Zero Data Leakage)

---

## 1. Comprehensive System Breakdown

| Audit Dimension | Existing System Property | Verified Detail / Experimental Evidence |
|---|---|---|
| **1. Current Dataset** | Textual job postings categorized across 5 distinct software engineering / tech domains stored in `data/`. | Loaded via `dataset.py` from raw text files. |
| **2. Dataset Size** | **100 total samples** (job descriptions) | Average document length: **65.8 words** (Min: 11 words, Max: 134 words). |
| **3. Number of Classes** | **5 job role categories** | - `AIML jobs`<br>- `Full stack jobs`<br>- `Software engineer jobs`<br>- `data analyst jobs`<br>- `python jobs` |
| **4. Class Distribution** | **Uniform / Perfectly Balanced** | **20 samples per class** (20% share per class). No class imbalance skew. |
| **5. Current Preprocessing** | Text cleaning pipeline in `pdf_utils.py` and `train_model.py`: | 1. Unicode NFKC normalization.<br>2. Removal of PDF bullet glyphs and symbols (`\uf000-\uf8ff`, `\u2022`, `\u25cf`).<br>3. Technical token preservation (`C++` $\rightarrow$ `cpp_lang`, `C#` $\rightarrow$ `csharp_lang`, `.NET` $\rightarrow$ `dotnet_framework`, etc.).<br>4. Lowercasing & whitespace normalization.<br>5. Domain HR stop-word removal (24 domain words + English standard). |
| **6. Current Feature Extraction** | **Baseline Pipeline:** | 1. **TF-IDF Vectorizer**: `ngram_range=(1, 2)`, `token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b"`, `min_df=2`, `max_df=0.75`, `sublinear_tf=True`.<br>2. **TruncatedSVD (LSA)**: Dimensionality reduction to `n_components=60`.<br>3. **Normalizer**: L2 norm applied to SVD embeddings. |
| **7. Current Model** | **Logistic Regression** | `LogisticRegression(C=1.5, max_iter=3000, class_weight='balanced')`. |
| **8. Training Procedure** | Scikit-learn Pipeline architecture | Pipeline trains end-to-end within standard supervised fit procedures. |
| **9. Validation Procedure** | **Repeated Stratified K-Fold CV** | 5 folds $\times$ 5 repeats (25 evaluation splits total) + Out-Of-Fold (OOF) prediction generation via `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. |
| **10. Current Accuracy** | **66.40% (CV Mean $\pm$ 9.11%)** / **67.00% (OOF Single-Pass)** | Rigorously measured (NOT hardcoded). |
| **11. Current Precision** | **65.27% (Macro Average)** / 65.27% (Weighted Average) | Measured from OOF confusion matrix. |
| **12. Current Recall** | **67.00% (Macro Average)** / 67.00% (Weighted Average) | Measured from OOF confusion matrix. |
| **13. Current F1-Score** | **65.67% (Macro F1)** / 65.67% (Weighted F1) | Measured from OOF classification report. |
| **14. Data Leakage Assessment** | **Zero Data Leakage** | All transformations (TF-IDF fitting, SVD fitting, Normalizer, and Classifier) are contained within the `Pipeline` and fit solely on the training fold during cross-validation. |
| **15. Overfitting Assessment** | **Moderate Overfitting in SVD Space** | Reducing 100 samples to 60 SVD components leaves only ~80 training samples per fold for 60 latent dimensions, creating high variance ($\pm 9.11\%$). |
| **16. Key Implementation Weaknesses** | Identified below. | 1. Lexical sparsity with TF-IDF.<br>2. SVD discarding discriminative vocabulary.<br>3. Overlapping skills across categories causing confusion on Software Engineer role (F1: 0.36). |

---

## 2. Verified Baseline Performance Report

### Model Pipeline
$$\text{Raw Text} \longrightarrow \text{TF-IDF } (1\text{-}2\text{ ngrams}) \longrightarrow \text{TruncatedSVD } (k=60) \longrightarrow \text{L2 Normalizer} \longrightarrow \text{Logistic Regression } (C=1.5)$$

### Detailed Baseline Evaluation Metrics

| Metric | Score | Evaluation Protocol |
|---|---|---|
| **Cross-Validation Accuracy** | **66.40% $\pm$ 9.11%** | 5-Fold Repeated Stratified CV (25 iterations) |
| **Out-Of-Fold (OOF) Accuracy** | **67.00%** | Single 5-Fold Stratified CV |
| **Macro Precision** | **65.27%** | Unweighted average across 5 classes |
| **Macro Recall** | **67.00%** | Unweighted average across 5 classes |
| **Macro F1-Score** | **65.67%** | Harmonic mean of Macro Precision & Recall |
| **Weighted F1-Score** | **65.67%** | Class-weighted F1-score |

### Baseline Classification Report by Job Role

```
                        precision    recall  f1-score   support

             AIML jobs       0.71      0.75      0.73        20
        Full stack jobs       0.74      0.85      0.79        20
Software engineer jobs       0.46      0.30      0.36        20
      data analyst jobs       0.68      0.75      0.71        20
            python jobs       0.67      0.70      0.68        20

               accuracy                           0.67       100
              macro avg       0.65      0.67      0.66       100
           weighted avg       0.65      0.67      0.66       100
```

### Baseline Confusion Matrix

$$\begin{pmatrix}
\text{Role} & \text{AIML} & \text{Full Stack} & \text{Soft Eng} & \text{Data Analyst} & \text{Python} \\
\textbf{AIML jobs} & 15 & 0 & 0 & 4 & 1 \\
\textbf{Full stack jobs} & 0 & 17 & 3 & 0 & 0 \\
\textbf{Software engineer jobs} & 2 & 5 & 6 & 2 & 5 \\
\textbf{data analyst jobs} & 2 & 0 & 2 & 15 & 1 \\
\textbf{python jobs} & 2 & 1 & 2 & 1 & 14 \\
\end{pmatrix}$$

---

## 3. Analysis of Baseline Weaknesses

1. **Severe Confusion in "Software Engineer jobs":**
   - The Software Engineer category achieves only **30% recall** and an **F1-score of 0.36**.
   - Out of 20 Software Engineer postings, 5 were misclassified as Full Stack, 5 as Python jobs, 2 as AIML, and 2 as Data Analyst.
   - **Root Cause:** Software engineering postings mention overlapping foundational terms (e.g., `git`, `python`, `sql`, `testing`, `algorithms`), which TF-IDF counts identically without contextual semantic disambiguation.

2. **Dimensionality Bottleneck (TruncatedSVD):**
   - Retaining 60 latent components from a small 100-sample vocabulary introduced noise rather than generalization.
