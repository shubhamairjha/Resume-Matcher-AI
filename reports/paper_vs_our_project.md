# Fair Comparison: Research Paper vs. Our Project

**Reference Paper:** *AI-Powered Model for Intelligent Resume Recommendation and Feedback* (IEEE ICCCNT 2024)  
**Our Project:** *Contextual Sentence Transformer & Multi-Model Semantic Resume Recommendation System*

---

## 1. Direct Methodological and Empirical Comparison Table

| Category | Research Paper (IEEE ICCCNT 2024) | Our Project (University Research Implementation) |
|---|---|---|
| **Dataset** | Technical Resumes and Job Descriptions (collected corpus) | 100 Real-world Job Descriptions across 5 Specialized Roles (`data/`) |
| **Dataset Size** | **Not specified in the paper** (proprietary / scraped corpus) | **100 annotated postings** (20 samples $\times$ 5 balanced classes) |
| **Features** | Contextual Token Embeddings (`bert-base-uncased`) & Word Frequencies | 384-dimensional dense semantic sentence embeddings (`all-MiniLM-L6-v2`) + TF-IDF (1-2 word & char n-grams) |
| **Preprocessing** | Lowercasing, tokenization, punctuation removal, stop-word removal | Unicode NFKC normalization, glyph/bullet cleaning, tech token preservation (`C++` $\rightarrow$ `cpp_lang`, etc.), lowercasing, stop-word filtering |
| **Main Model** | **BERT (`bert-base-uncased`)** with Classification Head | **Sentence Transformer (`all-MiniLM-L6-v2`) + Logistic Regression ($C=1.0$)** |
| **Other Models Evaluated** | 1. Linear SVM (66.2%)<br>2. Logistic Regression (65.9%)<br>3. Random Forest (65.2%)<br>4. Naive Bayes (53.3%) | 1. SBERT MPNet-base + LogReg (69.4%)<br>2. SBERT MiniLM + LinearSVC (68.4%)<br>3. Baseline TF-IDF + SVD + LogReg (66.4%)<br>4. TF-IDF (Word+Char) + LogReg (63.0%)<br>5. TF-IDF + LinearSVC (61.4%)<br>6. Hybrid SBERT + Skill Features (61.0%) |
| **Accuracy** | **81.60%** (Paper Best: BERT) | **71.60%** (5-Fold CV Mean) / **70.00%** (Out-Of-Fold Single-Pass) |
| **Precision** | **Not specified in the paper** (only accuracy reported) | **68.92%** (Macro Average) / 68.92% (Weighted Average) |
| **Recall** | **Not specified in the paper** | **70.00%** (Macro Average) / 70.00% (Weighted Average) |
| **F1-Score** | **Not specified in the paper** | **69.19%** (Macro F1) / 69.19% (Weighted F1) |
| **Evaluation Method** | Train/Test Split (Exact split ratio not specified) | **5-Fold Repeated Stratified Cross-Validation (25 iterations)** + OOF Validation with Zero Leakage |
| **Explainability / Feedback** | General role recommendation & high-level feedback | **Deterministic Skill Gap Analysis**: Matched skills, missing required skills, career trajectory prediction |

---

## 2. Scientific Interpretation & Contextual Validity

> [!IMPORTANT]
> **Scientific Notice on Metric Interpretation:**  
> Accuracy values should be interpreted strictly in the context of their respective datasets and evaluation procedures.  
> 
> 1. The research paper achieved **81.60%** using full `bert-base-uncased` fine-tuning on their specific dataset corpus.  
> 2. Our project achieved **71.60%** Repeated Stratified CV accuracy (**70.00%** Out-Of-Fold) on our 100-sample domain dataset using `all-MiniLM-L6-v2` frozen contextual embeddings + regularized Logistic Regression.  
> 3. Direct percentage point comparison ($-10.00\%$ CV difference) reflects distinct corpus dimensions, vocabulary distributions, and evaluation protocols rather than purely architectural superiority.

---

## 3. Key Methodological Insights

### Why Did Sentence Transformers (SBERT) Outperform TF-IDF Baselines in Our Project?
- **Semantic Generalization:** Our baseline TF-IDF + SVD + LogReg achieved **66.40%** CV accuracy. Upgrading to SBERT (`all-MiniLM-L6-v2`) improved CV accuracy to **71.60%** ($+5.20\%$ absolute improvement).
- **Resolving Class Confusion:** Python and Data Analyst role discrimination improved significantly (Precision: 85% on Python jobs, 75% on Data Analyst jobs) because SBERT captures the contextual intent of data pipelines vs. backend web architecture.

### Why Is Frozen SBERT Better than Fine-Tuning Full BERT on Small Datasets?
- Fine-tuning a 110-million parameter BERT model from scratch on 80 training samples per fold leads to catastrophic overfitting or optimization instability.
- Frozen bi-encoders (`all-MiniLM-L6-v2`) leverage pre-trained knowledge from over 1 billion sentence pairs to project text into a robust 384-dimensional metric space without parameter inflation.
