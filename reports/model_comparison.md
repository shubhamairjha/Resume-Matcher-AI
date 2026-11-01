# Model Comparison Report

**Dataset:** 100 job postings, 5 balanced classes, avg 65.8 words/posting
**Evaluation:** Repeated Stratified K-Fold (preprocessing inside each fold — zero leakage)

---

## Summary Table

| Model | CV Acc | Std | OOF Acc | Macro F1 | Time |
|---|---|---|---|---|---|
| TF-IDF + SVD(60) + LogReg [BASELINE] | 66.4% | ±9.1% | 67.0% | 65.7% | 1s |
| TF-IDF + LinearSVC | 61.4% | ±10.1% | 62.0% | 60.6% | 1s |
| TF-IDF (Word+Char ngrams) + LogReg | 63.0% | ±10.4% | 59.0% | 59.5% | 3s |
| TF-IDF (Word+Char ngrams) + LinearSVC | 60.2% | ±11.2% | 61.0% | 60.4% | 3s |
| TF-IDF + SkillFeatures + LogReg | 58.0% | ±8.5% | 59.0% | 59.6% | 6s |
| SBERT (MiniLM-L6) + LogReg | 71.6% | ±10.2% | 70.0% | 69.2% | 327s |
| SBERT (MiniLM-L6) + LinearSVC | 68.4% | ±9.7% | 63.0% | 60.9% | 215s |
| SBERT (MPNet-base) + LogReg | 69.4% | ±8.5% | 66.0% | 63.8% | 710s |
| SBERT (MPNet-base) + LinearSVC | 68.8% | ±8.4% | 64.0% | 62.4% | 1015s |
| Hybrid: SBERT (MiniLM) + SkillFeatures + LogReg | 61.0% | ±10.2% | 62.0% | 62.4% | 261s |
| Hybrid: SBERT (MPNet) + SkillFeatures + LogReg | nan% | ±nan% | 56.0% | 56.2% | 8135s |

---

## Research Paper Baselines (IEEE ICCCNT 2024)

| Classifier | Paper Accuracy |
|---|---|
| Naive Bayes | 53.3% |
| Random Forest | 65.2% |
| Logistic Regression | 65.9% |
| Linear SVM | 66.2% |
| BERT (bert-base-uncased) | **81.6%** |

> Note: Paper may use a different/larger dataset.

---

## Detailed Results

### TF-IDF + SVD(60) + LogReg [BASELINE]

- CV Acc: 0.6640 +/- 0.0911
- OOF Acc: 0.6700
- Macro F1: 0.6567
- Weighted F1: 0.6567
- Macro Precision: 0.6527
- Macro Recall: 0.6700
- Wall time: 1.0s

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

Confusion matrix:

```
AIML jobs             : [np.int64(15), np.int64(0), np.int64(0), np.int64(4), np.int64(1)]
Full stack jobs       : [np.int64(0), np.int64(17), np.int64(3), np.int64(0), np.int64(0)]
Software engineer jobs: [np.int64(2), np.int64(5), np.int64(6), np.int64(2), np.int64(5)]
data analyst jobs     : [np.int64(2), np.int64(0), np.int64(2), np.int64(15), np.int64(1)]
python jobs           : [np.int64(2), np.int64(1), np.int64(2), np.int64(1), np.int64(14)]
```

### TF-IDF + LinearSVC

- CV Acc: 0.6140 +/- 0.1005
- OOF Acc: 0.6200
- Macro F1: 0.6061
- Weighted F1: 0.6061
- Macro Precision: 0.6026
- Macro Recall: 0.6200
- Wall time: 1.1s

```
                        precision    recall  f1-score   support

             AIML jobs       0.65      0.65      0.65        20
       Full stack jobs       0.80      0.80      0.80        20
Software engineer jobs       0.31      0.20      0.24        20
     data analyst jobs       0.70      0.70      0.70        20
           python jobs       0.56      0.75      0.64        20

              accuracy                           0.62       100
             macro avg       0.60      0.62      0.61       100
          weighted avg       0.60      0.62      0.61       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(13), np.int64(0), np.int64(1), np.int64(3), np.int64(3)]
Full stack jobs       : [np.int64(0), np.int64(16), np.int64(4), np.int64(0), np.int64(0)]
Software engineer jobs: [np.int64(3), np.int64(4), np.int64(4), np.int64(2), np.int64(7)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(1), np.int64(14), np.int64(2)]
python jobs           : [np.int64(1), np.int64(0), np.int64(3), np.int64(1), np.int64(15)]
```

### TF-IDF (Word+Char ngrams) + LogReg

- CV Acc: 0.6300 +/- 0.1039
- OOF Acc: 0.5900
- Macro F1: 0.5952
- Weighted F1: 0.5952
- Macro Precision: 0.6017
- Macro Recall: 0.5900
- Wall time: 3.2s

```
                        precision    recall  f1-score   support

             AIML jobs       0.68      0.65      0.67        20
       Full stack jobs       0.74      0.70      0.72        20
Software engineer jobs       0.35      0.40      0.37        20
     data analyst jobs       0.79      0.75      0.77        20
           python jobs       0.45      0.45      0.45        20

              accuracy                           0.59       100
             macro avg       0.60      0.59      0.60       100
          weighted avg       0.60      0.59      0.60       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(13), np.int64(0), np.int64(2), np.int64(2), np.int64(3)]
Full stack jobs       : [np.int64(0), np.int64(14), np.int64(4), np.int64(0), np.int64(2)]
Software engineer jobs: [np.int64(2), np.int64(4), np.int64(8), np.int64(1), np.int64(5)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(1), np.int64(15), np.int64(1)]
python jobs           : [np.int64(1), np.int64(1), np.int64(8), np.int64(1), np.int64(9)]
```

### TF-IDF (Word+Char ngrams) + LinearSVC

- CV Acc: 0.6020 +/- 0.1118
- OOF Acc: 0.6100
- Macro F1: 0.6042
- Weighted F1: 0.6042
- Macro Precision: 0.6137
- Macro Recall: 0.6100
- Wall time: 3.0s

```
                        precision    recall  f1-score   support

             AIML jobs       0.69      0.55      0.61        20
       Full stack jobs       0.71      0.75      0.73        20
Software engineer jobs       0.50      0.35      0.41        20
     data analyst jobs       0.67      0.70      0.68        20
           python jobs       0.50      0.70      0.58        20

              accuracy                           0.61       100
             macro avg       0.61      0.61      0.60       100
          weighted avg       0.61      0.61      0.60       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(11), np.int64(1), np.int64(1), np.int64(4), np.int64(3)]
Full stack jobs       : [np.int64(0), np.int64(15), np.int64(2), np.int64(0), np.int64(3)]
Software engineer jobs: [np.int64(1), np.int64(4), np.int64(7), np.int64(2), np.int64(6)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(1), np.int64(14), np.int64(2)]
python jobs           : [np.int64(1), np.int64(1), np.int64(3), np.int64(1), np.int64(14)]
```

### TF-IDF + SkillFeatures + LogReg

- CV Acc: 0.5800 +/- 0.0849
- OOF Acc: 0.5900
- Macro F1: 0.5964
- Weighted F1: 0.5964
- Macro Precision: 0.6143
- Macro Recall: 0.5900
- Wall time: 6.2s

```
                        precision    recall  f1-score   support

             AIML jobs       0.73      0.55      0.63        20
       Full stack jobs       0.63      0.60      0.62        20
Software engineer jobs       0.40      0.50      0.44        20
     data analyst jobs       0.76      0.65      0.70        20
           python jobs       0.54      0.65      0.59        20

              accuracy                           0.59       100
             macro avg       0.61      0.59      0.60       100
          weighted avg       0.61      0.59      0.60       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(11), np.int64(0), np.int64(2), np.int64(1), np.int64(6)]
Full stack jobs       : [np.int64(0), np.int64(12), np.int64(8), np.int64(0), np.int64(0)]
Software engineer jobs: [np.int64(0), np.int64(6), np.int64(10), np.int64(2), np.int64(2)]
data analyst jobs     : [np.int64(2), np.int64(1), np.int64(1), np.int64(13), np.int64(3)]
python jobs           : [np.int64(2), np.int64(0), np.int64(4), np.int64(1), np.int64(13)]
```

### SBERT (MiniLM-L6) + LogReg

- CV Acc: 0.7160 +/- 0.1017
- OOF Acc: 0.7000
- Macro F1: 0.6919
- Weighted F1: 0.6919
- Macro Precision: 0.6892
- Macro Recall: 0.7000
- Wall time: 327.4s

```
                        precision    recall  f1-score   support

             AIML jobs       0.65      0.75      0.70        20
       Full stack jobs       0.73      0.80      0.76        20
Software engineer jobs       0.47      0.35      0.40        20
     data analyst jobs       0.75      0.75      0.75        20
           python jobs       0.85      0.85      0.85        20

              accuracy                           0.70       100
             macro avg       0.69      0.70      0.69       100
          weighted avg       0.69      0.70      0.69       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(15), np.int64(0), np.int64(2), np.int64(1), np.int64(2)]
Full stack jobs       : [np.int64(0), np.int64(16), np.int64(4), np.int64(0), np.int64(0)]
Software engineer jobs: [np.int64(3), np.int64(6), np.int64(7), np.int64(3), np.int64(1)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(2), np.int64(15), np.int64(0)]
python jobs           : [np.int64(2), np.int64(0), np.int64(0), np.int64(1), np.int64(17)]
```

### SBERT (MiniLM-L6) + LinearSVC

- CV Acc: 0.6840 +/- 0.0967
- OOF Acc: 0.6300
- Macro F1: 0.6089
- Weighted F1: 0.6089
- Macro Precision: 0.5995
- Macro Recall: 0.6300
- Wall time: 215.4s

```
                        precision    recall  f1-score   support

             AIML jobs       0.58      0.70      0.64        20
       Full stack jobs       0.62      0.75      0.68        20
Software engineer jobs       0.27      0.15      0.19        20
     data analyst jobs       0.67      0.70      0.68        20
           python jobs       0.85      0.85      0.85        20

              accuracy                           0.63       100
             macro avg       0.60      0.63      0.61       100
          weighted avg       0.60      0.63      0.61       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(14), np.int64(1), np.int64(1), np.int64(2), np.int64(2)]
Full stack jobs       : [np.int64(1), np.int64(15), np.int64(4), np.int64(0), np.int64(0)]
Software engineer jobs: [np.int64(4), np.int64(8), np.int64(3), np.int64(4), np.int64(1)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(3), np.int64(14), np.int64(0)]
python jobs           : [np.int64(2), np.int64(0), np.int64(0), np.int64(1), np.int64(17)]
```

### SBERT (MPNet-base) + LogReg

- CV Acc: 0.6940 +/- 0.0852
- OOF Acc: 0.6600
- Macro F1: 0.6376
- Weighted F1: 0.6376
- Macro Precision: 0.6336
- Macro Recall: 0.6600
- Wall time: 709.9s

```
                        precision    recall  f1-score   support

             AIML jobs       0.71      0.85      0.77        20
       Full stack jobs       0.58      0.70      0.64        20
Software engineer jobs       0.40      0.20      0.27        20
     data analyst jobs       0.71      0.75      0.73        20
           python jobs       0.76      0.80      0.78        20

              accuracy                           0.66       100
             macro avg       0.63      0.66      0.64       100
          weighted avg       0.63      0.66      0.64       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(17), np.int64(0), np.int64(1), np.int64(1), np.int64(1)]
Full stack jobs       : [np.int64(0), np.int64(14), np.int64(4), np.int64(1), np.int64(1)]
Software engineer jobs: [np.int64(3), np.int64(9), np.int64(4), np.int64(2), np.int64(2)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(1), np.int64(15), np.int64(1)]
python jobs           : [np.int64(1), np.int64(1), np.int64(0), np.int64(2), np.int64(16)]
```

### SBERT (MPNet-base) + LinearSVC

- CV Acc: 0.6880 +/- 0.0840
- OOF Acc: 0.6400
- Macro F1: 0.6236
- Weighted F1: 0.6236
- Macro Precision: 0.6173
- Macro Recall: 0.6400
- Wall time: 1015.0s

```
                        precision    recall  f1-score   support

             AIML jobs       0.74      0.85      0.79        20
       Full stack jobs       0.50      0.60      0.55        20
Software engineer jobs       0.33      0.20      0.25        20
     data analyst jobs       0.71      0.75      0.73        20
           python jobs       0.80      0.80      0.80        20

              accuracy                           0.64       100
             macro avg       0.62      0.64      0.62       100
          weighted avg       0.62      0.64      0.62       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(17), np.int64(0), np.int64(1), np.int64(1), np.int64(1)]
Full stack jobs       : [np.int64(0), np.int64(12), np.int64(6), np.int64(1), np.int64(1)]
Software engineer jobs: [np.int64(2), np.int64(11), np.int64(4), np.int64(2), np.int64(1)]
data analyst jobs     : [np.int64(3), np.int64(0), np.int64(1), np.int64(15), np.int64(1)]
python jobs           : [np.int64(1), np.int64(1), np.int64(0), np.int64(2), np.int64(16)]
```

### Hybrid: SBERT (MiniLM) + SkillFeatures + LogReg

- CV Acc: 0.6100 +/- 0.1020
- OOF Acc: 0.6200
- Macro F1: 0.6242
- Weighted F1: 0.6242
- Macro Precision: 0.6333
- Macro Recall: 0.6200
- Wall time: 261.2s

```
                        precision    recall  f1-score   support

             AIML jobs       0.69      0.55      0.61        20
       Full stack jobs       0.60      0.60      0.60        20
Software engineer jobs       0.38      0.45      0.41        20
     data analyst jobs       0.79      0.75      0.77        20
           python jobs       0.71      0.75      0.73        20

              accuracy                           0.62       100
             macro avg       0.63      0.62      0.62       100
          weighted avg       0.63      0.62      0.62       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(11), np.int64(0), np.int64(2), np.int64(2), np.int64(5)]
Full stack jobs       : [np.int64(0), np.int64(12), np.int64(8), np.int64(0), np.int64(0)]
Software engineer jobs: [np.int64(1), np.int64(7), np.int64(9), np.int64(2), np.int64(1)]
data analyst jobs     : [np.int64(2), np.int64(1), np.int64(2), np.int64(15), np.int64(0)]
python jobs           : [np.int64(2), np.int64(0), np.int64(3), np.int64(0), np.int64(15)]
```

### Hybrid: SBERT (MPNet) + SkillFeatures + LogReg

- CV Acc: nan +/- nan
- OOF Acc: 0.5600
- Macro F1: 0.5621
- Weighted F1: 0.5621
- Macro Precision: 0.5706
- Macro Recall: 0.5600
- Wall time: 8135.4s

```
                        precision    recall  f1-score   support

             AIML jobs       0.69      0.55      0.61        20
       Full stack jobs       0.50      0.50      0.50        20
Software engineer jobs       0.32      0.35      0.33        20
     data analyst jobs       0.72      0.65      0.68        20
           python jobs       0.62      0.75      0.68        20

              accuracy                           0.56       100
             macro avg       0.57      0.56      0.56       100
          weighted avg       0.57      0.56      0.56       100
```

Confusion matrix:

```
AIML jobs             : [np.int64(11), np.int64(0), np.int64(2), np.int64(1), np.int64(6)]
Full stack jobs       : [np.int64(0), np.int64(10), np.int64(9), np.int64(1), np.int64(0)]
Software engineer jobs: [np.int64(2), np.int64(8), np.int64(7), np.int64(2), np.int64(1)]
data analyst jobs     : [np.int64(2), np.int64(2), np.int64(1), np.int64(13), np.int64(2)]
python jobs           : [np.int64(1), np.int64(0), np.int64(3), np.int64(1), np.int64(15)]
```

---

## Final Model Selection

**Best sklearn model:** SBERT (MiniLM-L6) + LogReg

- CV Accuracy: 0.7160 +/- 0.1017
- Macro F1: 0.6919
- Improvement over TF-IDF+SVD+LR baseline (64.2%): +7.4%
