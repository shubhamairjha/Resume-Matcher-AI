"""Audit script to inspect dataset"""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/job_postings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
labels = [d['label'] for d in data]
from collections import Counter
counts = Counter(labels)
print("Class distribution:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")

texts = [d['text'] for d in data]
word_counts = [len(t.split()) for t in texts]
print(f"\nText length stats (words):")
print(f"  min = {min(word_counts)}")
print(f"  max = {max(word_counts)}")
print(f"  mean = {sum(word_counts)/len(word_counts):.1f}")

# Near-empty texts
short = [(i, d['label'], len(d['text'].split())) for i, d in enumerate(data) if len(d['text'].split()) < 30]
print(f"\nShort texts (<30 words): {len(short)}")
for idx, lbl, wc in short:
    print(f"  idx={idx} label={lbl} wc={wc}: {data[idx]['text'][:100]}")

# Duplicate check
unique_texts = set(texts)
print(f"\nUnique texts: {len(unique_texts)} / {len(texts)}")

# Near-duplicate check: find texts that share significant overlap
print("\nSample of texts per class:")
for role in sorted(counts.keys()):
    role_texts = [d['text'] for d in data if d['label'] == role]
    print(f"\n  [{role}] ({len(role_texts)} texts)")
    for t in role_texts[:3]:
        print(f"    ({len(t.split())} words): {t[:120]}")

# Baseline CV check - run actual cross validation
print("\n\n=== Running baseline cross-validation ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import numpy as np

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.75, sublinear_tf=True)),
    ("svd", TruncatedSVD(n_components=60, random_state=42)),
    ("norm", Normalizer(copy=False)),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.5)),
])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(pipe, texts, labels, cv=cv, n_jobs=-1)
print(f"Baseline TF-IDF+SVD+LR: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"  Min fold: {scores.min():.4f}, Max fold: {scores.max():.4f}")
