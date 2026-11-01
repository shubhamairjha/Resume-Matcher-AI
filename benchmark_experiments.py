import json
import sys
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

def main():
    # Load dataset
    with open('data/job_postings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [d['text'] for d in data]
    labels = [d['label'] for d in data]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    DOMAIN_STOPWORDS = [
        "experience", "years", "working", "work", "role", "candidate", "candidates",
        "required", "requirements", "skills", "ability", "strong", "team", "teams",
        "knowledge", "degree", "bachelor", "master", "qualifications", "responsibilities",
        "duties", "job", "position", "company", "opportunity", "environment", "business",
        "understanding", "proficient", "proficiency", "preferred", "plus", "must", "have"
    ]

    all_stops = list(TfidfVectorizer(stop_words="english").get_stop_words()) + DOMAIN_STOPWORDS

    print("=== BENCHMARKING MODELS ON RESUME/JOB DATASET ===", flush=True)

    print("\n--- BASELINE (Original Project: TF-IDF + SVD(60) + LogReg) ---", flush=True)
    base_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.75, sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=60, random_state=42)),
        ("norm", Normalizer(copy=False)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)),
    ])
    base_scores = cross_val_score(base_pipe, texts, labels, cv=cv, n_jobs=1)
    print(f"Original Baseline Accuracy: {base_scores.mean():.1%} (+/- {base_scores.std():.1%})\n", flush=True)

    experiments = [
        (
            "TF-IDF (Word ngrams 1-2 + tech token pattern + custom stops) + LogReg",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                    stop_words=all_stops,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True
                )),
                ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))
            ])
        ),
        (
            "TF-IDF + LinearSVC",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                    stop_words=all_stops,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True
                )),
                ("clf", LinearSVC(C=1.0, max_iter=3000, random_state=42))
            ])
        ),
        (
            "TF-IDF + Ridge Classifier",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                    stop_words=all_stops,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True
                )),
                ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced"))
            ])
        ),
        (
            "TF-IDF + Complement Naive Bayes",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                    stop_words=all_stops,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True
                )),
                ("clf", ComplementNB(alpha=0.5))
            ])
        ),
        (
            "FeatureUnion (Word n-grams + Char-wb n-grams) + LogisticRegression",
            Pipeline([
                ("features", FeatureUnion([
                    ("word", TfidfVectorizer(
                        lowercase=True,
                        token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                        stop_words=all_stops,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.8,
                        sublinear_tf=True
                    )),
                    ("char", TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_df=0.9,
                        sublinear_tf=True
                    ))
                ])),
                ("clf", LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced"))
            ])
        ),
        (
            "FeatureUnion (Word + Char-wb) + LinearSVC",
            Pipeline([
                ("features", FeatureUnion([
                    ("word", TfidfVectorizer(
                        lowercase=True,
                        token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                        stop_words=all_stops,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.8,
                        sublinear_tf=True
                    )),
                    ("char", TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_df=0.9,
                        sublinear_tf=True
                    ))
                ])),
                ("clf", LinearSVC(C=1.0, max_iter=3000, random_state=42))
            ])
        ),
        (
            "FeatureUnion + Neural Network (MLPClassifier)",
            Pipeline([
                ("features", FeatureUnion([
                    ("word", TfidfVectorizer(
                        lowercase=True,
                        token_pattern=r"(?u)\b[a-zA-Z0-9_\+#\.\-]{2,}\b",
                        stop_words=all_stops,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.8,
                        sublinear_tf=True
                    )),
                    ("char", TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_df=0.9,
                        sublinear_tf=True
                    ))
                ])),
                ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.01, random_state=42))
            ])
        ),
    ]

    for name, pipe in experiments:
        scores = cross_val_score(pipe, texts, labels, cv=cv, n_jobs=1)
        print(f"Model: {name}", flush=True)
        print(f"Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})\n", flush=True)

if __name__ == '__main__':
    main()
