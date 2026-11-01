"""
match.py
--------
Scores a resume PDF against the 5 trained job roles.

Supports all model types saved by train_model.py:
  - sbert_hybrid: SBERT embeddings + skill features
  - tfidf_char_svc: TF-IDF (word+char) + LinearSVC
  - tfidf_svd_logreg: original TF-IDF + SVD + LogReg

Outputs:
  1. ML Match %          - Classifier probability distribution
  2. Semantic Similarity % - Cosine similarity in embedding space
  3. Final Match %       - Weighted combination
  4. Candidate Feedback  - Strengths, gaps, recommendations
  5. Future Roles        - Career path predictions
  6. Explanation         - Why this role was matched

Usage:
    python match.py --model model.joblib --resume "resume.pdf"
"""

import argparse
import re
import sys
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from pdf_utils import extract_text_from_pdf
from feedback_engine import generate_candidate_feedback
from future_prediction import predict_future_job_opportunities


# =============================================================================
# TEXT CLEANING (mirrors train_model.py)
# =============================================================================

def clean_text_light(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\uf000-\uf8ff\ufffd\u2022\u25cf\u25aa]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_tfidf(text: str) -> str:
    text = clean_text_light(text)
    text = text.lower()
    text = re.sub(r"\bc\+\+\b", "cpp_lang", text)
    text = re.sub(r"\bc#\b", "csharp_lang", text)
    text = re.sub(r"\.net\b", "dotnet_framework", text)
    text = re.sub(r"\bnode\.js\b", "nodejs", text)
    text = re.sub(r"\bnext\.js\b", "nextjs", text)
    text = re.sub(r"\breact\.js\b", "reactjs", text)
    text = re.sub(r"\bvue\.js\b", "vuejs", text)
    text = re.sub(r"\bpower\s*bi\b|\bpowerbi\b", "powerbi_tool", text)
    return text


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def safe_predict_proba(clf, X: np.ndarray) -> np.ndarray:
    """Robustly compute probabilities across different scikit-learn versions and unpickled objects."""
    for attr, val in [("multi_class", "auto"), ("multi_class_", "auto")]:
        if not hasattr(clf, attr):
            try:
                setattr(clf, attr, val)
            except Exception:
                pass
    try:
        return clf.predict_proba(X)[0]
    except Exception:
        # Fallback: compute softmax from decision_function or coefficients if predict_proba raises AttributeError
        try:
            decision = clf.decision_function(X)
            if decision.ndim == 1:
                decision = np.vstack([-decision, decision]).T
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            proba = exp_d / np.sum(exp_d, axis=1, keepdims=True)
            return proba[0]
        except Exception:
            if hasattr(clf, "coef_") and hasattr(clf, "intercept_"):
                raw = np.dot(X, clf.coef_.T) + clf.intercept_
                exp_r = np.exp(raw - np.max(raw, axis=1, keepdims=True))
                proba = exp_r / np.sum(exp_r, axis=1, keepdims=True)
                return proba[0]
            n_classes = len(clf.classes_) if hasattr(clf, "classes_") else 5
            return np.ones(n_classes) / n_classes


def score_resume_sbert(artifacts: dict, resume_text: str) -> list:
    """Score resume using SBERT or SBERT hybrid model."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    sbert = artifacts["sbert"]
    clf = artifacts["classifier"]
    roles = list(clf.classes_)
    model_type = artifacts.get("model_type", "sbert_logreg")

    # Embed resume
    emb = sbert.encode([resume_text], batch_size=1, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)

    if model_type == "sbert_hybrid" and "skill_extractor" in artifacts:
        skill_ext = artifacts["skill_extractor"]
        skill_feats = skill_ext.transform([resume_text])
        skill_feats = np.clip(skill_feats, 0, 1)
        resume_vec = np.concatenate([emb, skill_feats], axis=1)
    else:
        resume_vec = emb

    # 1. Classifier probabilities (robust)
    proba = safe_predict_proba(clf, resume_vec)
    ml_match = dict(zip(roles, proba * 100))

    # 2. Cosine similarity in embedding space
    posting_vecs = artifacts["posting_vectors"]
    posting_labels = artifacts["posting_labels"]
    sims = cosine_similarity(resume_vec, posting_vecs)[0]

    sim_match = {}
    for role in roles:
        role_sims = sims[posting_labels == role]
        sim_match[role] = float(role_sims.mean() * 100) if len(role_sims) > 0 else 0.0

    results = []
    for role in roles:
        ml_pct  = round(float(ml_match[role]), 1)
        sim_pct = round(sim_match[role], 1)
        # Final match: 70% ML + 30% similarity
        final_pct = round(0.70 * ml_pct + 0.30 * sim_pct, 1)
        results.append({
            "role": role,
            "ml_match_pct": ml_pct,
            "similarity_pct": sim_pct,
            "final_match_pct": final_pct,
        })
    results.sort(key=lambda r: r["final_match_pct"], reverse=True)
    return results


def score_resume_tfidf(artifacts: dict, resume_text: str) -> list:
    """Score resume using TF-IDF (word+char) + LinearSVC model."""
    model_type = artifacts.get("model_type", "tfidf_svd_logreg")

    if model_type == "tfidf_char_svc":
        vec_w = artifacts["vectorizer_word"]
        vec_c = artifacts["vectorizer_char"]
        clf   = artifacts["classifier"]
        roles = list(clf.classes_)

        from scipy.sparse import hstack
        Xw = vec_w.transform([resume_text])
        Xc = vec_c.transform([resume_text])
        resume_vec = hstack([Xw, Xc]).toarray()
    else:
        # Original TF-IDF + SVD + LogReg
        vectorizer = artifacts["vectorizer"]
        svd        = artifacts.get("svd")
        normalizer = artifacts.get("normalizer")
        clf        = artifacts["classifier"]
        roles      = list(clf.classes_)

        resume_vec = vectorizer.transform([resume_text])
        if svd is not None:
            resume_vec = svd.transform(resume_vec)
        if normalizer is not None:
            resume_vec = normalizer.transform(resume_vec)
        if hasattr(resume_vec, "toarray"):
            resume_vec = resume_vec.toarray()

    proba = safe_predict_proba(clf, resume_vec)
    ml_match = dict(zip(roles, proba * 100))

    posting_vecs   = np.array(artifacts["posting_vectors"])
    posting_labels = artifacts["posting_labels"]
    sims = cosine_similarity(resume_vec, posting_vecs)[0]

    sim_match = {}
    for role in roles:
        role_sims = sims[posting_labels == role]
        sim_match[role] = float(role_sims.mean() * 100) if len(role_sims) > 0 else 0.0

    results = []
    for role in roles:
        ml_pct  = round(float(ml_match[role]), 1)
        sim_pct = round(sim_match[role], 1)
        final_pct = round(0.70 * ml_pct + 0.30 * sim_pct, 1)
        results.append({
            "role": role,
            "ml_match_pct": ml_pct,
            "similarity_pct": sim_pct,
            "final_match_pct": final_pct,
        })
    results.sort(key=lambda r: r["final_match_pct"], reverse=True)
    return results


def score_resume(artifacts: dict, resume_text: str) -> list:
    """Route to the correct scoring function based on model type."""
    model_type = artifacts.get("model_type", "tfidf_svd_logreg")

    if model_type in ("sbert_logreg", "sbert_hybrid"):
        text_clean = clean_text_light(resume_text)
        return score_resume_sbert(artifacts, text_clean)
    else:
        text_clean = clean_text_for_tfidf(resume_text)
        return score_resume_tfidf(artifacts, text_clean)


# =============================================================================
# EXPLANATION GENERATION
# =============================================================================

ROLE_KEYWORDS = {
    "AIML jobs": [
        "machine learning", "deep learning", "neural", "pytorch", "tensorflow",
        "scikit", "nlp", "computer vision", "transformer", "llm", "bert", "gpt",
        "generative", "mlops", "model", "training", "data science", "ai"
    ],
    "Full stack jobs": [
        "javascript", "typescript", "react", "angular", "vue", "html", "css",
        "node", "express", "rest", "api", "mongodb", "postgresql", "frontend",
        "backend", "full stack", "fullstack", "web", "graphql"
    ],
    "Software engineer jobs": [
        "algorithms", "data structures", "system design", "java", "c++", "c#",
        "microservices", "distributed", "kubernetes", "docker", "linux", "oop",
        "design patterns", "concurrency", "backend", "engineering"
    ],
    "data analyst jobs": [
        "sql", "excel", "tableau", "power bi", "visualization", "analytics",
        "dashboard", "reporting", "kpi", "business intelligence", "etl",
        "statistics", "data cleaning", "pandas", "matplotlib", "bi"
    ],
    "python jobs": [
        "python", "django", "flask", "fastapi", "pytest", "sqlalchemy",
        "celery", "redis", "asyncio", "orm", "pydantic", "scraping", "api"
    ]
}


def generate_explanation(resume_text: str, results: list) -> dict:
    """Generate human-readable match explanation from resume content."""
    best_role = results[0]["role"]
    text_lower = resume_text.lower()

    # Matched keywords for the top role
    role_kws  = ROLE_KEYWORDS.get(best_role, [])
    matched   = [kw for kw in role_kws if kw in text_lower]

    # Semantic strength signals
    strength_phrases = []
    if len(matched) >= 5:
        strength_phrases.append(f"Strong keyword alignment with {best_role}")
    if results[0]["similarity_pct"] > 50:
        strength_phrases.append("High semantic similarity to reference job postings")
    if results[0]["ml_match_pct"] > 40:
        strength_phrases.append("Classifier confidence above threshold")

    # What distinguishes this role from next-best
    gap = []
    if len(results) > 1:
        second = results[1]["role"]
        second_kws = ROLE_KEYWORDS.get(second, [])
        overlap_second = [kw for kw in second_kws if kw in text_lower]
        if len(overlap_second) > len(matched):
            gap.append(f"Note: Strong overlap with {second} as well")

    return {
        "best_role": best_role,
        "matched_keywords": matched[:12],
        "strength_signals": strength_phrases,
        "notes": gap,
    }


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(results: list, resume_path: str, resume_text: str = "",
                  model_type: str = "unknown"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n" + "=" * 70)
    print(f"RESUME MATCHING & RECOMMENDATION SYSTEM")
    print(f"Resume: {resume_path}")
    print(f"Model:  {model_type}")
    print("=" * 70)

    header = f"{'Job Role':<26s} {'ML Match%':>11s} {'Similarity%':>13s} {'Final Match%':>14s}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        marker = " [TOP MATCH]" if r == results[0] else ""
        print(
            f"{r['role']:<26s} "
            f"{r['ml_match_pct']:>10.1f}% "
            f"{r['similarity_pct']:>12.1f}% "
            f"{r['final_match_pct']:>13.1f}%"
            f"{marker}"
        )

    best_role = results[0]["role"]
    best_ml   = results[0]["ml_match_pct"]
    best_sim  = results[0]["similarity_pct"]
    best_fin  = results[0]["final_match_pct"]

    print(f"\n>> BEST-FIT RECOMMENDATION: {best_role}")
    print(f"   ML Match: {best_ml:.1f}%  |  Similarity: {best_sim:.1f}%  |  Final: {best_fin:.1f}%")

    # ── Explanation ──────────────────────────────────────────────────────────
    if resume_text:
        expl = generate_explanation(resume_text, results)
        print("\n" + "=" * 70)
        print("MATCH EXPLANATION")
        print("=" * 70)
        if expl["matched_keywords"]:
            print("\n[+] Strong Semantic Matches Found in Resume:")
            for kw in expl["matched_keywords"]:
                print(f"    - {kw}")
        else:
            print("\n[-] Limited exact keyword matches (relying on semantic embedding).")
        if expl["strength_signals"]:
            print("\n[+] Match Strength Signals:")
            for sig in expl["strength_signals"]:
                print(f"    - {sig}")
        if expl["notes"]:
            print("\n[!] Notes:")
            for note in expl["notes"]:
                print(f"    - {note}")

    # ── Feedback ─────────────────────────────────────────────────────────────
    feedback = generate_candidate_feedback(best_role, resume_text, best_ml)

    print("\n" + "=" * 70)
    print("CONSTRUCTIVE CANDIDATE FEEDBACK & GAP ANALYSIS")
    print("(IEEE Research Paper Feedback Mechanism)")
    print("=" * 70)
    print(f"Target Role:           {feedback['role_title']}")
    print(f"Selection Status:      {feedback['status']}")
    print(f"Verdict:               {feedback['verdict_msg']}")
    print(f"Skill Coverage:        {feedback['skill_coverage_pct']}% of core requirements")

    print("\n[+] Matched Core Skills:")
    if feedback["matched_skills"]:
        for sk in feedback["matched_skills"]:
            print(f"    - {sk}")
    else:
        print("    - None explicitly identified.")

    print("\n[-] Missing / Prerequisite Skills:")
    if feedback["missing_skills"]:
        for sk in feedback["missing_skills"][:6]:
            print(f"    - {sk}")
    else:
        print("    - All core prerequisites satisfied!")

    print("\n[*] Recommended Certifications:")
    for cert in feedback["recommended_certifications"]:
        print(f"    - {cert}")

    print("\n[*] Recommended Projects:")
    for proj in feedback["recommended_projects"]:
        print(f"    - {proj}")

    # ── Future Roles ─────────────────────────────────────────────────────────
    future_jobs = predict_future_job_opportunities(resume_text)

    print("\n" + "=" * 70)
    print("PREDICTIVE FUTURE JOB OPPORTUNITIES")
    print("=" * 70)

    for i, fj in enumerate(future_jobs, 1):
        print(f"\n{i}. Role: {fj['future_role']}")
        print(f"   Market Demand:       {fj['market_demand']}")
        print(f"   Candidate Readiness: {fj['readiness_score']:.0f}%")
        print(f"   Skills Possessed:    {', '.join(fj['foundational_skills_possessed'])}")
        print(f"   Emerging Skills:     {', '.join(fj['recommended_emerging_skills'][:4])}")
        print(f"   Outlook:             {fj['industry_outlook']}")

    print("\n" + "=" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score a resume against trained job roles with feedback."
    )
    parser.add_argument("--model",  default="model.joblib", help="Path to trained model .joblib")
    parser.add_argument("--resume", required=True, help="Path to the resume PDF")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    artifacts    = joblib.load(args.model)
    resume_text  = extract_text_from_pdf(args.resume)
    model_type   = artifacts.get("model_type", "unknown")
    cv_acc       = artifacts.get("cv_accuracy", None)

    if cv_acc is not None:
        print(f"[Model CV Accuracy: {cv_acc:.1%} +/- {artifacts.get('cv_std', 0):.1%}]")

    if len(resume_text.split()) < 5:
        print("[error] Could not extract meaningful text from resume PDF.", file=sys.stderr)
        return 1

    results = score_resume(artifacts, resume_text)
    print_results(results, args.resume, resume_text, model_type=model_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
