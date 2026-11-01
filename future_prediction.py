"""
future_prediction.py
--------------------
Implements the Career Foresight and Future Job Opportunities Prediction
feature described in Section I of the IEEE research paper:
"AI-Powered Model for Intelligent Resume Recommendation and Feedback" (ICCCNT 2024).

"This predictive feature is allowing candidate to strategically plan their
career paths and stay ahead of evolving industry demands."
"""

from typing import Dict, List, Set
from feedback_engine import extract_skills_from_text


# Emerging & High-Growth Future Roles Mapping
FUTURE_CAREER_PATHS = [
    {
        "future_role": "Generative AI & LLM Systems Specialist",
        "market_demand": "Very High (Next 2-5 Years)",
        "prerequisite_triggers": ["python", "deep learning", "machine learning", "pytorch", "nlp"],
        "emerging_skills_needed": ["RAG Architecture", "LangChain / LlamaIndex", "Vector Databases (Pinecone/Chroma)", "Prompt Engineering", "Fine-Tuning (LoRA/QLoRA)"],
        "industry_outlook": "Exploding demand across enterprise tech, fintech, healthcare, and consumer software."
    },
    {
        "future_role": "MLOps / AI Platform Infrastructure Engineer",
        "market_demand": "High (Next 2-4 Years)",
        "prerequisite_triggers": ["python", "machine learning", "docker", "git", "linux"],
        "emerging_skills_needed": ["Kubeflow / MLflow", "CI/CD for ML Models", "Kubernetes", "Model Drift Monitoring", "Triton Inference Server"],
        "industry_outlook": "Critical bottleneck for companies moving AI prototypes into production at scale."
    },
    {
        "future_role": "Cloud-Native Full Stack Solutions Architect",
        "market_demand": "High (Next 3-5 Years)",
        "prerequisite_triggers": ["javascript", "react", "node.js", "rest api", "sql", "git"],
        "emerging_skills_needed": ["Next.js App Router", "Serverless Architecture (AWS Lambda)", "GraphQL Federation", "Micro-Frontends", "Terraform / IaC"],
        "industry_outlook": "Evolving beyond traditional web dev into distributed cloud systems and multi-tenant architectures."
    },
    {
        "future_role": "Modern Data Platform & Analytics Engineer",
        "market_demand": "Very High (Next 2-4 Years)",
        "prerequisite_triggers": ["sql", "python", "pandas", "tableau", "power bi", "excel"],
        "emerging_skills_needed": ["dbt (data build tool)", "Snowflake / BigQuery Modern Stack", "Apache Spark / Databricks", "Data Modeling & Mesh", "Airflow Orchestration"],
        "industry_outlook": "Shift from static dashboard reporting to automated real-time semantic data pipelines."
    },
    {
        "future_role": "Distributed Systems & High-Throughput Backend Lead",
        "market_demand": "Stable & High (Next 3-6 Years)",
        "prerequisite_triggers": ["data structures", "algorithms", "c++", "java", "oop", "system design"],
        "emerging_skills_needed": ["Apache Kafka / Event Streaming", "gRPC & Protocol Buffers", "Rust Programming", "Distributed Consensus (Raft)", "Observability (OpenTelemetry)"],
        "industry_outlook": "Crucial for high-frequency trading, cloud infrastructure, gaming backends, and networking systems."
    }
]


def predict_future_job_opportunities(resume_text: str) -> List[Dict]:
    """
    Predicts upcoming and adjacent high-growth job opportunities based on the
    candidate's existing skills set.
    """
    candidate_skills = extract_skills_from_text(resume_text)
    predictions = []

    for path in FUTURE_CAREER_PATHS:
        triggers = path["prerequisite_triggers"]
        # Count overlapping trigger skills
        overlap = [t for t in triggers if t in candidate_skills]
        readiness_pct = round((len(overlap) / max(len(triggers), 1)) * 100, 1)

        # Include if candidate has at least 1-2 foundational skills
        if len(overlap) >= 1:
            predictions.append({
                "future_role": path["future_role"],
                "market_demand": path["market_demand"],
                "readiness_score": readiness_pct,
                "foundational_skills_possessed": overlap,
                "recommended_emerging_skills": path["emerging_skills_needed"],
                "industry_outlook": path["industry_outlook"]
            })

    # Sort by candidate readiness
    predictions.sort(key=lambda x: x["readiness_score"], reverse=True)
    return predictions[:3]  # Return top 3 future role pathways
