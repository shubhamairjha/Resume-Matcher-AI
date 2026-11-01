"""
feedback_engine.py
------------------
Implements the Intelligent Candidate Feedback Mechanism described in the
research paper: "AI-Powered Model for Intelligent Resume Recommendation and Feedback"
(IEEE ICCCNT 2024).

Features:
  1. Skill Extraction & Gap Analysis:
     Extracts candidate skills and compares them against target job prerequisites.
  2. Candidate Status Determination:
     - "Selected for Interview / Strong Fit"
     - "Conditional Candidate / Bridgeable Skills Gap"
     - "Reasons Candidate Not Selected"
  3. Actionable Recommendations:
     Provides specific technical skills, projects, and certifications needed to
     maximize hiring alignment.
"""

import re
from typing import Dict, List, Set, Tuple

# Comprehensive Skill Taxonomy for the 5 Roles
ROLE_REQUIREMENTS = {
    "AIML jobs": {
        "title": "AI / Machine Learning Engineer",
        "core_skills": [
            "python", "pytorch", "tensorflow", "scikit-learn", "deep learning",
            "machine learning", "neural networks", "nlp", "computer vision",
            "transformers", "large language models", "llm", "generative ai",
            "sql", "pandas", "numpy", "mlops", "langchain", "docker"
        ],
        "nice_to_have": ["xgboost", "opencv", "rag", "huggingface", "aws sagemaker", "keras"],
        "recommended_certifications": [
            "DeepLearning.AI Deep Learning Specialization",
            "AWS Certified Machine Learning - Specialty",
            "TensorFlow Developer Certificate"
        ],
        "recommended_projects": [
            "Build an End-to-End RAG / LLM application with LangChain and vector DB",
            "Train and deploy a fine-tuned transformer or vision model using Docker and FastAPI",
            "Implement an MLOps pipeline with CI/CD, experiment tracking, and model monitoring"
        ]
    },
    "Full stack jobs": {
        "title": "Full Stack Web Developer",
        "core_skills": [
            "javascript", "typescript", "react", "html", "css", "node.js",
            "express", "rest api", "sql", "mongodb", "postgresql", "git",
            "responsive design", "tailwind", "next.js"
        ],
        "nice_to_have": ["graphql", "docker", "redux", "vue.js", "ci/cd", "aws", "flutter"],
        "recommended_certifications": [
            "Meta Front-End / Back-End Developer Professional Certificate",
            "AWS Certified Developer - Associate",
            "Full Stack Open by University of Helsinki"
        ],
        "recommended_projects": [
            "Full-stack SaaS app with Next.js, TypeScript, PostgreSQL, and Stripe integration",
            "Real-time collaborative application using WebSockets, React, and Node.js",
            "Containerized microservices backend with Docker and REST/GraphQL APIs"
        ]
    },
    "Software engineer jobs": {
        "title": "Software Engineer (Systems & Backend)",
        "core_skills": [
            "data structures", "algorithms", "c++", "java", "c#", "python",
            "system design", "microservices", "oop", "git", "linux", "sql",
            "unit testing", "ci/cd", "distributed systems"
        ],
        "nice_to_have": ["docker", "kubernetes", "spring boot", "multithreading", "low latency", "go"],
        "recommended_certifications": [
            "Oracle Certified Professional: Java SE Developer",
            "CKA: Certified Kubernetes Administrator",
            "AWS Certified Solutions Architect"
        ],
        "recommended_projects": [
            "High-throughput concurrent distributed key-value store in C++ or Go",
            "Scalable microservices backend with Spring Boot, Kafka, and Redis caching",
            "Implementation of core algorithms, custom memory allocators, and concurrency primitives"
        ]
    },
    "data analyst jobs": {
        "title": "Data Analyst / BI Specialist",
        "core_skills": [
            "sql", "excel", "tableau", "power bi", "python", "pandas",
            "data visualization", "exploratory data analysis", "eda",
            "business intelligence", "statistics", "data cleaning", "reporting", "kpis"
        ],
        "nice_to_have": ["snowflake", "bigquery", "etl", "data warehousing", "r", "dbt", "looker"],
        "recommended_certifications": [
            "Google Data Analytics Professional Certificate",
            "Microsoft Certified: Power BI Data Analyst Associate (PL-300)",
            "Tableau Desktop Specialist"
        ],
        "recommended_projects": [
            "Interactive executive dashboard in Power BI / Tableau analyzing sales or customer churn",
            "End-to-end ETL data pipeline in Python and SQL feeding a star-schema data warehouse",
            "Advanced statistical exploratory data analysis report answering business questions"
        ]
    },
    "python jobs": {
        "title": "Backend Python Developer",
        "core_skills": [
            "python", "django", "flask", "fastapi", "sql", "postgresql",
            "rest api", "git", "pytest", "unit testing", "orm", "sqlalchemy",
            "docker", "redis", "asyncio"
        ],
        "nice_to_have": ["celery", "web scraping", "beautifulsoup", "pandas", "linux", "ci/cd"],
        "recommended_certifications": [
            "PCAP - Certified Associate in Python Programming",
            "AWS Certified Developer",
            "Test-Driven Development with Python & Django"
        ],
        "recommended_projects": [
            "Asynchronous REST API service built with FastAPI, PostgreSQL, and Redis caching",
            "Distributed task queuing system with Celery and RabbitMQ/Redis in Python",
            "Automated web scraper and data processing pipeline with BeautifulSoup and Scrapy"
        ]
    }
}


def extract_skills_from_text(text: str) -> Set[str]:
    """Scan candidate text for all technical skills defined across roles."""
    text_lower = " " + text.lower() + " "
    found_skills = set()

    for role_info in ROLE_REQUIREMENTS.values():
        all_skills = role_info["core_skills"] + role_info["nice_to_have"]
        for skill in all_skills:
            # Handle special symbols
            esc = re.escape(skill)
            pattern = r"(?:^|[\s,;/()\[\]])" + esc + r"(?:$|[\s,;/()\[\]])"
            if re.search(pattern, text_lower):
                found_skills.add(skill)

    # Common variations
    if re.search(r"\bc\+\+\b", text_lower):
        found_skills.add("c++")
    if re.search(r"\bc#\b", text_lower):
        found_skills.add("c#")
    if re.search(r"\bpower\s*bi\b", text_lower):
        found_skills.add("power bi")
    if re.search(r"\bnode(?:\.js)?\b", text_lower):
        found_skills.add("node.js")
    if re.search(r"\breact(?:\.js)?\b", text_lower):
        found_skills.add("react")
    if re.search(r"\bnext(?:\.js)?\b", text_lower):
        found_skills.add("next.js")
    if re.search(r"\bdeep\s*learning\b", text_lower):
        found_skills.add("deep learning")
    if re.search(r"\bmachine\s*learning\b", text_lower):
        found_skills.add("machine learning")
    return found_skills


def generate_candidate_feedback(role: str, resume_text: str, ml_match_pct: float) -> Dict:
    """
    Generates intelligent structured feedback for a candidate against a target role.
    Implements the IEEE research paper feedback criteria.
    """
    role_info = ROLE_REQUIREMENTS.get(role, ROLE_REQUIREMENTS["AIML jobs"])
    core_reqs = role_info["core_skills"]
    candidate_skills = extract_skills_from_text(resume_text)

    matched = [s for s in core_reqs if s in candidate_skills]
    missing = [s for s in core_reqs if s not in candidate_skills]

    skill_coverage_pct = round((len(matched) / max(len(core_reqs), 1)) * 100, 1)

    # Determine selection status as per Research Paper Flowchart (Figure 1)
    if ml_match_pct >= 40.0 and skill_coverage_pct >= 35.0:
        status = "HIGH ALIGNMENT / SHORTLIST RECOMMENDED"
        verdict_msg = (
            f"Your profile strongly aligns with the {role_info['title']} role. "
            f"You possess key required proficiencies ({', '.join(matched[:4])})."
        )
    elif ml_match_pct >= 20.0 or skill_coverage_pct >= 25.0:
        status = "CONDITIONAL FIT / SKILLS GAP DETECTED"
        verdict_msg = (
            f"Your profile shows foundational relevance for {role_info['title']}, "
            f"but has notable prerequisite gaps."
        )
    else:
        status = "NOT SELECTED FOR THIS ROLE (SKILLS MISMATCH)"
        verdict_msg = (
            f"Reason for Non-Selection: Key required skill sets for {role_info['title']} "
            f"are missing from the submitted resume."
        )

    return {
        "role": role,
        "role_title": role_info["title"],
        "status": status,
        "verdict_msg": verdict_msg,
        "ml_match_pct": ml_match_pct,
        "skill_coverage_pct": skill_coverage_pct,
        "matched_skills": matched,
        "missing_skills": missing,
        "recommended_certifications": role_info["recommended_certifications"],
        "recommended_projects": role_info["recommended_projects"],
    }
