"""
dataset.py
----------
Builds the training dataset for the job-role matcher.

Supports two data sources:
  1. Job posting PDF directories:
     <data_dir>/
         AIML jobs/
         Full stack jobs/
         Software engineer jobs/
         data analyst jobs/
         python jobs/
  2. Cached JSON dataset (`data/job_postings.json`) as a reliable fallback.
"""

import json
import os
from dataclasses import dataclass
from typing import List

from pdf_utils import extract_text_from_pdf


@dataclass
class Example:
    text: str
    label: str
    source_file: str


def load_job_role_dataset(data_dir: str = ".") -> List[Example]:
    """Walk every job-role sub-folder under data_dir and extract training examples.
    If no role folders are found, falls back to `data/job_postings.json`."""
    examples: List[Example] = []

    role_folders = []
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        role_folders = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.lower().endswith("jobs")
        )

    if role_folders:
        for role in role_folders:
            role_path = os.path.join(data_dir, role)
            pdf_files = sorted(f for f in os.listdir(role_path) if f.lower().endswith(".pdf"))

            for pdf_file in pdf_files:
                pdf_path = os.path.join(role_path, pdf_file)
                try:
                    text = extract_text_from_pdf(pdf_path)
                except Exception as exc:
                    print(f"  [skip] {pdf_path}: {exc}")
                    continue

                if len(text.split()) < 5:
                    print(f"  [skip] {pdf_path}: too little extractable text")
                    continue

                examples.append(Example(text=text, label=role, source_file=pdf_file))
        return examples

    # Fallback to cached JSON dataset
    json_path = os.path.join(os.path.dirname(__file__), "data", "job_postings.json")
    if not os.path.exists(json_path):
        json_path = os.path.join("data", "job_postings.json")

    if os.path.exists(json_path):
        print(f"Loading cached job postings from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            examples.append(Example(
                text=item["text"],
                label=item["label"],
                source_file=f"posting_{i+1}.json"
            ))
        return examples

    raise ValueError(f"No job-role folders found under {data_dir!r} and no cached dataset at {json_path!r}")


def dataset_summary(examples: List[Example]) -> str:
    counts = {}
    for ex in examples:
        counts[ex.label] = counts.get(ex.label, 0) + 1
    lines = [f"  {role:<25s} {n:>3d} postings" for role, n in sorted(counts.items())]
    return "\n".join(lines)
