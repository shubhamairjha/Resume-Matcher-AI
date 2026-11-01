"""
main.py
-------
One-command entry point: trains the job-role matching model on the
'<role> jobs' folders and immediately scores a resume against all 5 roles.

Example:
    python main.py \\
        --data-dir "/path/to/DRDO-internship-portal--main" \\
        --resume "/path/to/DRDO-internship-portal--main/Sample_AI_ML_Resume_Fictional.pdf"

Re-running with --skip-training will reuse an already-saved model.joblib
instead of retraining from scratch.
"""

import argparse
import os
import sys

import joblib

from train_model import train
from match import score_resume, print_results
from pdf_utils import extract_text_from_pdf


def main():
    parser = argparse.ArgumentParser(description="Train + run the AI job-role resume matcher.")
    parser.add_argument("--data-dir", default=".",
                         help="Folder containing the '<role> jobs' sub-folders")
    parser.add_argument("--resume", required=True,
                         help="Path to the resume PDF to score")
    parser.add_argument("--model-out", default="model.joblib",
                         help="Where to save/load the trained model")
    parser.add_argument("--skip-training", action="store_true",
                         help="Reuse an existing model.joblib instead of retraining")
    parser.add_argument("--tune", action="store_true",
                         help="Grid-search hyperparameters instead of using the pre-tuned defaults")
    args = parser.parse_args()

    if args.skip_training and os.path.exists(args.model_out):
        print(f"Loading existing model from: {args.model_out}")
        artifacts = joblib.load(args.model_out)
    else:
        artifacts = train(args.data_dir, args.model_out, tune=args.tune)

    resume_text = extract_text_from_pdf(args.resume)
    if len(resume_text.split()) < 5:
        print("[error] Could not extract meaningful text from the resume PDF.", file=sys.stderr)
        return 1

    results = score_resume(artifacts, resume_text)
    print_results(results, args.resume, resume_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
