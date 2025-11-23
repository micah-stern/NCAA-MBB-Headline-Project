import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Tuple
import argparse


class HeadlineEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.smoothie = SmoothingFunction().method4  # Smoothing function for BLEU

    def calculate_bleu(
        self, references: List[List[str]], candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores (1-4) for the given references and candidates.

        Args:
            references: List of reference headlines (each as a list of tokens)
            candidates: List of generated headlines (each as a string)

        Returns:
            Dictionary containing BLEU-1 through BLEU-4 scores
        """
        # Tokenize candidates
        candidates_tok = [word_tokenize(cand.lower()) for cand in candidates]

        # Make sure references are in the right format (list of lists of tokens)
        refs = [
            [word_tokenize(ref.lower())] if isinstance(ref, str) else [ref]
            for ref in references
        ]

        # Calculate BLEU scores
        bleu_scores = {
            "bleu-1": corpus_bleu(
                refs,
                candidates_tok,
                weights=(1, 0, 0, 0),
                smoothing_function=self.smoothie,
            ),
            "bleu-2": corpus_bleu(
                refs,
                candidates_tok,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=self.smoothie,
            ),
            "bleu-3": corpus_bleu(
                refs,
                candidates_tok,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=self.smoothie,
            ),
            "bleu-4": corpus_bleu(
                refs,
                candidates_tok,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothie,
            ),
        }
        return bleu_scores

    def calculate_rouge(
        self, references: List[str], candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores (1, 2, L) for the given references and candidates.

        Args:
            references: List of reference headlines
            candidates: List of generated headlines

        Returns:
            Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        try:
            scores = self.rouge.get_scores(candidates, references, avg=True)
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"],
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

    def calculate_metrics(
        self, references: List[str], candidates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all metrics for the given references and candidates.

        Args:
            references: List of reference headlines
            candidates: List of generated headlines

        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}

        # Calculate BLEU scores
        bleu_scores = self.calculate_bleu(references, candidates)
        metrics.update(bleu_scores)

        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge(references, candidates)
        metrics.update(rouge_scores)

        return metrics


def load_data(reference_file: str, generated_file: str) -> Tuple[List[str], List[str]]:
    """
    Load reference and generated headlines from files.

    Args:
        reference_file: Path to file containing reference headlines
        generated_file: Path to file containing generated headlines

    Returns:
        Tuple of (references, candidates) lists
    """
    with open(reference_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    with open(generated_file, "r", encoding="utf-8") as f:
        candidates = [line.strip() for line in f if line.strip()]

    if len(references) != len(candidates):
        print(
            f"Warning: Mismatched number of references ({len(references)}) and candidates ({len(candidates)})"
        )
        # Truncate to the shorter list
        min_len = min(len(references), len(candidates))
        references = references[:min_len]
        candidates = candidates[:min_len]

    return references, candidates


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated headlines against references"
    )
    parser.add_argument(
        "--references",
        type=str,
        required=True,
        help="Path to file with reference headlines (one per line)",
    )
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="Path to file with generated headlines (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    references, candidates = load_data(args.references, args.generated)

    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = HeadlineEvaluator()

    # Calculate metrics
    print("Calculating metrics...")
    metrics = evaluator.calculate_metrics(references, candidates)

    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, score in metrics.items():
        print(f"{metric.upper()}: {score:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    # Install required packages if not already installed
    import subprocess
    import sys

    required_packages = ["nltk", "rouge", "pandas", "numpy"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Download NLTK data
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    main()
