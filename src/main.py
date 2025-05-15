"""
Main pipeline script for the video recommendation system.
This script orchestrates the entire workflow:
1. Preprocessing raw data
2. Generating engagement features
3. Building a content-based recommendation model
4. Generating recommendations
5. Evaluating the model performance
"""

import os
import logging
import argparse
import time
import pandas as pd
from pathlib import Path

# Setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = DATA_DIR / "results"

if not os.path.exists(RAW_DATA_DIR):
    raise FileNotFoundError(f"Raw data directory not found at {RAW_DATA_DIR}")

# Ensure directories exist
for directory in [PROCESSED_DATA_DIR, FEATURES_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main-pipeline")


def run_preprocessing(
    big_matrix="big_matrix.csv", small_matrix="small_matrix.csv", force=False
):
    """
    Preprocess raw data files into merged datasets for training and testing.
    """
    from src.data.preprocess import preprocess_matrix

    train_out = PROCESSED_DATA_DIR / "train_merged.csv"
    test_out = PROCESSED_DATA_DIR / "test_merged.csv"

    if force or not train_out.exists():
        logger.info("Preprocessing training data...")
        preprocess_matrix(big_matrix, "train_merged.csv")
    else:
        logger.info("Training data already preprocessed, skipping.")

    if force or not test_out.exists():
        logger.info("Preprocessing testing data...")
        preprocess_matrix(small_matrix, "test_merged.csv")
    else:
        logger.info("Testing data already preprocessed, skipping.")


def generate_features(force=False):
    """
    Generate engagement features for recommendation.
    """
    features_file = FEATURES_DIR / "item_engagement_features.csv"

    if force or not features_file.exists():
        logger.info("Generating item engagement features...")
        from src.data.item_engagement_features import generate_item_features

        generate_item_features("train_merged.csv", "item_engagement_features.csv")
    else:
        logger.info("Item features already exist, skipping generation.")


def train_and_recommend(top_n=10, force=False):
    """
    Train the content-based model and generate recommendations.
    """
    output_file = RESULTS_DIR / f"content_based_top{top_n}_recommendations.csv"

    if force or not output_file.exists():
        logger.info(f"Training model and generating top-{top_n} recommendations...")
        from src.recommenders.content_based_recommender import main as recommend_main

        recommend_main(top_n=top_n)
    else:
        logger.info(f"Recommendations already exist at {output_file}, skipping.")

    return output_file


def evaluate_model(k_values=[1, 3, 5, 10, 5000], watch_threshold=0.5, force=False):
    """
    Evaluate model performance using precision, recall, MAP, and NDCG.
    """
    eval_file = RESULTS_DIR / "content_based_evaluation_metrics.csv"

    if force or not eval_file.exists():
        logger.info(
            f"Evaluating model with k={k_values}, threshold={watch_threshold}..."
        )
        from src.evaluation.evaluate_recommendations import evaluate_model

        evaluate_model(k_values=k_values, watch_threshold=watch_threshold)
    else:
        logger.info(f"Evaluation already exists at {eval_file}, displaying results...")
        results = pd.read_csv(eval_file)
        print("\nEvaluation Results:")
        print(results)

        # Display a simple summary
        best_k = results.loc[results["precision"].idxmax()]["k"]
        print(f"\nBest precision at k={best_k}")

    return eval_file


def main(args):
    """Main pipeline execution"""
    start_time = time.time()
    logger.info("Starting recommendation pipeline...")

    if args.full_pipeline or args.preprocess:
        run_preprocessing(force=args.force)

    if args.full_pipeline or args.features:
        generate_features(force=args.force)

    if args.full_pipeline or args.recommend:
        train_and_recommend(top_n=args.top_n, force=args.force)

    if args.full_pipeline or args.evaluate:
        k_values = [int(k) for k in args.k_values.split(",")]
        evaluate_model(
            k_values=k_values, watch_threshold=args.threshold, force=args.force
        )

    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the video recommendation pipeline"
    )

    # Pipeline control arguments
    parser.add_argument(
        "--full-pipeline", action="store_true", help="Run the complete pipeline"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Run only preprocessing"
    )
    parser.add_argument(
        "--features", action="store_true", help="Run only feature generation"
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Run only model training and recommendation",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run only model evaluation"
    )

    # Parameter customization
    parser.add_argument(
        "--top-n", type=int, default=5000, help="Number of recommendations per user"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10,5000",
        help="Comma-separated list of k values for evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Watch ratio threshold for relevance",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all files"
    )

    args = parser.parse_args()

    # If no specific stage is selected, run the full pipeline
    if not any(
        [
            args.full_pipeline,
            args.preprocess,
            args.features,
            args.recommend,
            args.evaluate,
        ]
    ):
        args.full_pipeline = True

    main(args)
