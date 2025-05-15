import pandas as pd
import numpy as np
import os
import logging

PROCESSED_DATA_DIR = "data/processed/"
FEATURES_OUTPUT_DIR = "data/features/"
os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_item_features(merged_filename, output_filename):
    """
    Generate comprehensive item-level features for videos.
    """
    logger.info(f"Generating item features from {merged_filename}...")
    merged_path = os.path.join(PROCESSED_DATA_DIR, merged_filename)
    merged = pd.read_csv(merged_path)

    item_agg = merged.groupby("video_id").agg(
        {
            "watch_ratio": ["mean", "std", "count"],
            "like_cnt": ["mean", "sum"],
            "comment_cnt": ["mean", "sum"],
            "share_cnt": ["mean", "sum"],
        }
    )

    item_agg.columns = [f"{col[0]}_{col[1]}" for col in item_agg.columns]
    item_agg = item_agg.reset_index()

    # Calculate popularity metrics
    item_agg["popularity"] = item_agg["watch_ratio_count"]
    item_agg["popularity_score"] = (
        item_agg["popularity"] - item_agg["popularity"].min()
    ) / (item_agg["popularity"].max() - item_agg["popularity"].min() + 1e-10)

    # Calculate engagement score (weighted combination)
    w1, w2, w3, w4, w5 = 0.30, 0.10, 0.25, 0.15, 0.20  # Weights
    item_agg["engagement_score"] = (
        w1 * item_agg["watch_ratio_mean"]
        + w2
        * (
            item_agg["watch_ratio_std"] + 0.01
        )  # Add small constant to avoid zero values
        + w3 * item_agg["like_cnt_mean"]
        + w4 * item_agg["comment_cnt_mean"]
        + w5 * item_agg["share_cnt_mean"]
    )

    # Normalize engagement score
    item_agg["engagement_score"] = (
        item_agg["engagement_score"] - item_agg["engagement_score"].min()
    ) / (
        item_agg["engagement_score"].max() - item_agg["engagement_score"].min() + 1e-10
    )

    # Create a hybrid score (engagement + popularity)
    item_agg["hybrid_score"] = (
        0.7 * item_agg["engagement_score"] + 0.3 * item_agg["popularity_score"]
    )

    # Save to CSV
    output_path = os.path.join(FEATURES_OUTPUT_DIR, output_filename)
    item_agg.to_csv(output_path, index=False)
    logger.info(f"Saved item features to {output_path} with shape: {item_agg.shape}")

    return item_agg


if __name__ == "__main__":
    train_features = generate_item_features(
        "train_merged.csv", "item_engagement_features.csv"
    )
