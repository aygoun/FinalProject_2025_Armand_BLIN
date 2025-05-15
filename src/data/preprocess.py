import pandas as pd
import os
import logging
from IPython.display import display

RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_matrix(matrix_filename, output_filename):
    interactions = pd.read_csv(os.path.join(RAW_DATA_DIR, matrix_filename))

    user_features = pd.read_csv(os.path.join(RAW_DATA_DIR, "user_features.csv"))

    item_categories = pd.read_csv(os.path.join(RAW_DATA_DIR, "item_categories.csv"))
    item_categories["feat"] = item_categories["feat"].apply(
        lambda x: (
            [str(item) for item in eval(x)]
            if isinstance(x, str)
            else ([] if pd.isna(x) else [str(item) for item in x])
        )
    )

    # Load item daily features (contains per-day stats for each video)
    item_daily_features = pd.read_csv(
        os.path.join(RAW_DATA_DIR, "item_daily_features.csv")
    )

    # Merge user features
    merged = interactions.merge(user_features, on="user_id", how="left")

    # Merge item categories
    merged = merged.merge(item_categories, on="video_id", how="left")

    # Merge item daily features (aggregate by video_id)
    item_daily_agg = (
        item_daily_features.groupby("video_id").mean(numeric_only=True).reset_index()
    )
    merged = merged.merge(item_daily_agg, on="video_id", how="left")

    # Keep only useful cols
    merged = merged[
        [
            "user_id",
            "video_id",
            "watch_ratio",
            "like_cnt",
            "comment_cnt",
            "share_cnt",
            "feat",
        ]
    ]

    merged = merged.fillna(0)

    # Save processed data
    merged.to_csv(os.path.join(PROCESSED_DATA_DIR, output_filename), index=False)
    logger.info(f"Saved {output_filename} with shape: {merged.shape}")
    display(merged.head())


if __name__ == "__main__":
    logger.info("Preprocessing big matrix...")
    preprocess_matrix("big_matrix.csv", "train_merged.csv")
    logger.info("Preprocessing small matrix...")
    preprocess_matrix("small_matrix.csv", "test_merged.csv")
