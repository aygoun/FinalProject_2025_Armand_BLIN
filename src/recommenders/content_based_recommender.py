import pandas as pd
import os
import numpy as np
from src.models.content_model import ContentBasedRecommender
import logging

RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
FEATURES_DIR = "data/features/"
RESULTS_DIR = "data/results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_WEIGHTS = {
    "engagement_score": 3.5,
    "hybrid_score": 3.0,
    "popularity_score": 15,
    "feat": 3.5,  # tags
}


def load_data():
    """Load and return train and test data."""
    train_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train_merged.csv"))
    logger.info(f"Loaded train data with {len(train_data)} interactions")

    test_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_merged.csv"))
    logger.info(f"Loaded test data with {len(test_data)} interactions")

    return train_data, test_data


def load_item_features():
    """Load and process item features."""
    item_features = pd.read_csv(
        os.path.join(FEATURES_DIR, "item_engagement_features.csv")
    )
    logger.info(f"Loaded item features with {len(item_features)} items")

    return item_features


def generate_recommendations(recommender, test_users, top_n=10):
    """Generate recommendations for test users."""
    logger.info(f"Generating recommendations for {len(test_users)} users in test set")

    recommendations = []
    for i, user_id in enumerate(test_users):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(test_users)} users")

        recs = recommender.recommend(
            user_id,
            n_items=top_n,
            exclude_watched=True,
            diversity_factor=0.2,
        )

        for rank, (video_id, score) in enumerate(recs, 1):
            recommendations.append(
                {"user_id": user_id, "video_id": video_id, "rank": rank, "score": score}
            )

    return recommendations


def save_recommendations(recommendations, top_n):
    """Save recommendations to CSV and print statistics."""
    recs_df = pd.DataFrame(recommendations)
    output_path = os.path.join(
        RESULTS_DIR, f"content_based_top{top_n}_recommendations.csv"
    )
    recs_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(recs_df)} recommendations to {output_path}")

    return recs_df


def print_recommendation_stats(recs_df, test_users):
    """Print statistics about recommendations."""
    recs_per_user = recs_df.groupby("user_id").size()
    logger.info(f"Average recommendations per user: {recs_per_user.mean():.2f}")
    logger.info(f"Min recommendations per user: {recs_per_user.min()}")
    logger.info(
        f"Recommendation coverage: {len(recs_per_user)}/{len(test_users)} users ({100*len(recs_per_user)/len(test_users):.2f}%)"
    )


def main(top_n=10):
    train_data, test_data = load_data()

    video_features = load_item_features()

    recommender = ContentBasedRecommender(feature_weights=FEATURE_WEIGHTS)
    recommender.fit(video_features, train_data)

    test_users = test_data["user_id"].unique()
    recommendations = generate_recommendations(recommender, test_users, top_n)

    recs_df = save_recommendations(recommendations, top_n)

    print_recommendation_stats(recs_df, test_users)
    logger.info(f"Generated recommendations with weights: {FEATURE_WEIGHTS}")


if __name__ == "__main__":
    main(top_n=10)
