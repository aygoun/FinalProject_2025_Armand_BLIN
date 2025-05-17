import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = "data/processed/"


class ContentBasedRecommender:
    def __init__(self, feature_weights=None):
        self.video_features = None
        self.user_profiles = None
        self.video_ids = None
        self.feature_weights = feature_weights or {}
        self.already_watched = {}
        self.scaler = StandardScaler()

    def fit(self, video_features_df, user_interactions_df):
        """
        Build content profiles and user profiles

        Parameters:
        -----------
        video_features_df: DataFrame with video_id, engagement_score, tags, interaction_count
        user_interactions_df: DataFrame with user_id, video_id, watch_ratio
        """
        logger.info("Building content-based recommender...")

        # Store user interactions for filtering recommendations
        for user_id, group in user_interactions_df.groupby("user_id"):
            self.already_watched[user_id] = set(group["video_id"].values)

        # Process video features with standardization and weighting
        self.video_features = self._process_video_features(video_features_df)
        self.video_ids = self.video_features.index.tolist()

        # Build user profiles based on watched videos
        self.user_profiles = self._build_user_profiles(user_interactions_df)

        logger.info(
            f"Content-based model built with {len(self.video_ids)} videos and {len(self.user_profiles)} users"
        )

    def _process_video_features(self, video_df):
        """Process video features including standardization and weighting"""
        # Clone the dataframe to avoid modifying the original
        processed_df = video_df.copy()

        if "feat" in processed_df.columns:
            from sklearn.preprocessing import MultiLabelBinarizer

            mlb = MultiLabelBinarizer(sparse_output=False)

            # Handle empty DataFrames or all-empty lists
            if len(processed_df) > 0 and any(len(x) > 0 for x in processed_df["feat"]):
                tags_encoded = pd.DataFrame(
                    mlb.fit_transform(processed_df["feat"]),
                    columns=mlb.classes_,
                    index=(
                        processed_df.index
                        if "video_id" not in processed_df.columns
                        else None
                    ),
                )

                processed_df = pd.concat(
                    [processed_df.drop(columns=["feat"]), tags_encoded], axis=1
                )

        processed_df = processed_df.set_index("video_id")

        # Identify numeric columns for standardization
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            try:
                # Apply standardization to numeric features
                processed_df[numeric_cols] = self.scaler.fit_transform(
                    processed_df[numeric_cols]
                )
            except ValueError as e:
                logger.error(f"Error standardizing data: {e}")

        if self.feature_weights:
            for feature, weight in self.feature_weights.items():
                if feature in processed_df.columns:
                    processed_df[feature] *= weight
                elif feature == "tags" and "feat" in video_df.columns:
                    tag_cols = [
                        col
                        for col in processed_df.columns
                        if col not in numeric_cols and col not in ["video_id"]
                    ]
                    for col in tag_cols:
                        if col in processed_df.columns:
                            processed_df[col] *= weight

        # Prioritize hybrid_score over engagement_score
        important_features = ["hybrid_score", "engagement_score", "popularity_score"]
        for i, feat in enumerate(important_features):
            if feat in processed_df.columns:
                boost_factor = 3.0 - (i * 0.5)
                processed_df[feat] *= boost_factor

        # Fill NaN
        processed_df = processed_df.fillna(0)

        return processed_df

    def _build_user_profiles(self, interactions_df):
        """Build user profiles based on videos they've watched with watch ratio weighting"""
        user_profiles = {}

        # Group by user_id and create a weighted average of video features
        for user_id, group in interactions_df.groupby("user_id"):
            # Get the videos this user has watched
            watched_videos = group["video_id"].tolist()

            # Get watch ratios as weights, apply non-linear transformation to emphasize high watch ratios
            weights = np.power(group["watch_ratio"].values, 2)

            if len(watched_videos) == 0:
                continue

            valid_indices = [
                i
                for i, v in enumerate(watched_videos)
                if v in self.video_features.index
            ]
            if not valid_indices:
                continue

            valid_videos = [watched_videos[i] for i in valid_indices]
            valid_weights = weights[valid_indices]

            # Normalize weights
            valid_weights = (
                valid_weights / np.sum(valid_weights)
                if np.sum(valid_weights) > 0
                else np.ones_like(valid_weights) / len(valid_weights)
            )

            # Get feature vectors for watched videos
            video_vectors = self.video_features.loc[valid_videos].values

            # Calculate weighted average
            user_profile = np.average(video_vectors, axis=0, weights=valid_weights)
            user_profiles[user_id] = user_profile

        return user_profiles

    def recommend(
        self, user_id, n_items=10, exclude_watched=True, diversity_factor=0.2
    ):
        """
        Recommend items for a user with optional diversity
        diversity_factor: How much to diversify recommendations (0-1)

        Returns:
            List of (video_id, similarity_score) tuples
        """
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found in model")
            return []

        user_profile = self.user_profiles[user_id]

        similarities = cosine_similarity([user_profile], self.video_features.values)[0]

        # Create (video_id, similarity) pairs
        video_scores = list(zip(self.video_ids, similarities))

        # Filter out already watched videos if requested
        if exclude_watched and user_id in self.already_watched:
            watched = self.already_watched[user_id]
            video_scores = [
                (vid, score) for vid, score in video_scores if vid not in watched
            ]

        if not video_scores:
            return []

        # Apply diversity - select top items, then some random items from next tier
        if diversity_factor > 0 and len(video_scores) > n_items * 2:
            video_scores.sort(key=lambda x: x[1], reverse=True)

            # Split recommendations: top ones and diverse ones
            top_count = int(n_items * (1 - diversity_factor))
            top_recommendations = video_scores[:top_count]

            # Get diverse recommendations from the next tier
            diversity_pool = video_scores[top_count : top_count + n_items * 3]
            if diversity_pool:
                # Add some randomness in selection from the pool
                diverse_count = n_items - top_count
                indices = np.random.choice(
                    len(diversity_pool),
                    size=min(diverse_count, len(diversity_pool)),
                    replace=False,
                )
                diverse_recommendations = [diversity_pool[i] for i in indices]

                # Combine and sort again
                recommendations = top_recommendations + diverse_recommendations
                recommendations.sort(key=lambda x: x[1], reverse=True)
                return recommendations[:n_items]

        # If no diversity needed or not enough items for diversity
        video_scores.sort(key=lambda x: x[1], reverse=True)
        return video_scores[:n_items]
