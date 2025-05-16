import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = "data/results/"
PROCESSED_DIR = "data/processed/"
FEATURES_DIR = "data/features/"
EVAL_OUTPUT = os.path.join(RESULTS_DIR, "content_based_evaluation_metrics.csv")


# Helper functions for metrics
def precision_at_k(recommended, relevant, k):
    """
    Precision@k = (# of recommended items @k that are relevant) / k
    """
    if len(recommended) == 0 or k == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len([item for item in recommended_k if item in relevant_set]) / k


def recall_at_k(recommended, relevant, k):
    """
    Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
    """
    if len(recommended) == 0 or len(relevant) == 0 or k == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len([item for item in recommended_k if item in relevant_set]) / len(relevant)


def average_precision(recommended, relevant, k):
    """
    Average Precision = average of precision@j for each relevant item j in the top k
    """
    if len(recommended) == 0 or len(relevant) == 0 or k == 0:
        return 0.0
    relevant_set = set(relevant)
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    """
    Normalized Discounted Cumulative Gain (NDCG) at k
    """
    if len(recommended) == 0 or len(relevant) == 0 or k == 0:
        return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            # Using binary relevance (1 if relevant, 0 if not)
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Compute Ideal DCG (IDCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(k_values=[5, 10], watch_threshold=0.5):
    """Evaluate content-based recommendations at different k values"""
    # Load recommendations
    recs_file = os.path.join(
        RESULTS_DIR, f"content_based_top{max(k_values)}_recommendations.csv"
    )
    recs = pd.read_csv(recs_file)
    print(f"Loaded {len(recs)} recommendations")

    # Load test data (ground truth)
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_merged.csv"))
    print(f"Loaded test data with {len(test)} interactions")

    # Only consider videos with significant watch time as relevant
    user2relevant = defaultdict(set)
    for row in test.itertuples():
        if (
            hasattr(row, "watch_ratio")
            and getattr(row, "watch_ratio") > watch_threshold
        ):
            user2relevant[getattr(row, "user_id")].add(getattr(row, "video_id"))

    relevant_counts = [len(videos) for _, videos in user2relevant.items()]
    if relevant_counts:
        avg_relevant = sum(relevant_counts) / len(relevant_counts)
        print(
            f"Average relevant videos per user (watch_ratio > {watch_threshold}): {avg_relevant:.2f}"
        )
        print(
            f"Users with at least one relevant video: {sum(1 for c in relevant_counts if c > 0)}/{len(relevant_counts)}"
        )
    else:
        print("No users have relevant videos with the current threshold")
        return

    # Build recommendations: user_id -> list of recommended video_ids (sorted by rank)
    user2recs = defaultdict(list)
    for row in recs.itertuples():
        user_id = getattr(row, "user_id")
        video_id = getattr(row, "video_id")
        rank = getattr(row, "rank")

        while len(user2recs[user_id]) < rank:
            user2recs[user_id].append(None)
        if len(user2recs[user_id]) == rank:
            user2recs[user_id].append(video_id)
        else:
            user2recs[user_id][rank - 1] = video_id

    # Remove None values that might have been added during sorting
    for user_id in user2recs:
        user2recs[user_id] = [v for v in user2recs[user_id] if v is not None]

    # Compute metrics for each k value
    results = []
    for k in k_values:
        print(f"\nEvaluating metrics at k={k}")
        metrics = []
        # Compute metrics for each user
        for user_id in user2recs:
            recommended = user2recs[user_id]
            relevant = list(user2relevant.get(user_id, []))

            prec = precision_at_k(recommended, relevant, k)
            rec = recall_at_k(recommended, relevant, k)
            ap = average_precision(recommended, relevant, k)
            ndcg = ndcg_at_k(recommended, relevant, k)

            metrics.append(
                {
                    "user_id": user_id,
                    f"precision@{k}": prec,
                    f"recall@{k}": rec,
                    f"MAP@{k}": ap,
                    f"NDCG@{k}": ndcg,
                }
            )

        # Calculate average metrics across all users
        metrics_df = pd.DataFrame(metrics)
        # Exclude user_id from mean calculation - not a metric
        avg_metrics = metrics_df.drop(columns=["user_id"]).mean()

        print(f"Average metrics at k={k}:")
        print(avg_metrics)

        # Store results for this k
        results.append(
            {
                "k": k,
                "precision": avg_metrics[f"precision@{k}"],
                "recall": avg_metrics[f"recall@{k}"],
                "MAP": avg_metrics[f"MAP@{k}"],
                "NDCG": avg_metrics[f"NDCG@{k}"],
            }
        )

        # Save detailed per-user metrics
        metrics_df.to_csv(
            os.path.join(RESULTS_DIR, f"content_based_metrics_k{k}.csv"), index=False
        )

    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv(EVAL_OUTPUT, index=False)
    print(f"\nSaved evaluation summary to {EVAL_OUTPUT}")

    # Plot metrics vs k
    plt.figure(figsize=(12, 8))
    metrics = ["precision", "recall", "MAP", "NDCG"]
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(results_df["k"], results_df[metric], marker="o")
        plt.title(f"{metric} vs k")
        plt.xlabel("k")
        plt.ylabel(metric)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "content_based_metrics_plot.png"))
    print(
        f"Saved metrics plot to {os.path.join(RESULTS_DIR, 'content_based_metrics_plot.png')}"
    )


if __name__ == "__main__":
    evaluate_model(k_values=[1, 3, 5, 10])
