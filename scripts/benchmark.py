import json
import os
import pandas as pd
from app.engine import (
    clip_ranking, svd_ranking, content_based_recommendation,
    random_recommendation, popularity_recommendation,
    precision_at_k, semantic_relevance_at_k, embedding_diversity, latency
)


def get_color(value, metric_type):
    if metric_type in ["precision", "relevance", "diversity"]:
        if value >= 0.8: return "brightgreen"
        if value >= 0.5: return "yellow"
        return "red"
    if metric_type == "latency":
        if value <= 0.1: return "brightgreen"
        if value <= 0.5: return "yellow"
        return "red"
    return "blue"


def save_badge_json(model_name, metric_name, value):
    directory = "badges"
    if not os.path.exists(directory):
        os.makedirs(directory)

    data = {
        "schemaVersion": 1,
        "label": f"{model_name} {metric_name}",
        "message": f"{value:.3f}",
        "color": get_color(value, metric_name.lower())
    }

    filename = f"{directory}/{model_name.lower()}_{metric_name.lower()}.json"
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    query = "aquatic animal fish"
    models = {
        "CLIP": lambda q: clip_ranking(q),
        "SVD": lambda q: svd_ranking(q),
        "TFIDF": lambda q: content_based_recommendation(q),
        "Random": lambda q: random_recommendation(),
        "Popularity": lambda q: popularity_recommendation()
    }

    for name, func in models.items():
        indices = func(query) if name not in ["Random", "Popularity"] else func()

        metrics = {
            "Precision": precision_at_k(indices),
            "Relevance": semantic_relevance_at_k(indices, query) if name not in ["Random", "Popularity"] else 0,
            "Diversity": embedding_diversity(indices),
            "Latency": latency(lambda: func(query) if name not in ["Random", "Popularity"] else func())
        }

        for m_name, val in metrics.items():
            save_badge_json(name, m_name, val)