import json
import os
import time
import pandas as pd
import numpy as np

from app.engine import ImageRetrievalEngine


def latency(func):
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    return end - start


def get_color(value, metric_type):
    if metric_type in ["precision", "relevance", "diversity"]:
        if value >= 0.8:
            return "brightgreen"
        if value >= 0.5:
            return "yellow"
        return "red"

    if metric_type == "latency":
        if value <= 0.1:
            return "brightgreen"
        if value <= 0.5:
            return "yellow"
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

    images = [None] * 100
    labels = np.random.randint(0, 5, size=100)
    clip_embeddings = np.random.rand(100, 512)

    engine = ImageRetrievalEngine(
        images=images,
        labels=labels,
        clip_embeddings=clip_embeddings
    )

    models = {
        "CLIP": lambda q: engine.clip_ranking(q),
        "SVD": lambda q: engine.svd_ranking(q),
        "TFIDF": lambda q: engine.content_based_ranking(q),
        "Random": lambda: engine.random_recommendation(),
        "Popularity": lambda: engine.popularity_recommendation()
    }

    for name, func in models.items():
        indices = (
            func(query)
            if name not in ["Random", "Popularity"]
            else func()
        )

        metrics = {
            "Precision": engine.precision_at_k(indices),

            "Relevance": (
                engine.semantic_relevance(indices, query)
                if name not in ["Random", "Popularity"]
                else 0
            ),

            "Diversity": engine.embedding_diversity(indices),

            "Latency": latency(
                lambda: (
                    func(query)
                    if name not in ["Random", "Popularity"]
                    else func()
                )
            )
        }

        for metric_name, value in metrics.items():
            save_badge_json(name, metric_name, value)