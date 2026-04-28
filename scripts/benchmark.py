import json
import os
import pickle
import time

import kagglehub
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

    dataset_path = kagglehub.dataset_download("fedesoriano/cifar100")
    train_file = os.path.join(dataset_path, "train")

    with open(train_file, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(data[b"fine_labels"])

    engine = ImageRetrievalEngine(
        images=images,
        labels=labels,
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