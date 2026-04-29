import os
import pickle

from flask import Flask, render_template_string, request
import numpy as np

from app.engine import ImageRetrievalEngine
import kagglehub
import base64
from io import BytesIO
from PIL import Image

def image_to_data_url(img_array):
    img = Image.fromarray(img_array.astype("uint8"))

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


app = Flask(__name__)

PROMPTS = [
    "aquatic animal fish",
    "small mammal rodent",
    "vehicle transport",
    "domestic animal",
    "small entity",
]

def build_engine():
    dataset_path = kagglehub.dataset_download("fedesoriano/cifar100")
    train_file = os.path.join(dataset_path, "train")

    with open(train_file, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(data[b"fine_labels"])


    return ImageRetrievalEngine(
        images=images,
        labels=labels,
    )

model_engine = None

def get_engine():
    global model_engine
    if model_engine is None:
        model_engine = build_engine()
    return model_engine


def normalize_indices(indices):
    arr = np.array(indices).flatten()
    return [int(x) for x in arr[:9]]


def run_all_models(query):
    models = {
        "CLIP": get_engine().clip_ranking(query),
        "SVD": get_engine().svd_ranking(query),
        "TFIDF": get_engine().content_based_ranking(query),
        "Random": get_engine().random_recommendation(),
        "Popularity": get_engine().popularity_recommendation(),
    }

    result = {}

    for model_name, indices in models.items():
        fixed_indices = normalize_indices(indices)
        result[model_name] = [
            image_to_data_url(get_engine().images[i])
            for i in fixed_indices
        ]

    return result


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Retrieval Demo</title>
</head>
<body>
    <h1>Выберите запрос</h1>

    <form method="post" action="/search">
        <select name="query" required>
            {% for prompt in prompts %}
                <option value="{{ prompt }}">{{ prompt }}</option>
            {% endfor %}
        </select>

        <button type="submit">Запустить</button>
    </form>
</body>
</html>
"""


LOADING_AND_RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Results</title>
</head>
<body>
    <h1>Результаты для: {{ query }}</h1>

    {% for model_name, images in results.items() %}
        <h2>{{ model_name }}</h2>
        <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:30px;">
            {% for image in images %}
                <img src="{{ image }}" width="220" alt="result image">
            {% endfor %}
        </div>
    {% endfor %}

    <form method="get" action="/">
        <button type="submit">Начать заново</button>
    </form>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, prompts=PROMPTS)


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    results = run_all_models(query)

    return render_template_string(
        LOADING_AND_RESULT_HTML,
        query=query,
        results=results,
    )


if __name__ == "__main__":
    get_engine()
    app.run(host="0.0.0.0", port=5000)
