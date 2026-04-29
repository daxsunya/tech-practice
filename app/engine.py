import os
import pickle


import torch
import numpy as np
from collections import Counter
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from PIL import Image

device = "mps" if torch.mps.is_available() else "cpu"


class ImageRetrievalEngine:
    def __init__(self, images, labels):
        self.device = "mps" if torch.mps.is_available() else "cpu"
        model_path = "/models/clip-vit-base-patch32"

        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

        self.images = images
        self.labels = labels

        print("Вычисление эмбеддингов изображений (offline этап)")
        self.clip_image_embeddings = self.compute_clip_embeddings(images)

        self.svd = TruncatedSVD(n_components=64, random_state=42)
        self.svd_embeddings = self.svd.fit_transform(self.clip_image_embeddings)

        class_descriptions = {0: "aquatic animal fish", 1: "small mammal rodent", 2: "vehicle transport",
                              3: "domestic animal", 4: "wild animal"}
        texts = [class_descriptions.get(c, "object image") for c in labels]
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(texts)


    def compute_clip_embeddings(self, images, batch_size=64):
        embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="CLIP embeddings"):
            batch = images[i:i + batch_size]

            inputs = self.processor(
                images=[Image.fromarray(img) for img in batch],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)

            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs).cpu().numpy()

    def clip_ranking(self, query_text, k=9):
        user_emb = self.get_text_embedding(query_text)
        sims = cosine_similarity(user_emb, self.clip_image_embeddings).ravel()
        return np.argsort(sims)[-k:]

    def svd_ranking(self, query_text, k=9):
        user_emb = self.get_text_embedding(query_text)
        user_latent = self.svd.transform(user_emb)
        sims = cosine_similarity(user_latent, self.svd_embeddings).ravel()
        return np.argsort(sims)[-k:]

    def content_based_ranking(self, query_text, k=9):
        query_vec = self.tfidf.transform([query_text])
        sims = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
        return np.argsort(sims)[-k:]

    def random_recommendation(self, k=9):
        return np.random.choice(len(self.labels), k, replace=False)

    def popularity_recommendation(self, k=9):
        popular_classes = [
            c for c, _ in Counter(self.labels).most_common()
        ]

        indices = []

        for cls in popular_classes:
            cls_indices = np.where(self.labels == cls)[0]
            indices.extend(cls_indices.tolist())

            if len(indices) >= k:
                break

        return np.array(indices[:k])

    def precision_at_k(self, indices):
        if len(indices) == 0:
            return 0

        classes = self.labels[indices]
        dominant_class, count = Counter(classes).most_common(1)[0]
        return count / len(indices)

    def embedding_diversity(self, indices):
        embs = self.clip_image_embeddings[indices]
        sims = cosine_similarity(embs)
        upper = sims[np.triu_indices_from(sims, k=1)]
        return 1 - np.mean(upper)

    def semantic_relevance(self, indices, query_text):
        query_emb = self.get_text_embedding(query_text)
        imgs_emb = self.clip_image_embeddings[indices]
        sims = cosine_similarity(query_emb, imgs_emb).ravel()
        return float(np.mean(sims))

    def ndcg_at_k(self, indices, query_text):
        query_emb = self.get_text_embedding(query_text)
        actual_sims = cosine_similarity(query_emb, self.clip_image_embeddings[indices]).ravel()
        ideal_sims = np.sort(actual_sims)[::-1]

        def dcg(s): return np.sum([val / np.log2(i + 2) for i, val in enumerate(s)])

        actual_dcg = dcg(actual_sims)
        idcg = dcg(ideal_sims)
        return actual_dcg / idcg if idcg > 0 else 0