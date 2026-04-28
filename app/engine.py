import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

class ImageRetrievalEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs).cpu().numpy()

    def clip_ranking(self, query_text, image_embeddings, k=9):
        user_emb = self.get_text_embedding(query_text)
        sims = cosine_similarity(user_emb, image_embeddings).ravel()
        return np.argsort(sims)[-k:]