import pickle
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
from open_clip import create_model_and_transforms
from sklearn.decomposition import PCA

class ImageSearcher:
    def __init__(self, embeddings_path, image_folder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ViT-B-32"
        self.pretrained = "openai"

        # Load model and transforms
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = self.preprocess_val

        # Tokenizer
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Load embeddings
        df = pd.read_pickle(embeddings_path)
        self.file_names = df['file_name'].tolist()
        self.embeddings = np.stack(df['embedding'].values, axis=0)

        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

        self.image_folder = image_folder

        # Precompute PCA with max 50 components
        self.max_pca_components = 50
        self.pca = PCA(n_components=self.max_pca_components)
        self.pca.fit(self.embeddings)

    def encode_text(self, text_query):
        with torch.no_grad():
            text_tokens = self.tokenizer([text_query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
        return text_features.cpu().numpy()[0]

    def encode_image(self, image_path, use_pca=False, k=50):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = F.normalize(image_features, p=2, dim=1)
        emb = image_features.cpu().numpy()[0]

        if use_pca:
            # Apply PCA transform and then take first k principal components
            emb_pca = self.pca.transform(emb.reshape(1, -1))
            emb_pca = emb_pca[:, :k]
            # Normalize the PCA-reduced embedding
            emb_pca = emb_pca / np.linalg.norm(emb_pca)
            return emb_pca[0]
        else:
            return emb

    def search(self, query_embedding, top_k=5, use_pca=False, k=50):
        # Determine if we need to apply PCA to database embeddings
        if use_pca:
            emb_to_search = self.pca.transform(self.embeddings)
            emb_to_search = emb_to_search[:, :k]
            emb_to_search = emb_to_search / np.linalg.norm(emb_to_search, axis=1, keepdims=True)
        else:
            emb_to_search = self.embeddings

        sim = emb_to_search @ query_embedding.reshape(1, -1).T
        sim = sim.ravel()
        top_indices = np.argsort(sim)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "filename": self.file_names[idx],
                "score": float(sim[idx]),
                "image_url": "/images/" + self.file_names[idx]
            })
        return results
