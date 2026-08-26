import numpy as np
import ollama


def generate_embedding(texto: str, model: str = "nomic-embed-text", normalize: bool = True):
    response = ollama.embeddings(model=model, prompt=texto)
    emb = np.array(response["embedding"])
    if normalize:
        emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.tolist()
