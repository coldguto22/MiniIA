import os

from miniia.memory.embedder import generate_embedding
from miniia.memory.store import get_or_create_memory_collection


def project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def memory_collection():
    return get_or_create_memory_collection(base_dir=project_root())


def embed(texto: str, normalize: bool = True):
    return generate_embedding(texto=texto, normalize=normalize)
