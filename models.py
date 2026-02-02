# models.py
import torch
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, models as st_models
import config

def make_st_from_hf(hf_id: str, max_seq_length: int = 512, pooling: str = "mean") -> SentenceTransformer:
    # config.MAX_SEQ_LENGTH kullansak da parametre olarak geçmek esneklik sağlar
    word = st_models.Transformer(hf_id, max_seq_length=max_seq_length)
    
    if pooling == "mean":
        pool = st_models.Pooling(word.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    elif pooling == "cls":
        pool = st_models.Pooling(word.get_word_embedding_dimension(), pooling_mode_cls_token=True)
    else:
        raise ValueError("pooling must be 'mean' or 'cls'")
        
    return SentenceTransformer(modules=[word, pool], device=config.DEVICE)

@torch.no_grad()
def st_encode(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

@torch.no_grad()
def dense_score_matrix(q_embs: np.ndarray, d_embs: np.ndarray) -> np.ndarray:
    # Normalize edilmiş embeddingler için Matrix Multiplication = Cosine Similarity
    return q_embs @ d_embs.T
