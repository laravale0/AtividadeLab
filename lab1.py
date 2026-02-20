import numpy as np


def normalized_exponential_distribution(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def query_key_value_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = normalized_exponential_distribution(scores)
    output = attention_weights @ V
    return output, attention_weights
