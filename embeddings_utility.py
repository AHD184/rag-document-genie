# We first pull a ollama embedding model from the terminal (ollama pull nomic-embed-text)
# What an embedding model does is convert a text to a vector (of numbers)
# We then compare the query vector with each chunk's vector and find the best match based on the query

# We implement batch embedding to speed up the convo, we use Ollama /api/embed
# We embed all the chunks at once when the PDF is uploaded.

import requests  # To make HTTP requests to Ollama (it's running as a local server)
import numpy as np  # For computing cosine similarity (fast vector math)
from typing import Union, List, Tuple  # Union lets us allow different input types

url = "http://localhost:11434/api/embed"

def ollama_embedding(
    text: Union[str, List[str]],
    model: str = "nomic-embed-text",
    timeout: int = 120
) -> Union[List[float], np.ndarray]:
    # Send Ollama's embedding model a text/texts and get embedding(s) back
    # List[str] is a batch of strings
    # If input was a string, we get a list back, which is an embedding vector
    # Otherwise we get a numpy array, with rows being the number of embedding vectors

    # If input was a single text, we convert it to a batch of size 1 (always batch-shaped)
    single_input = isinstance(text, str)
    inputs = [text] if single_input else text

    # Payload is the JSON we send to Ollama
    payload = {
        "model": model,
        "input": inputs  # list of strings
    }

    response = requests.post(url, json=payload, timeout=timeout)

    # Raise an error if something go wrong
    response.raise_for_status()

    # Convert the JSON response to a python dictionary
    data = response.json()

    # Ollama returns a list of embeddings (one per input string)
    embs = data["embeddings"]

    # If user gave a single string, return a single vector (list[float])
    if single_input:
        return embs[0]

    # If user gave a list of strings, return a matrix (n_chunks, embedding_dim)
    return np.array(embs)

# We measure the angle between two vectors [1.0 -> same direction; 0.0 -> 90 degrees apart (unrelated); -1.0 -> opposite direction]
# cos(theta) = (A.B)/||A||||B|| [dot product over magnitudes]
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    # numpy makes dot products and norms easy
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot = float(np.dot(a, b))  # Typecast from a NumPy object
    norm_a = float(np.linalg.norm(a))  # Vector length
    norm_b = float(np.linalg.norm(b))

    # Avoid division by 0 (possible scenario)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)

def top_k_cosine(query_vec: Union[List[float], np.ndarray], chunk_matrix: np.ndarray, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    # This function compares ONE query embedding to ALL chunk embeddings fast
    # It returns:
    # - the indices of the top k most similar chunks
    # - their cosine similarity scores

    # Convert query vector to numpy
    q = np.array(query_vec)

    # We add a very small value (1e-8) to norms to avoid division by 0
    q_norm = np.linalg.norm(q) + 1e-8

    # Norms of each chunk vector (one norm per chunk)
    c_norms = np.linalg.norm(chunk_matrix, axis=1) + 1e-8

    # Dot product of every chunk with the query (fast)
    dots = np.dot(chunk_matrix, q)

    # Cosine similarity for all chunks at once:
    # sims[i] = (chunk_i dot query) / (||chunk_i|| * ||query||)
    sims = dots / (c_norms * q_norm)

    # argsort sorts ascending, but we want highest similarity first
    # so we use -sims (bigger sims -> smaller negative)
    top_indices = np.argsort(-sims)[:k]
    top_scores = sims[top_indices]

    return top_indices, top_scores