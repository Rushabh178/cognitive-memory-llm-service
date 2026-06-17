# ============================================================
# FILE: embeddings.py
#
# WHAT THIS FILE DOES:
# Loads the sentence-transformer model once at startup and exposes a single
# encode() function that converts any text string into a list of floats
# (an embedding vector). Every other module that needs embeddings imports
# from here — the model is never loaded more than once.
#
# KEY CONCEPT FOR LEARNING:
# An embedding is a way of representing meaning as a point in space.
# Imagine a city map where similar neighbourhoods are close together and
# different ones are far apart. "I love pizza" and "Pizza is my favourite food"
# would land very close together on this map, while "The stock market crashed"
# would land far away. That's what an embedding model does — it reads text and
# returns coordinates on that meaning-map as a list of numbers (a vector).
# Because similar text produces similar vectors, we can find related memories
# by finding vectors that are close together — that's semantic search.
#
# WHERE IT FITS IN THE PIPELINE:
# Raw text (string) → embeddings.py → vector (List[float]) → ChromaDB storage / search
# ============================================================

from sentence_transformers import SentenceTransformer
from typing import List
import config

# ---------------------------------------------------------------
# Load the model ONCE when this module is first imported.
#
# Why at module level (not inside the function)?
# Loading a model downloads weights from disk and allocates memory.
# If we did this inside encode(), every single API request would
# reload the model — adding ~1 second of latency per call.
# By loading once here, every subsequent encode() call is fast.
#
# The model "all-MiniLM-L6-v2":
#   - Size: ~80 MB — small enough to run on a laptop CPU
#   - Output: 384-dimensional vector (384 floats per text input)
#   - Quality: excellent for semantic similarity tasks
#   - Speed: encodes a sentence in <10 ms on CPU
# ---------------------------------------------------------------
_model = SentenceTransformer(config.EMBEDDING_MODEL)


def encode(text: str) -> List[float]:
    """
    Convert a text string into a fixed-size vector of floats.

    The vector captures the semantic meaning of the text. Two sentences that
    mean similar things will produce vectors that are close together in
    384-dimensional space, even if they share no words in common.

    Args:
        text: Any string — a user message, a stored memory, a search query.

    Returns:
        A list of 384 floats representing the semantic meaning of the text.
    """

    # Call the model's encode method.
    # - convert_to_python=True is implicit when we call .tolist() below.
    # - The model returns a numpy array; .tolist() converts it to a plain
    #   Python list of floats, which is what ChromaDB and JSON serialisation expect.
    embedding = _model.encode(text)

    # Convert numpy array → plain Python list of floats.
    # ChromaDB expects List[float], not numpy arrays.
    return embedding.tolist()
