from datasets import load_dataset
import numpy as np
from usearch.index import Index
from sentence_transformers.util import quantize_embeddings

dataset = load_dataset("mixedbread-ai/wikipedia-2023-11-embed-en-pre-1", split="train")
embeddings = np.array(dataset["emb"], dtype=np.float32)

int8_embeddings = quantize_embeddings(embeddings, "int8")
index = Index(ndim=1024, metric="ip", dtype="i8")
index.add(np.arange(len(int8_embeddings)), int8_embeddings)
index.save("wikipedia_int8_usearch_1m.index")
