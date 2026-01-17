# Quantized Retrieval with Binary Search + int8 Rescoring

This repository demonstrates a **large-scale, memory-efficient semantic search system** using **quantized embeddings**.  
It combines **binary (ubinary) search for fast candidate retrieval** with **int8 rescoring** for accuracy, enabling search over **tens of millions of documents on CPU**.

The demo uses Wikipedia embeddings and provides a **Gradio UI** for interactive search.

---

## Key Idea

Instead of storing and searching float32 embeddings (very expensive at scale), this system:

1. Uses **binary embeddings (ubinary)** for fast, low-memory approximate search
2. Loads **int8 embeddings on demand** for rescoring
3. Keeps **accuracy close to float32** at a **fraction of the cost**

This approach enables **fast, scalable, and cheap retrieval** on commodity hardware.

---

## Intended Usage

This demo is intended for:
- Research on embedding quantization
- Large-scale semantic retrieval
- CPU-only RAG pipelines
- Cost-sensitive search systems
- Benchmarking retrieval latency at scale

## Architecture Diagram

```mermaid
flowchart TD
    U[User Query]
    E[SentenceTransformer<br/>(float32 embedding)]
    Q[Binary Quantization<br/>(ubinary)]
    B[Binary FAISS Index<br/>(RAM)]
    I[Candidate IDs<br/>(Top K × Multiplier)]
    L[Lazy Load int8 Embeddings<br/>(Disk)]
    R[Float32 × int8 Rescoring]
    S[Final Top-K Sorting]
    O[Titles & Text Display<br/>(Gradio UI)]

    U --> E
    E --> Q
    Q --> B
    B --> I
    I --> L
    L --> R
    R --> S
    S --> O
```
## Retrieval Pipeline

1. **Query embedding**
   - Query is embedded using `mixedbread-ai/mxbai-embed-large-v1`
   - Output is a float32 vector (1024 dims)

2. **Binary quantization**
   - Query embedding is quantized to **ubinary**
   - Used for fast search in a binary FAISS index

3. **Binary search**
   - Search over millions of documents using:
     - `IndexBinaryFlat` (exact) or
     - `IndexBinaryIVF` (approximate)
   - Retrieves top `K × rescore_multiplier` candidates

4. **Lazy int8 loading**
   - Corresponding int8 document embeddings are loaded **only for candidates**

5. **Rescoring**
   - Float32 query × int8 document embeddings
   - Produces accurate similarity scores

6. **Final ranking**
   - Top-K results returned with title + text

---

## Repository Structure

```text
.
├── app.py                 # Gradio demo application
├── save_binary_index.py   # Script to build ubinary FAISS index
├── save_int8_index.py     # Script to build int8 USearch index
└── README.md
## Quantized Retrieval - Binary Search with Scalar (int8) Rescoring

This demo showcases retrieval using quantized embeddings on a CPU.  
The corpus consists of 41 million texts from Wikipedia articles.

<details>
<summary>Click to learn about the retrieval process</summary>

### Retrieval Process

1. The query is embedded using the `mixedbread-ai/mxbai-embed-large-v1`
   SentenceTransformer model.

2. The query is quantized to binary using the `quantize_embeddings`
   function from the SentenceTransformers library.

3. A binary index (41M binary embeddings; ~5.2GB of memory/disk space)
   is searched using the quantized query for the top 80 documents.

4. The top 80 documents are loaded on the fly from an int8 index on disk
   (41M int8 embeddings; 0 bytes of memory, ~47.5GB of disk space).

5. The top 80 documents are rescored using the float32 query embedding
   and the int8 document embeddings to get the top 20 documents.

6. The top 20 documents are sorted by score.

7. The titles and texts of the top 20 documents are loaded on the fly
   from disk and displayed.

---

### Design Goals

This process is designed to be:

- Memory efficient
- Fast on CPU
- Scalable to tens of millions of documents

At runtime, the system keeps only:

1. The embedding model in memory
2. The binary FAISS index in memory
3. The int8 embeddings on disk (loaded lazily)

---

### Memory Cost Breakdown (1024 Dimensions)

- Binary index size:  
  `1024 / 8 * num_docs` bytes

- int8 index size:  
  `1024 * num_docs` bytes

This is significantly cheaper than float32 embeddings:

- float32 index size:  
  `4 * 1024 * num_docs` bytes

That means:
- **32× less memory than float32**
- **4× less disk space than float32**
- **Up to 32× faster search**

---

### Performance Summary

- Binary search is extremely fast
- Rescoring with int8 embeddings is efficient
- Accuracy remains close to float32 retrieval
- Suitable for large-scale, low-cost retrieval systems

---

### Notes

- Approximate search uses a binary IVF index
- The IVF index is in beta and not heavily trained
- Exact binary search is fully supported

</details>
