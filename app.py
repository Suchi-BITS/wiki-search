import time
import gradio as gr
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
import faiss
import numpy as np

# Load titles, texts, and int8 embeddings in a lazy Dataset, allowing us to efficiently access specific rows on demand
# Note that we never actually use the int8 embeddings for search directly, they are only used for rescoring after the binary search
title_text_int8_dataset = load_dataset("sentence-transformers/quantized-retrieval-data", split="train").select_columns(["url", "title", "text", "embedding"])
# title_text_int8_dataset = load_from_disk("wikipedia-mxbai-embed-int8-index").select_columns(["url", "title", "text", "embedding"])

# Load the binary indices
binary_index_path = hf_hub_download(repo_id="sentence-transformers/quantized-retrieval-data", filename="wikipedia_ubinary_faiss_50m.index", local_dir=".", repo_type="dataset")
binary_ivf_index_path = hf_hub_download(repo_id="sentence-transformers/quantized-retrieval-data", filename="wikipedia_ubinary_ivf_faiss_50m.index", local_dir=".", repo_type="dataset")

binary_index: faiss.IndexBinaryFlat = faiss.read_index_binary(binary_index_path)
binary_ivf_index: faiss.IndexBinaryIVF = faiss.read_index_binary(binary_ivf_index_path)

# Load the SentenceTransformer model for embedding the queries
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
if model.device.type == "cuda":
    model.bfloat16()

warmup_queries = [
    "What is the capital of France?",
    "Who is the president of the United States?",
    "What is the largest mammal?",
    "How to bake a chocolate cake?",
    "What is the theory of relativity?",
]
model.encode(warmup_queries)


def search(
    query,
    top_k: int = 20,
    rescore_multiplier: int = 4,
    use_approx: bool = False,
    display_score: bool = True,
    display_binary_rank: bool = False,
):
    # 1. Embed the query as float32
    start_time = time.time()
    query_embedding = model.encode_query(query)
    embed_time = time.time() - start_time

    # 2. Quantize the query to ubinary
    start_time = time.time()
    query_embedding_ubinary = quantize_embeddings(
        query_embedding.reshape(1, -1), "ubinary"
    )
    quantize_time = time.time() - start_time

    # 3. Search the binary index (either exact or approximate)
    index = binary_ivf_index if use_approx else binary_index
    start_time = time.time()
    _scores, binary_ids = index.search(
        query_embedding_ubinary, top_k * rescore_multiplier
    )
    binary_ids = binary_ids[0]
    search_time = time.time() - start_time

    # 4. Load the corresponding int8 embeddings
    start_time = time.time()
    int8_embeddings = np.array(
        title_text_int8_dataset[binary_ids]["embedding"], dtype=np.int8
    )
    load_int8_time = time.time() - start_time

    # 5. Rescore the top_k * rescore_multiplier using the float32 query embedding and the int8 document embeddings
    start_time = time.time()
    scores = query_embedding @ int8_embeddings.T
    rescore_time = time.time() - start_time

    # 6. Sort the scores and return the top_k
    start_time = time.time()
    indices = scores.argsort()[::-1][:top_k]
    top_k_indices = binary_ids[indices]
    top_k_scores = scores[indices]
    sort_time = time.time() - start_time

    # 7. Load titles and texts for the top_k results
    start_time = time.time()
    top_k_titles = title_text_int8_dataset[top_k_indices]["title"]
    top_k_urls = title_text_int8_dataset[top_k_indices]["url"]
    top_k_texts = title_text_int8_dataset[top_k_indices]["text"]
    top_k_titles = [f"[{title}]({url})" for title, url in zip(top_k_titles, top_k_urls)]
    load_text_time = time.time() - start_time

    rank = np.arange(1, top_k + 1)
    data = {
        "Score": [f"{score:.2f}" for score in top_k_scores],
        "#": rank,
        "Binary #": indices + 1,
        "Title": top_k_titles,
        "Text": top_k_texts,
    }
    if not display_score:
        del data["Score"]
    if not display_binary_rank:
        del data["Binary #"]
        del data["#"]
    df = pd.DataFrame(data)

    return df, {
        "Embed Time": f"{embed_time:.4f} s",
        "Quantize Time": f"{quantize_time:.4f} s",
        "Search Time": f"{search_time:.4f} s",
        "Load int8 Time": f"{load_int8_time:.4f} s",
        "Rescore Time": f"{rescore_time:.4f} s",
        "Sort Time": f"{sort_time:.4f} s",
        "Load Text Time": f"{load_text_time:.4f} s",
        "Total Retrieval Time": f"{quantize_time + search_time + load_int8_time + rescore_time + sort_time + load_text_time:.4f} s",
    }


with gr.Blocks(title="Quantized Retrieval") as demo:
    gr.Markdown(
        """
## Quantized Retrieval - Binary Search with Scalar (int8) Rescoring
This demo showcases retrieval using [quantized embeddings](https://huggingface.co/blog/embedding-quantization) on a CPU. The corpus consists of [41 million texts](https://huggingface.co/datasets/sentence-transformers/quantized-retrieval-data) from Wikipedia articles.

<details><summary>Click to learn about the retrieval process</summary>

Details:
1. The query is embedded using the [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) SentenceTransformer model.
2. The query is quantized to binary using the `quantize_embeddings` function from the SentenceTransformers library.
3. A binary index (41M binary embeddings; 5.2GB of memory/disk space) is searched using the quantized query for the top 80 documents.
4. The top 80 documents are loaded on the fly from an int8 index on disk (41M int8 embeddings; 0 bytes of memory, 47.5GB of disk space).
5. The top 80 documents are rescored using the float32 query and the int8 embeddings to get the top 20 documents.
6. The top 20 documents are sorted by score.
7. The titles and texts of the top 20 documents are loaded on the fly from disk and displayed.

This process is designed to be memory efficient and fast, with the binary index being small enough to fit in memory and the int8 index being loaded as a view to save memory. 
In total, this process requires keeping 1) the model in memory, 2) the binary index in memory, and 3) the int8 index on disk. With a dimensionality of 1024, 
we need `1024 / 8 * num_docs` bytes for the binary index and `1024 * num_docs` bytes for the int8 index.

This is notably cheaper than doing the same process with float32 embeddings, which would require `4 * 1024 * num_docs` bytes of memory/disk space for the float32 index, i.e. 32x as much memory and 4x as much disk space.
Additionally, the binary index is much faster (up to 32x) to search than the float32 index, while the rescoring is also extremely efficient. In conclusion, this process allows for fast, scalable, cheap, and memory-efficient retrieval.

Feel free to check out the [code for this demo](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval/blob/main/app.py) to learn more about how to apply this in practice.

Notes:
- The approximate search index (a binary Inverted File Index (IVF)) is in beta and has not been trained with a lot of data.

</details>
"""
    )
    with gr.Row():
        with gr.Column(scale=60):
            query = gr.Textbox(
                label="Query for Wikipedia articles",
                placeholder="Enter a query to search for relevant texts from Wikipedia.",
            )
        with gr.Column(scale=25):
            use_approx = gr.Radio(
                choices=[("Exact Search", False), ("Approximate Search", True)],
                value=True,
                label="Search Settings",
            )
        with gr.Column(scale=15):
            display_score = gr.Checkbox(
                label="Display Score",
                value=True,
            )
            display_binary_rank = gr.Checkbox(
                label='Display Binary Rank',
                value=False,
            )

    with gr.Row():
        with gr.Column(scale=2):
            top_k = gr.Slider(
                minimum=10,
                maximum=1000,
                step=1,
                value=20,
                label="Number of documents to retrieve",
                info="Number of documents to retrieve from the binary search",
            )
        with gr.Column(scale=2):
            rescore_multiplier = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=4,
                label="Rescore multiplier",
                info="Search for `rescore_multiplier` as many documents to rescore",
            )

    search_button = gr.Button(value="Search")

    with gr.Row():
        with gr.Column(scale=4):
            output = gr.Dataframe(
                headers=["Score", "#", "Binary #", "Title", "Text"],
                datatype="markdown",
            )
        with gr.Column(scale=1):
            json = gr.JSON()

    examples = gr.Examples(
        examples=[
            "What is the coldest metal to the touch?",
            "Who won the FIFA World Cup in 2018?",
            "How to make a paper airplane?",
            "Who was the first woman to cross the Pacific ocean by plane?",
        ],
        fn=search,
        inputs=[query],
        outputs=[output, json],
        cache_examples=False,
        run_on_click=True,
    )

    query.submit(
        search,
        inputs=[query, top_k, rescore_multiplier, use_approx, display_score, display_binary_rank],
        outputs=[output, json],
    )
    search_button.click(
        search,
        inputs=[query, top_k, rescore_multiplier, use_approx, display_score, display_binary_rank],
        outputs=[output, json],
    )
    display_score.change(
        search,
        inputs=[query, top_k, rescore_multiplier, use_approx, display_score, display_binary_rank],
        outputs=[output, json],
    )
    display_binary_rank.change(
        search,
        inputs=[query, top_k, rescore_multiplier, use_approx, display_score, display_binary_rank],
        outputs=[output, json],
    )

demo.queue()
demo.launch()
