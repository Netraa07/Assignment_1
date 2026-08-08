import re
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

PDF_PATH = "sample.pdf"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks
TOP_K = 3             # number of chunks to retrieve per question


# ------------------------------------------------
# 1. Document Ingestion
# ------------------------------------------------
def read_pdf(file_path):
    text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


# ------------------------------------------------
# 2. Text Chunking
# ------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    # Collapse whitespace so chunk boundaries aren't full of newlines
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ------------------------------------------------
# 3. Embedding Creation (TF-IDF vectors instead of neural embeddings)
# ------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")


def embed_chunks(chunks):
    # Fit the vectorizer on all chunks and return the TF-IDF matrix
    return vectorizer.fit_transform(chunks)


# ------------------------------------------------
# 4. Vector Database (simple in-memory store)
# ------------------------------------------------
class VectorStore:

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings  # shape: (num_chunks, embedding_dim)

    def search(self, query_embedding, top_k=TOP_K):
        scores = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


# ------------------------------------------------
# 5. Query Processing + 6. Context Retrieval
# ------------------------------------------------
def retrieve_context(question, store, top_k=TOP_K):
    query_embedding = vectorizer.transform([question])
    return store.search(query_embedding, top_k=top_k)


# ------------------------------------------------
# 7. Answer Generation
# ------------------------------------------------
print("Loading generation model...")
gen_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


def generate_answer(question, context_chunks):

    context = "\n".join(chunk for chunk, score in context_chunks)

    prompt = (
        "Answer the question using only the context below. "
        "Write the answer in 5 clear sentences.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=180,
        min_new_tokens=40,
        do_sample=False,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ------------------------------------------------
# Build the pipeline
# ------------------------------------------------
print("\nReading PDF...")
pdf_text = read_pdf(PDF_PATH)

print("Chunking text...")
chunks = chunk_text(pdf_text)
print(f"Created {len(chunks)} chunks.")

print("Embedding chunks...")
chunk_embeddings = embed_chunks(chunks)

store = VectorStore(chunks, chunk_embeddings)

print("\nRAG system ready! Ask questions about the document.")


# ------------------------------------------------
# Question Loop
# ------------------------------------------------
while True:

    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    retrieved = retrieve_context(question, store)

    print("\n--- Retrieved Chunks ---")
    for i, (chunk, score) in enumerate(retrieved, 1):
        preview = chunk[:120].replace("\n", " ")
        print(f"[{i}] (score={score:.3f}) {preview}...")

    answer = generate_answer(question, retrieved)

    print("\n==============================")
    print("Answer:")
    print(answer)
    print("==============================")