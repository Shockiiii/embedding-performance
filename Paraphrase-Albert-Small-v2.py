import os
import time
from dotenv import load_dotenv
import faiss
import numpy as np
from chunk_document import chunk_text
import torch
from sentence_transformers import SentenceTransformer
import openai

# Initialize the SentenceTransformer model
model = SentenceTransformer('Paraphrase-Albert-Small-v2')

# Load environment variables
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ============== Document Loading ==============
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# ============== Embedding Generation ==============
def get_embedding(text, model=model):
    """
    Generate embeddings for the text using SentenceTransformer.
    """
    return model.encode(text, convert_to_numpy=True)

# ============== Vector Store Construction ==============
def build_vector_store(chunks):
    """
    Generate embeddings for each text chunk and build a FAISS index.
    """
    print("Generating embeddings for chunks...")
    embeddings = model.encode(chunks, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# ============== Similarity Search ==============
def search(query, index, embeddings, chunks, k=3):
    """
    Retrieve the top k most relevant text chunks for the given query.
    """
    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# ============== LLM Answer Generation ==============
def generate_answer(query, context_chunks):
    """
    Use the retrieved chunks as context to generate a detailed answer via the LLM.
    """
    context_text = "\n\n".join(context_chunks)
    prompt = f"Below is some reference content:\n\n{context_text}\n\nBased on the reference above, please answer the following question:\n\n{query}\n\nPlease provide a simple answer, Say you can't find answer if you you can't find anything in the kowledgebase and dont answer."

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    answer = response.choices[0].message.content
    return answer

# ============== Main ==============
if __name__ == "__main__":
    # 1. Load and chunk the document
    raw_text = load_document(DATA_FILE_PATH)
    chunks = chunk_text(raw_text)

    # 2. Build the vector store
    print("Generating embeddings and building index...")
    start_time = time.time()
    index, embeddings = build_vector_store(chunks)
    end_time = time.time()
    print(f"Index built in {end_time - start_time:.2f} seconds.")

    # 3. Interactive query loop
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.strip().lower() == 'exit':
            break

        # 4. Retrieve relevant text chunks
        retrieve_start = time.time()
        relevant_chunks = search(query, index, embeddings, chunks, k=3)
        retrieve_time = time.time() - retrieve_start

        # 5. Generate the answer
        answer_start = time.time()
        answer = generate_answer(query, relevant_chunks)
        answer_time = time.time() - answer_start

        # 6. Print the answer and performance info
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== PERFORMANCE INFO ===")
        print(f"Retrieval time: {retrieve_time:.2f} seconds")
        print(f"Answer generation time: {answer_time:.2f} seconds\n")
