import json
import chromadb
from sentence_transformers import SentenceTransformer
import re

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "tds_virtual_ta"
CHUNK_SIZE = 3  # Number of sentences per chunk

def chunk_sentences(text, chunk_size=3):
    # Split text into sentences (simple regex, can be improved)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size]).strip()
        if chunk:
            yield chunk

def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if any(collection.name == COLLECTION_NAME for collection in client.list_collections()):
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    with open('data/knowledge_baseV3.json', 'r') as f:
        knowledge_base = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(knowledge_base):
        content = item.get('content', '').strip()
        if not content:
            continue  # Skip entries without meaningful content
        # Chunk long content into smaller pieces
        for chunk in chunk_sentences(content, CHUNK_SIZE):
            documents.append(chunk)
            metadatas.append({'source': item.get('source', 'unknown')})
            ids.append(str(len(ids) + 1))  # Ensure unique string IDs

    print(f"Total valid (chunked) documents to add: {len(documents)}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents).tolist()

    # collection.add(
    #     documents=documents,
    #     metadatas=metadatas,
    #     ids=ids,
    #     embeddings=embeddings
    # )

    BATCH_SIZE = 5461

    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
        documents=documents[i:i+BATCH_SIZE],
        metadatas=metadatas[i:i+BATCH_SIZE],
        ids=ids[i:i+BATCH_SIZE],
        embeddings=embeddings[i:i+BATCH_SIZE])

    print("Data preparation complete.")

if __name__ == "__main__":
    main()