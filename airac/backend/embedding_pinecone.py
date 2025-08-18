import os
import json
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "gcp")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east1-gcp")

DIMENSION = 1024  

def get_jina_embedding(text: str):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": "jina-embeddings-v3", "input": text, "task": "retrieval.passage"}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Jina API Error: {response.text}")

    return response.json()["data"][0]["embedding"]

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    print(f"Creating new index '{PINECONE_INDEX_NAME}' with dimension {DIMENSION}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

index = pc.Index(PINECONE_INDEX_NAME)

with open("json_data/parent.json", "r", encoding="utf-8") as f:
    parents = json.load(f)

with open("json_data/child.json", "r", encoding="utf-8") as f:
    children = json.load(f)

parent_lookup = {p["parent_id"]: p for p in parents}

BATCH_SIZE = 50

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunk_list(children, BATCH_SIZE):
    vectors = []
    for child in batch:
        emb = get_jina_embedding(child["text"])
        parent = parent_lookup.get(child["parent_id"], {})

        vectors.append({
            "id": child["child_id"],
            "values": emb,
            "metadata": {
                "parent_id": child["parent_id"],
                "parent_source": parent.get("source", ""),
                "parent_title": parent.get("title", ""),
                "parent_text": parent.get("text", ""),
                "child_text": child["text"],
                "original_data": json.dumps(child.get("original_data", {})),
                "parent_tables": json.dumps(parent.get("tables", [])),
            }
        })

    print(f"Uploading batch of {len(vectors)} vectors...")
    index.upsert(vectors=vectors)

print("✅ All vectors have been uploaded to Pinecone.")

query = "What time is breakfast served in the mess?"
query_embedding = get_jina_embedding(query)

results = index.query(vector=query_embedding, top_k=2, include_metadata=True)

for match in results["matches"]:
    print(f"\n--- Child (Score: {match['score']:.4f}) ---")
    print(match["metadata"]["child_text"])
    print("\n--- Parent ---")
    print(f"Title: {match['metadata']['parent_title']}")
    print(f"Source: {match['metadata']['parent_source']}")
    print(match["metadata"]["parent_text"])
