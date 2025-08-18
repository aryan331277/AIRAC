import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL")


embedding_model = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

with open("parent.json", "r", encoding="utf-8") as f:
    parents = json.load(f)

with open("child.json", "r", encoding="utf-8") as f:
    children = json.load(f)

parent_lookup = {p["parent_id"]: p for p in parents}

vector_store = Chroma(
    collection_name="parent_child_chunks",
    embedding_function=embedding_model,
    persist_directory="./badal_db"
)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

BATCH_SIZE = 50

for batch in chunk_list(children, BATCH_SIZE):
    documents = []
    for child in batch:
        child_id = child["child_id"]
        parent_id = child["parent_id"]
        parent_obj = parent_lookup.get(parent_id, {})

        documents.append(
            Document(
                page_content=child["text"],
                metadata={
                    "id": child_id,
                    "parent_id": parent_id,
                    "parent_source": parent_obj.get("source", ""),
                    "parent_title": parent_obj.get("title", ""),
                    "parent_text": parent_obj.get("text", ""),
                    "child_text": child["text"],
                    "original_data": json.dumps(child.get("original_data", {})),
                    "parent_tables": json.dumps(parent_obj.get("tables", []))
                }
            )
        )

    print(f"Uploading batch of {len(documents)} documents...")
    vector_store.add_documents(documents)

query = "What time is breakfast served in the mess?"
results = vector_store.query(texts=[query], top_k=2, include_metadata=True)

for match in results["matches"]:
    print("\n--- Child ---")
    print(match["metadata"]["child_text"])
    print("\n--- Parent ---")
    print(f"Title: {match['metadata']['parent_title']}")
    print(f"Source: {match['metadata']['parent_source']}")
    print(match["metadata"]["parent_text"])
