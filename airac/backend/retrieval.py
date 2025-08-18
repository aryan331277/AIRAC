import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import json

class RetrievePinecone:
    def __init__(self):
        load_dotenv()
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        self.JINA_API_KEY = os.getenv("JINA_API_KEY")
        self.DIMENSION = 1024

        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)

        if self.PINECONE_INDEX_NAME not in [i["name"] for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.PINECONE_INDEX_NAME,
                dimension=self.DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD"),
                    region=os.getenv("PINECONE_REGION")
                )
            )

        self.index = self.pc.Index(self.PINECONE_INDEX_NAME)

    def get_jina_embedding(self, text: str):
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.JINA_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": "jina-embeddings-v3", "input": text, "task": "retrieval.passage"}
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Jina API Error: {response.text}")
        return response.json()["data"][0]["embedding"]

    def get(self, query):
        query_embedding = self.get_jina_embedding(query)
        results = self.index.query(vector=query_embedding, top_k=1, include_metadata=True)
        return results

if __name__ == "__main__":
    obj = RetrievePinecone()
    doc = obj.get("What are the mess timings")
    
    parent_tables = doc["matches"][0].metadata["parent_tables"]

    if isinstance(parent_tables, str):
        parent_tables = json.loads(parent_tables)

    first_row = parent_tables[0]
    print(first_row["Meal"])
