from dotenv import load_dotenv
import os
import requests
import json
from langchain_core.documents import Document
from pinecone import Pinecone as PineconeClient, ServerlessSpec

class Cache:
    def __init__(self):
        load_dotenv()
        self.JINA_API_KEY = os.getenv("JINA_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "semantic-cache-jina-api"
        self.dimension = 1024

        self.pinecone = PineconeClient(api_key=pinecone_api_key)

        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Creating a new cache index named '{self.index_name}' with {self.dimension} dimensions...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pinecone.Index(self.index_name)

    def _get_jina_embedding(self, text: str) -> list[float]:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.JINA_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": "jina-embeddings-v3", "input": text, "task": "retrieval.passage"}

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Jina API Error in Cache: {response.text}")

        return response.json()["data"][0]["embedding"]

    def get(self, query: str):
        try:
            query_embedding = self._get_jina_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=1,
                include_metadata=True
            )
            matches = results.get("matches", [])
            if not matches:
                return None

            top_match = matches[0]
            if top_match['score'] >= 0.95:
                metadata = {
                    "parent_text": top_match["metadata"].get("parent_text", ""),
                    "parent_tables": json.loads(top_match["metadata"].get("parent_tables", "[]"))
                }
                return [Document(page_content=query, metadata=metadata)]

            return None
        except Exception as e:
            print(f"An error occurred during cache retrieval: {e}")
            return None

    def add(self, query: str, parent_text: str, parent_tables: list):
        try:
            query_embedding = self._get_jina_embedding(query)
            metadata = {
                "parent_text": parent_text,
                "parent_tables": json.dumps(parent_tables)
            }
            self.index.upsert(vectors=[{
                "id": query,
                "values": query_embedding,
                "metadata": metadata
            }])
        except Exception as e:
            print(f"An error occurred during cache add: {e}")
