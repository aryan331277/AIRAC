import sys
import os
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from retrieval import RetrievePinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from cache import Cache
from langchain_groq import ChatGroq
from itertools import cycle


# -------------------------------
# 1. State for LangGraph
# -------------------------------
class ChatState(TypedDict):
    query: str
    retrieved_text: str
    retrieved_tables: list[str]
    answer: str


# -------------------------------
# 2. Key Manager for Groq
# -------------------------------
class GroqKeyManager:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        load_dotenv()
        keys = os.getenv("GROQ_KEYS", "").split(",")
        if not keys or keys == [""]:
            raise ValueError("❌ No Groq API keys found in .env file (GROQ_KEYS).")

        self.keys = cycle(keys)  # infinite round-robin
        self.model_name = model_name
        self.current_key = next(self.keys)

    def get_model(self):
        return ChatGroq(model=self.model_name, api_key=self.current_key)

    def rotate_key(self):
        self.current_key = next(self.keys)
        print(f"🔄 Switching API key... Now using: {self.current_key[:6]}***")
        return self.get_model()


# -------------------------------
# 3. RAG Pipeline (Badal)
# -------------------------------
class Badal:
    def __init__(self):
        self.key_manager = GroqKeyManager()
        self.model = self.key_manager.get_model()

        self.cache = Cache()
        self.retriever = RetrievePinecone()   # <-- Jina embeddings assumed
        self.str_parser = StrOutputParser()
        self.graph = self.graph_building()

    # -------------------------------
    # Document Retrieval
    # -------------------------------
    def retrieve_doc(self, state: ChatState):
        query = state["query"]
        
        # Try cache first
        doc = self.cache.get(query)
        
        if doc is None:
            # Retrieve using Pinecone retriever
            retriever_result = self.retriever.get(query)
            
            # Check if Pinecone returned matches
            matches = getattr(retriever_result, "matches", [])
            if not matches:
                text = ""
                tables = []
            else:
                top_match = matches[0]
                text = top_match.metadata.get("parent_text", "")
                tables = top_match.metadata.get("parent_tables", [])
                # Add to cache for next time
                self.cache.add(query=query, parent_text=text, parent_tables=tables)
        else:
            text = doc[0].metadata.get("parent_text", "")
            tables = doc[0].metadata.get("parent_tables", [])

        return {"retrieved_text": text, "retrieved_tables": tables}




    # -------------------------------
    # Answer Generation
    # -------------------------------
    def get_answer(self, state: ChatState):
        query = state['query']
        parent_text = state["retrieved_text"]
        parent_tables = state["retrieved_tables"]

        prompt = PromptTemplate(
            input_variables=["query", "parent_text", "parent_tables"],
            template="""
                You are a helpful assistant. Use the given **text** and **table rows** to answer questions. 
                Question : {query}

                (NOTE: TABLES AND ROWS data may and may not exist.)

                --- Parent Text ---
                {parent_text}

                --- Parent Tables (rows as JSON) ---
                {parent_tables}

                Based on both the text and the tables, give a clear and concise response.

                Only reply with the answer and nothing else.
            """
        )

        chain = prompt | self.model | self.str_parser

        try:
            answer = chain.invoke({
                "query": query,
                "parent_text": parent_text,
                "parent_tables": parent_tables
            })
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                # rotate key + retry
                self.model = self.key_manager.rotate_key()
                chain = prompt | self.model | self.str_parser
                answer = chain.invoke({
                    "query": query,
                    "parent_text": parent_text,
                    "parent_tables": parent_tables
                })
            else:
                raise e

        return {"answer": answer}

    # -------------------------------
    # Graph Construction
    # -------------------------------
    def graph_building(self):
        builder = StateGraph(ChatState)
        builder.add_node(self.retrieve_doc)
        builder.add_node(self.get_answer)

        builder.add_edge(START, "retrieve_doc")
        builder.add_edge("retrieve_doc", "get_answer")
        builder.add_edge("get_answer", END)

        return builder.compile()

    # -------------------------------
    # Entry Point
    # -------------------------------
    def invoke(self, query):
        init_state = {"query": query}
        return self.graph.invoke(init_state)["answer"]

