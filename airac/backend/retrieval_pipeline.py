import sys
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from retrieval import RetrievePinecone  # <-- Jina-based retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from cache import Cache

class ChatState(TypedDict):
    query: str
    retrieved_text: str
    retrieved_tables: list[str]
    answer: str

class Badal:
    def __init__(self):
        load_dotenv()
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.cache = Cache()
        self.retriever = RetrievePinecone()   # <-- Jina embeddings
        self.str_parser = StrOutputParser()
        self.graph = self.graph_building()

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
        answer = chain.invoke({"query": query, "parent_text": parent_text, "parent_tables": parent_tables})
        return {"answer": answer}

    def graph_building(self):
        builder = StateGraph(ChatState)
        builder.add_node(self.retrieve_doc)
        builder.add_node(self.get_answer)

        builder.add_edge(START, "retrieve_doc")
        builder.add_edge("retrieve_doc", "get_answer")
        builder.add_edge("get_answer", END)

        graph = builder.compile()
        return graph

    def invoke(self, query):
        init_state = {"query": query}
        answer = self.graph.invoke(init_state)["answer"]
        return answer


# --- MAIN EXECUTION BLOCK ---
# This code will run when you execute `python retrieval_pipeline.py`
if __name__ == "__main__":
    # Check if a query was provided from the command line
    if len(sys.argv) > 1:
        # Join all arguments after the script name to form the query
        user_query = " ".join(sys.argv[1:])
    else:
        # If no query is given, use a default one or ask the user
        user_query = "What is the capital of France?" # Example query

    print(f"Processing query: '{user_query}'")
    
    # 1. Create an instance of your RAG pipeline
    badal_pipeline = Badal()
    
    # 2. Invoke the pipeline with the user's query
    final_answer = badal_pipeline.invoke(user_query)
    
    # 3. Print the final answer
    print("\n--- Final Answer ---")
    print(final_answer)
