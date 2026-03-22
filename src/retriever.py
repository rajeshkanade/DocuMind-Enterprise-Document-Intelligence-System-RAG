from langchain.schema import BaseRetriever

class Retriever(BaseRetriever):
    def __init__(self, vector_store, top_k=5):
        self.vector_store = vector_store
        self.top_k = top_k

    def get_relevant_documents(self, query):
        # LangChain expects this method
        return self.vector_store.similarity_search(query, k=self.top_k)
