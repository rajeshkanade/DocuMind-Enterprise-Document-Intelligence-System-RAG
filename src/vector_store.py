import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.persist_directory = persist_directory
        self.vector_store = None

    def create_vector_store(self, documents):
        # 🧹 Step 1: Clear existing Chroma DB before creating new one
        shutil.rmtree(self.persist_directory, ignore_errors=True)

        # 🆕 Step 2: Create new vector store with fresh embeddings
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Optional: persist it (saves the new DB)
        self.vector_store.persist()

        return self.vector_store

    def load_vector_store(self):
        # Load existing Chroma DB (used when you don't want to delete)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store

    def close(self):
        if self.vector_store:
            try:
                self.vector_store._client = None  # release the client
            except Exception:
                pass
            self.vector_store = None
