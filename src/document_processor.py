from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class DocumentProcessor:
    def __init__(self,chunk_size=1000,chunk_overlap=200):
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _clean_text(self, text: str) -> str:
        # Remove extra spaces and line breaks
        text = re.sub(r'\s+', ' ', text)
        return text.strip()    

    def load_pdf(self,file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks=self.text_splitter.split_documents(documents)
        return chunks   

    def load_docx(self,file_path):
            loader = Docx2txtLoader(file_path)
            data = loader.load()
            chunks=self.text_splitter.split_documents(data)
            return chunks    
    
    def load_text(self,file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks=self.text_splitter.split_documents(documents)
        return chunks     
    
    def load_url(self,url):
            loader = WebBaseLoader(url)
            docs = loader.load()
            chunks=self.text_splitter.split_documents(docs)
            return chunks
    
    def load_csv(self,file_path):
        loader = CSVLoader(file_path)
        docs = loader.load() 
        chunks = self.text_splitter.split_documents(docs) 
        return chunks
