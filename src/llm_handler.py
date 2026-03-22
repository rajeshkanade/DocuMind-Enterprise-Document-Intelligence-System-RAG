from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

class LLMHandler:
    def __init__(self, model_name="gpt-4o-mini", memory=None, temperature=0.7,max_tokens=500):
        load_dotenv()

        self.memory = memory if memory else ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        if model_name in openai_models:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature,max_tokens=max_tokens)
        else:
            raise ValueError(f"Model {model_name} not supported. Choose from {openai_models}")

    def create_qa_chain(self, retriever):
        template = """You are a knowledgeable AI assistant helping users understand their documents.

Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know - don't make up an answer.
Keep your answers conversational and reference the chat history when relevant.

Context from documents:
{context}

Previous conversation:
{chat_history}

Current question: {question}

Helpful answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )
        return qa_chain

    def clear_memory(self):
        self.memory.clear()

    def get_conversation_history(self):
        return self.memory.load_memory_variables({})
