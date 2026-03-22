import os
import shutil
import streamlit as st
from langchain.memory import ConversationBufferMemory
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_handler import LLMHandler

st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Session State Initialization ------------------
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = LLMHandler(memory=st.session_state.memory)

# ------------------ Title ------------------
st.title("🤖 Enterprise Document Intelligence System")
st.caption("Upload or link documents (PDF, DOCX, TXT, CSV, URL) and chat with them using AI")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, CSV)",
        type=['pdf', 'docx', 'txt', 'csv'],
        accept_multiple_files=True,
        help="Upload documents to analyze"
    )

    urls_input = st.text_area(
        "🌐 Add URLs (one per line)",
        placeholder="https://example.com/article1\nhttps://example.com/article2"
    )

    # ------------------ Process Documents ------------------
    if st.button("🚀 Process Documents", type="primary"):
        if uploaded_files or urls_input.strip():
            with st.spinner("Processing documents... This may take a moment."):
                try:
                    vector_manager = VectorStoreManager()

                    # Close old vector store safely
                    if st.session_state.vector_store:
                        try:
                            st.session_state.vector_store._client = None
                        except:
                            pass
                        st.session_state.vector_store = None

                    # Delete old ChromaDB folder safely
                    if os.path.exists(vector_manager.persist_directory):
                        shutil.rmtree(vector_manager.persist_directory)

                    processor = DocumentProcessor()
                    all_chunks = []
                    progress_bar = st.progress(0)

                    # Uploaded files
                    for idx, file in enumerate(uploaded_files):
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        if file.name.endswith(".pdf"):
                            chunks = processor.load_pdf(temp_path)
                        elif file.name.endswith(".docx"):
                            chunks = processor.load_docx(temp_path)
                        elif file.name.endswith(".txt"):
                            chunks = processor.load_text(temp_path)
                        elif file.name.endswith(".csv"):
                            chunks = processor.load_csv(temp_path)
                        else:
                            chunks = []
                        all_chunks.extend(chunks)
                        progress_bar.progress((idx + 1) / (len(uploaded_files) + len(urls_input.splitlines())))

                    # URLs
                    if urls_input.strip():
                        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
                        for idx, url in enumerate(urls, start=len(uploaded_files)):
                            url_chunks = processor.load_url(url)
                            all_chunks.extend(url_chunks)
                            progress_bar.progress((idx + 1) / (len(uploaded_files) + len(urls)))

                    # Create new vector store
                    vector_store = vector_manager.create_vector_store(all_chunks)
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 5})

                    st.success(f"✅ Successfully processed {len(all_chunks)} chunks from all sources!")

                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
        else:
            st.warning("⚠️ Please upload at least one document or enter URLs.")

    # ------------------ Document Stats ------------------
    if st.session_state.vector_store:
        st.divider()
        st.subheader("📊 Document Stats")
        st.info("Documents loaded and ready!")

    # ------------------ Clear All ------------------
    if st.button("🗑️ Clear All"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.llm_handler.clear_memory()

        # Close and delete vector store
        if 'vector_store' in st.session_state and st.session_state.vector_store:
            try:
                st.session_state.vector_store._client = None
            except:
                pass
            st.session_state.vector_store = None
        if 'retriever' in st.session_state:
            del st.session_state.retriever

        # Delete persisted ChromaDB folder safely
        vector_manager = VectorStoreManager()
        if os.path.exists(vector_manager.persist_directory):
            shutil.rmtree(vector_manager.persist_directory)

        st.success("Cleared conversation and document data!")
        st.rerun()

    # ------------------ LLM Settings ------------------
    with st.expander("⚙️ Settings"):
        model_choice = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="Select the language model to use"
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.7, 0.1,
            help="Higher = more creative, Lower = more focused"
        )

        reload_model = (
            'llm_model_choice' not in st.session_state or
            st.session_state.llm_model_choice != model_choice or
            st.session_state.llm_handler_temperature != temperature
        )

        if reload_model:
            with st.spinner(f"Loading {model_choice} model..."):
                st.session_state.llm_handler = LLMHandler(
                    model_name=model_choice,
                    memory=st.session_state.memory,
                    temperature=temperature
                )
                st.session_state.llm_handler_temperature = temperature
                st.session_state.llm_model_choice = model_choice

# ------------------ Main Chat Area ------------------
st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:** {source}")
                    st.divider()

# Chat input
if prompt := st.chat_input("💬 Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents..."):
                try:
                    qa_chain = st.session_state.llm_handler.create_qa_chain(st.session_state.retriever)
                    result = qa_chain.invoke({"question": prompt})

                    response = result['answer']
                    source_docs = result['source_documents']

                    st.markdown(response)

                    if source_docs:
                        seen_sources = set()
                        with st.expander("📚 View Sources"):
                            for doc in source_docs:
                                source_id = doc.metadata.get("source", "Unknown")
                                if source_id not in seen_sources:
                                    st.write(f"📄 **File:** {source_id}")
                                    st.write(f"📖 **Page:** {doc.metadata.get('page', 'N/A')}")
                                    st.write(f"**Content:** {doc.page_content[:300]}...")
                                    st.divider()
                                    seen_sources.add(source_id)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": [
                            f"📄 {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
                            for doc in source_docs
                        ]
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please check your OpenAI API key or model availability.")
    else:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload and process documents first using the sidebar!")

st.divider()
st.caption("Built with LangChain, ChromaDB, and Streamlit | Powered by OpenAI")
