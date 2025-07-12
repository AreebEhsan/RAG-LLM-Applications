import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# ---- LangChain imports (v0.2+ paths) ----
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
# -----------------------------------------

from HTMLtemplates import css, bot_template, user_template

# ---------- Helper functions ----------
def get_pdf_text(pdf_files) -> str:
    """Extract raw text from a list of uploaded PDFs."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(raw_text):
    """Split raw text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    """Embed chunks and store them in a FAISS index."""
    embeddings = OpenAIEmbeddings()  # or HuggingFaceInstructEmbeddings(...)
    return FAISS.from_texts(text_chunks, embeddings)

def get_conversation_chain(vector_store):
    """Create a conversational retrieval chain with chat history memory."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)  # change as desired
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
# --------------------------------------

def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, msg in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()  # loads OPENAI_API_KEY, etc.

    st.set_page_config(page_title="Multiple PDFs Chatbot", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)
    st.header("📄 Chat with multiple PDFs")

    # ---------- Chat box ----------
    user_question = st.text_input("Ask a question about your PDFs")
    if user_question and st.session_state.get("conversation"):
        handle_userinput(user_question)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.subheader("Upload your documents")
        pdf_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process") and pdf_files:
            with st.spinner("🔄 Reading & indexing PDFs …"):
                raw_text = get_pdf_text(pdf_files)
                chunks = get_text_chunks(raw_text)
                vector_store = get_vectorstore(chunks)

                # Persist in session state
                st.session_state.vector_store = vector_store
                st.session_state.conversation = get_conversation_chain(vector_store)

            st.success("✅ PDFs processed! Ask away.")

    # ---------- Session defaults ----------
    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])

if __name__ == "__main__":
    main()
