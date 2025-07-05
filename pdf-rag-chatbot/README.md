# 🤖 PDF RAG Chatbot

**`pdf-rag-chatbot/`** is an intelligent, interactive application that allows users to upload PDF documents and ask natural language questions about their content. By combining modern NLP techniques like **embeddings**, **semantic search**, and **Retrieval-Augmented Generation (RAG)**, the app delivers context-aware answers directly from the PDF.



## 📚 Problem Statement

Large Language Models (LLMs) like GPT-4 have **context window limits**, which means they cannot handle entire books or large PDFs in a single query. Simply dumping an entire document into a prompt is inefficient and costly.

This project solves that by:
- Splitting PDFs into digestible chunks
- Creating vector embeddings of those chunks
- Searching semantically relevant chunks using a vector store
- Passing only the most relevant information to the LLM to generate accurate answers

---

## 🛠️ Tech Stack

| Layer                  | Technology                              |
|-----------------------|------------------------------------------|
| Frontend              | [Streamlit](https://streamlit.io)       |
| PDF Parsing           | [PyPDF2](https://pypi.org/project/PyPDF2/) |
| Chunking              | LangChain `CharacterTextSplitter`       |
| Embeddings            | OpenAI Embeddings (`text-embedding-ada-002`) |
| Vector Store          | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM Orchestration     | [LangChain](https://www.langchain.com)  |
| Secrets Management    | `python-dotenv` + `.env` file           |

---

## 🧠 Core Architecture

```mermaid
graph TD
    A[User Uploads PDF] --> B[PyPDF2 Extracts Text]
    B --> C[LangChain Splits Text into Chunks]
    C --> D[OpenAI Embeddings Vectorize Chunks]
    D --> E[FAISS Stores Embeddings]
    F[User Enters Question] --> G[OpenAI Embeds Question]
    G --> H[FAISS Searches for Similar Chunks]
    H --> I[LLM via LangChain Answers the Question]
