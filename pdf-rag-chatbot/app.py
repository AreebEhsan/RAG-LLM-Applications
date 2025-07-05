from dotenv import load_dotenv
import os
import streamlit as st
import PyPDF2 as pypdf
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain   
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon=":robot:")
    st.header("PDF RAG Chatbot")

    pdf = st.file_uploader("Upload a PDF file", type="pdf") # Upload PDF file

    if pdf is not None:
        pdf_reader = pypdf.PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        st.text_area("PDF Content", text, height=300)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # Adjust chunk size as needed
            chunk_overlap=200,  # Adjust overlap as needed
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        st.write(f"Number of chunks created: {len(chunks)}")
        
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings) # Facebook AI Similarity Search (FAISS) vector store

        user_question = st.text_input("Ask a question about the PDF content")
        if user_question:
            docs = vector_store.similarity_search(user_question)
            st.write(f"Found {len(docs)} relevant chunks.",docs)
              
            llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)


                


            
            
            st.write("Response from the model:")
            st.write(response)
            

        # Here you would typically call your RAG model to process the text
        # For demonstration, we will just show a placeholder response
        if st.button("Ask a question"):
            st.write("This is where the RAG model's response would be displayed.")

if __name__ == "__main__":
    main()