import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


load_dotenv() # Load environment variables


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(
        document_chunks,OpenAIEmbeddings())
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
        
    ])

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context: {context}."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
        response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })

        return response['answer']


# app config
st.set_page_config(page_title="Website Chatbot", page_icon=":robot_face:", layout="wide")
st.title("Website Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! How can I assist you today?"),
    ]

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL to start the chatbot.")
else: 
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I assist you today?"),
        ]

     # Check if vector store is already created   
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
   
    vector_store = get_vectorstore_from_url(website_url)


    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


  
    # Conversation
    
    for message in st.session_state.chat_history: # loop iterates through the chat history stored in the session state
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        
        
    
    
    
    