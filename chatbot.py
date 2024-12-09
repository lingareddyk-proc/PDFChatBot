import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

# Add these imports
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def init_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",  # You can choose different models
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
def get_conversational_chain(vectorstore, user_question):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Get retriever directly from vectorstore
    retriever = vectorstore.as_retriever()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    response = chain({"question": user_question})
    return response['answer']
def vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    st.title("Chat with your PDFs 💬")
    
    # Store vectorstore in session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        if st.session_state.vectorstore:
            user_input(user_question)
        else:
            st.warning("Please upload and process PDF files first")

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                st.session_state.vectorstore = vector_store(text_chunks)
                st.success("Done")

def user_input(user_question):
    response = get_conversational_chain(st.session_state.vectorstore, user_question)
    st.write(response)

if __name__ == "__main__":
    main()