import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from apikey import apikey
from langchain.chains import RetrievalQA
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey

def save_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        with open(os.path.join('src/uploads', uploaded_file.name), 'wb') as file:
            file.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully!")

def create_vector_store():
    loader = DirectoryLoader('src/uploads', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    persist_directory = 'db'
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding,persist_directory=persist_directory)
    vectordb.persist()
    st.success("Vector Store is Created Successfully!")

st.title("📁 Multi Pdf Chat 📁")
st.subheader("Explore the power of Generative AI with your own pdfs")

with st.sidebar:
    st.subheader("Upload your PDF")
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", help="You can upload multiple PDF files.", accept_multiple_files=True)
    if uploaded_files:
        save_uploaded_files(uploaded_files)
        create_vector_store()

prompt = st.text_input('Ask your question here')
repo_id = "facebook/bart-large-cnn" 
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":100})
model_name = "sentence-transformers/all-mpnet-base-v2"
persist_directory = 'db'
embedding = HuggingFaceEmbeddings(model_name=model_name)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()

if prompt: 
    title_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    title = title_chain(prompt)
    st.write(title['result']) 