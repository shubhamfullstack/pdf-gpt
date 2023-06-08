from apikey import apikey
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
import os
import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey

repo_id = "google/flan-t5-xl" 
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":100})


st.title("📁 Multi Pdf Chat 📁")
st.subheader("Explore the power of Generative AI with your own pdfs")

st.subheader("Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="You can upload multiple PDF files.")

if uploaded_file:
    file_contents = uploaded_file.read()
    file_name = uploaded_file.name
    file_size = uploaded_file.size

    # Save file to disk
    save_path = os.path.join("pdfs", file_name)
    with open(save_path, "wb") as f:
        f.write(file_contents)
    st.success(f"Saved file: {file_name}")
    loader = DirectoryLoader('pdfs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
    persist_directory = 'db'
    embedding = instructor_embeddings
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding,persist_directory=persist_directory)
    vectordb.persist()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever()
    query = "What is the factors that affects inflation?"
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
# col1 = st.columns(1)

# # Left column - File uploader
# with col1:
    

# Right column - Input text field
# with col2:
#     st.subheader("Enter text")
#     user_input = st.text_input("Enter some text")


# llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":100})


# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "Who won the FIFA World Cup in the year 1994? "

# print(llm_chain.run(question))