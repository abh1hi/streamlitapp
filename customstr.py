import streamlit as st
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (like API keys) from a .env file
load_dotenv()

# Initialize the conversational AI model (LLM) using the Google Gemini API
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# Initialize embeddings using Google Gemini's embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  

# Define the system-level instructions for the LLM
system_prompt = (
    "You are an assistant for question-answering tasks. Related to uploaded file, "
    "you provide accurate and helpful answers."
    "when user says hi you provide few details about your knowledge of uploaded document and provide few example queries user can ask about"
    "Use the following pieces of retrieved context to answer, "
    "if you don't know the answer say 'I don't know'."
    "Answer concisely."
    "\n\n{context}"
)

# Create a chat prompt template for the system message and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # System-level instructions
        ("human", "{input}"),  # The user's input will replace "{input}"
    ]
)

# Initialize global variables for the vector store and RAG chain
vectorstore = None
rag_chain = None

# Function to clear the system cache for Chroma
def clear_chroma_cache():
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    st.write("Chroma system cache cleared.")

# Function to handle PDF file path and load the document
def load_pdf(file_path):
    clear_chroma_cache()  # Clear the cache before loading the new document
    
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file_path)
    data = loader.load()  # Load the PDF data into a document object

    # Initialize a text splitter to break the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)  # Split the document into smaller text chunks

    # Create a vector store from the documents and their embeddings using Chroma
    global vectorstore
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="path_to_your_local_directory")

    # Initialize a retriever to retrieve documents based on similarity
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Create the RAG chain
    global rag_chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.write(f"PDF at '{file_path}' successfully loaded and processed.")

# Function to handle the user's query and display response in the Streamlit interface
def handle_query(user_input):
    global rag_chain
    if rag_chain is None:
        st.write("Please load a PDF file first.")
        return

    if user_input.lower() == 'exit':
        st.write("Exiting the application.")
        return

    # Invoke the RAG chain with the user's input and retrieve the response
    response = rag_chain.invoke({"input": user_input})
    
    # Display the user's input and the assistant's response in the chat window
    st.write(f"You: {user_input}")
    st.write(f"Assistant: {response['answer']}")

# Streamlit UI components
st.title("PDF Document QA Assistant")

# Set the predefined file path
file_path = "20240716890312078.pdf"  # Replace with your actual PDF file path

# Load the predefined PDF file
load_pdf(file_path)

# User input for questions
user_input = st.text_input("Ask a question about the document:")

# If user submits a question, handle the query
if user_input:
    handle_query(user_input)
