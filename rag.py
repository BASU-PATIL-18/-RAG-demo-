import streamlit as st# to create a web app interface
import os # to access environment variables
from dotenv import load_dotenv # to load environment variables from .env file
load_dotenv() # Load environment variables from .env file
from PyPDF2 import PdfReader # to read PDF files
import google.generativeai as genai # to use Google's Gemini API for generating responses
from langchain_huggingface import HuggingFaceEmbeddings # to create embeddings using Hugging Face models
from langchain_core.documents import Document # to create Document objects for storing text and metadata
from langchain_community.vectorstores import FAISS # to create a vector store for storing and retrieving embeddings
from langchain_text_splitters import CharacterTextSplitter # to split text into smaller chunks for embedding
key=os.getenv('google_api_key')  # Get the Google API key from environment variables
genai.configure(api_key=key)  # Configure the Gemini API with the API key
model=genai.GenerativeModel('gemini-2.5-flash')  # Initialize the Gemini model
def load_embedding():
  return HuggingFaceEmbeddings(model= "all-MiniLM-L6-v2")  # Load the Hugging Face embedding model converts text into vector representations
st.set_page_config(page_title="RAG Demo", page_icon=":books:")  # Set the page title and icon for the Streamlit app
st.title("RAG assistant :blue[using Gemini]")  # Set the title of the Streamlit app
st.subheader(':green[your intelligent document assistant]')    # Set a subheader for the app
with st.spinner('Loading the embedding model...'):  # Show a spinner while loading the embedding model
  embedding=load_embedding()  # Load the embedding model
  uploaded_file=st.file_uploader("Upload your PDF document here", type=["pdf"])  # Create a file uploader for PDF documents
  if uploaded_file:
    pdf=PdfReader(uploaded_file)  # Read the uploaded PDF file
    raw_text=""  # Initialize an empty string to store the extracted text
    for page in pdf.pages:  # Iterate through each page in the PDF
      raw_text+=page.extract_text()  # Extract text from the page and append it to raw_text
    if raw_text.strip():
      doc=Document(page_content=raw_text)  # Create a Document object with the extracted text and metadata
      text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Initialize a text splitter to split the text into smaller chunks
      chunk_text=text_splitter.split_documents([doc])  # Split the document into smaller chunks
      text=[i.page_content for i in chunk_text]  # Extract the text content from each chunk /converts the chunks into simple text format
      vector_store=FAISS.from_texts(text, embedding)  # Create a FAISS vector store from the text chunks and their embeddings
      retrive=vector_store.as_retriever()  # Create a retriever from the vector store to retrieve relevant chunks based on user queries
      st.success("Document loaded successfully! You can now ask questions about the document.")  # Show a success message when the document is loaded
      user_input=st.text_input("Ask a question about the document:")  # Create a text input for the user to ask questions about the document
      if user_input:
        with st.chat_message('human'):  # Display the user's question in the chat interface
          with st.spinner('Generating response...'):  # Show a spinner while generating the response
           relevant_docs=retrive.invoke(user_input)  # it finds most similar Retrieve relevant document chunks based on the user's question
           content="\n\n".join([doc.page_content for doc in relevant_docs])  # Combine the relevant document chunks into a single context string
           prompt=f" you are an AI expert .use the content generated to Answer the question asked by the user.if u are unsure you should say 'i am unsure about the question asked' content:\n\n{content}\n\nQuestion: {user_input}" # Create a prompt for the Gemini model that includes the context and the user's question
           response=model.generate_content(prompt)  # Generate a response from the Gemini model based on the prompt
           st.markdown('### :green[Response]')
           st.write(response.text)  # Display the generated response in the Streamlit app
  else:
    st.warning("The uploaded PDF does not contain any text. Please upload a valid PDF document.")  # Show an error message if the PDF does not contain any text
    