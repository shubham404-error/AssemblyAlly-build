import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import requests
from dotenv import load_dotenv
import os
import google.generativeai as gen_ai
import google.ai.generativelanguage as glm
# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file.read())
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to embed text
def embed_text(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    return embeddings

# Function to search for similar sentences in FAISS index
def search_similar_sentences(query_embedding, index, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# Streamlit App
st.title("Assembly Ally: Your Mechanic Assistant")

uploaded_file = st.file_uploader("Upload a PDF Manual")

if uploaded_file is not None:
    # Process PDF and create FAISS index
    text = extract_text_from_pdf(uploaded_file)
    embeddings = embed_text(text)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Initialize chat session
    model = glm.GenerativeModel(API_KEY)
    chat = model.start_chat(history=[])

    # Get user query
    query = st.text_input("Ask a question about the manual:")

    if query:
        # Add user query to chat history
        chat.send_message(query)

        # Embed query and search for similar sentences
        query_embedding = embed_text([query])[0]
        similar_sentence_indices = search_similar_sentences(query_embedding, index)

        # Retrieve context
        context = " ".join([text.split(".")[idx] for idx in similar_sentence_indices])

        # Generate response with context and safety settings
        response = chat.send_message(
            context, safety_settings={"HARASSMENT": "block_none"}
        )

        # Display results
        st.write("Relevant Context:")
        st.write(context)
        st.write("AI Response:")
        st.write(response.text)
