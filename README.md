
from fpdf import FPDF.docx

1 / 5
from fpdf import FPDF

# Create a PDF document
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Set title
pdf.set_font("Arial", size=16, style='B')
pdf.cell(200, 10, "Retrieval-Augmented Generation (RAG) Pipeline for PDF Query Handling", ln=True, align="C")

# Add an introductory section
pdf.ln(10)
pdf.set_font("Arial", size=12)
intro_text = """
The goal of this project is to implement a Retrieval-Augmented Generation (RAG) pipeline  
that allows users to interact with semi-structured data stored in PDF files. This system  
extracts data from PDF files, generates embeddings for efficient retrieval, and uses an LLM  
for query handling and response generation.

The pipeline includes the following key steps:
1. Data Ingestion: Extract data from PDF files, chunk it, generate embeddings, and store them.
2. Query Handling: Convert queries into embeddings, perform similarity search, and generate responses.
3. Comparison Queries: Handle complex queries that involve comparing data across multiple PDFs.
4. Response Generation: Use retrieved data to generate accurate and informative responses.
"""

pdf.multi_cell(0, 10, intro_text)

# Code implementation section
2 / 5
pdf.ln(10)
pdf.set_font("Arial", size=12, style='B')
pdf.cell(200, 10, "Python Code for RAG Pipeline", ln=True)

pdf.set_font("Arial", size=10)
code_text = """
import fitz  # PyMuPDF for PDF parsing
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# Load pre-trained model for embeddings (using a BERT-based model)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# FAISS setup: initialize an index to store vectors
index = faiss.IndexFlatL2(768)  # Dimension size for MiniLM-L6-v2 embeddings

def extract_text_from_pdf(pdf_path):
   """
   Extract text from a PDF using PyMuPDF (fitz).
   """
   doc = fitz.open(pdf_path)
   text = ""
   for page in doc:
       text += page.get_text()
   return text

def chunk_text(text, chunk_size=500):
   """
3 / 5
   Split the extracted text into chunks of a given size.
   """
   chunks = []
   start = 0
   while start < len(text):
       chunks.append(text[start:start+chunk_size])
       start += chunk_size
   return chunks
   
from fpdf import FPDF.docx

1 / 5
from fpdf import FPDF

# Create a PDF document
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Set title
pdf.set_font("Arial", size=16, style='B')
pdf.cell(200, 10, "Retrieval-Augmented Generation (RAG) Pipeline for PDF Query Handling", ln=True, align="C")

# Add an introductory section
pdf.ln(10)
pdf.set_font("Arial", size=12)
intro_text = """
The goal of this project is to implement a Retrieval-Augmented Generation (RAG) pipeline  
that allows users to interact with semi-structured data stored in PDF files. This system  
extracts data from PDF files, generates embeddings for efficient retrieval, and uses an LLM  
for query handling and response generation.

The pipeline includes the following key steps:
1. Data Ingestion: Extract data from PDF files, chunk it, generate embeddings, and store them.
2. Query Handling: Convert queries into embeddings, perform similarity search, and generate responses.
3. Comparison Queries: Handle complex queries that involve comparing data across multiple PDFs.
4. Response Generation: Use retrieved data to generate accurate and informative responses.
"""

pdf.multi_cell(0, 10, intro_text)

# Code implementation section
2 / 5
pdf.ln(10)
pdf.set_font("Arial", size=12, style='B')
pdf.cell(200, 10, "Python Code for RAG Pipeline", ln=True)

pdf.set_font("Arial", size=10)
code_text = """
import fitz  # PyMuPDF for PDF parsing
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# Load pre-trained model for embeddings (using a BERT-based model)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# FAISS setup: initialize an index to store vectors
index = faiss.IndexFlatL2(768)  # Dimension size for MiniLM-L6-v2 embeddings

def extract_text_from_pdf(pdf_path):
   """
   Extract text from a PDF using PyMuPDF (fitz).
   """
   doc = fitz.open(pdf_path)
   text = ""
   for page in doc:
       text += page.get_text()
   return text

def chunk_text(text, chunk_size=500):
   """
3 / 5
   Split the extracted text into chunks of a given size.
   """
   chunks = []
   start = 0
   while start < len(text):
       chunks.append(text[start:start+chunk_size])
       start += chunk_size
   return chunks

def generate_embeddings(text_chunks):
   """
   Generate embeddings for each chunk of text using a pre-trained model.
   """
   embeddings = []
   for chunk in text_chunks:
       inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
       with torch.no_grad():
           outputs = model(**inputs)
           embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
       embeddings.append(embedding)
   return np.array(embeddings)

def store_embeddings(embeddings):
   """
   Store embeddings in the FAISS index for efficient similarity search.
   """
   index.add(embeddings)

def query_to_embedding(query):
   """
4 / 5
   Convert the user's query into an embedding.
   """
   inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
   with torch.no_grad():
       outputs = model(**inputs)
       embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
   return embedding

def retrieve_top_k_similar(query_embedding, k=3):
   """
   Retrieve the top-k most similar chunks from the FAISS index.
   """
   distances, indices = index.search(np.array([query_embedding]), k)
   return distances, indices

def generate_response(retrieved_indices, text_chunks):
   """
   Generate a response based on the retrieved chunks.
   """
   response = ""
   for idx in retrieved_indices[0]:
       response += text_chunks[idx] + "\\n\\n"
   return response

# Main process
pdf_path = 'sample.pdf'  # Path to your PDF file
text = extract_text_from_pdf(pdf_path)

# Chunk the text
chunks = chunk_text(text)
   
