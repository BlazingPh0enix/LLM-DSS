import pandas as pd
from typing import List, Dict, Tuple, Any
import numpy as np
from datetime import datetime
import pymupdf
import pymupdf4llm
import os
import io
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
import openai
from transformers import pipeline, AutoTokenizer
import re
from pathlib import Path

class DataExtractor:
    def __init__(self):
        self.equation_patterns = [
            r'\$.*?\$',  # LaTeX inline equations
            r'\\\[.*?\\\]',  # LaTeX display equations
            r'\\\(.*?\\\)',  # LaTeX parentheses equations
            r'[A-Za-z]+\s*=\s*[^.]*?(?=\s|$|\.)',  # Basic equations like F=ma
            r'\b[A-Z][a-z]*\s*=\s*.*?(?=\s[A-Z]|$|\.|,)',  # Physics formulas
            r'\d+\.\d+\s*[a-zA-Z]+',  # Numbers with units
            r'[A-Za-z]+\s*[+\-*/=]\s*[A-Za-z0-9]+',  # Mathematical expressions
        ]

    def _is_likely_equation(self, text: str, span: Dict)-> bool:
        #Check for mathematical symbols
        math_symbols = set('=+-*/^<>√∑∫πθΔ∞')
        has_math_symbol = any(char in math_symbols for char in text)

        # Check for equation patterns
        matches_pattern = any(re.search(pattern, text) for pattern in self.equation_patterns)

        # Check for font characteristics
        font_size = span.get('font', '').lower()
        is_math_font = any(keyword in font_size for keyword in ['math', 'symbol', 'italic', 'bold', 'times'])

        return has_math_symbol or matches_pattern or is_math_font
    
    def _merge_text_sources(self, regular_text: str, markdown_text: str) -> str:
        # Use markdown text where equations are detected
        if '$' in markdown_text or '$$' in markdown_text:
            return markdown_text
        return regular_text

    def extract_text_from_pdf(self, pdf_path):
        # Open the PDF file
        doc = pymupdf.open(pdf_path)

        # Initialize storage for extracted content
        extracted_content = {
            'text': '',
            'equations': [],
            'metadata': {
                'title': doc.metadata.get('title', ''),  # type: ignore
                'page_count': doc.page_count,
                'file_path': pdf_path
            }
        }
        print(f"Processing PDF: {extracted_content['metadata']['title']}")

        full_text = []
        equations = []
        
        print(f"Number of pages to process in the PDF: {doc.page_count}")
        print("Initializing text extraction...")
        # Iterate through each page
        for page_num in range(doc.page_count):
            print(f"Extracting text from page {page_num + 1}/{doc.page_count}")
            page = doc.load_page(page_num)

            # Extract text blocks
            blocks = page.get_text('dict') # type: ignore
            page_text = " "

            # Process each block and identify equations
            for block in blocks['blocks']:
                if 'lines' in block:
                    for line in block['lines']:
                        for span in line['spans']:
                            text += span['text']

                            # Check if the span is likely an equation
                            if self._is_likely_equation(text, span):
                                equations.append({
                                    'text': text,
                                    'page': page_num + 1
                                })

                            page_text += text + " "

            try:
                markdown_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
                # Merge regular text with markdown text
                page_text = self._merge_text_sources(page_text, markdown_text)
            except Exception as e:
                print(f"Markdown conversion failed on page {page_num + 1}: {e}")

            full_text.append(page_text)

        print("Text extraction completed.")

        extracted_content['text'] = "\n\n".join(full_text)
        extracted_content['equations'] = equations

        doc.close()

        return extracted_content
    
class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.equation_patterns = DataExtractor().equation_patterns

    def _clean_text(self, text: str) -> str:
        # Clean text while preserving equations
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines

        physics_preserving_patterns = [
            (r'([A-Za-z])\s*=\s*([^.]+)', r'\1 = \2'),  # Equations
            (r'(\d+)\s*×\s*(\d+)', r'\1 × \2'),  # Multiplication
            (r'(\d+)\s*m/s', r'\1 m/s'),  # Units
        ]

        for pattern, replacement in physics_preserving_patterns:
            text = re.sub(pattern, replacement, text)

        return text
    
    def _extract_equations_from_chunk(self, chunk: str) -> List[str]:
        equations = []

        for pattern in self.equation_patterns:
            matches = re.findall(pattern, chunk)
            equations.extend(matches)

        return list(set(equations))  # Return unique equations
    
    def chunk_text(self, text: str, chunk_size: int = 700) -> List[Dict]:
        # Clean the text
        print("Cleaning text...")
        cleaned_text = self._clean_text(text)

        # Split text into chunks based on semantics
        doc = self.nlp(cleaned_text)
        chunks = []
        current_chunk = ""
        current_size = 0

        print("Chunking text...")
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_size = len(sent_text.split())

            if current_size + sent_size > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'size': current_size,
                    'equations': self._extract_equations_from_chunk(current_chunk)
                    })
                current_chunk = sent_text + " "
                current_size = sent_size
            else:
                current_chunk += sent_text + " "
                current_size += sent_size

        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'size': current_size,
                'equations': self._extract_equations_from_chunk(current_chunk)
                })
        print(f"Total chunks created: {len(chunks)}")
            
        return chunks
    
class VectorStore:
    def __init__(self, db_path: str = "./physics_vectordb"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None

    def create_collection(self, collection_name: str):
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Physics corpus for LLM-based DSS"}
            )
            print(f"Collection '{collection_name}' created successfully.")
        except:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists. Using the existing collection.")

        return self.collection
    
    def add_documents(self, chunks: List[Dict], metadata: List[Dict]):
        if not self.collection:
            self.create_collection("physics_corpus")

        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()

        ids = [f"chunk_{i}" for i in range(len(texts))]

        self.collection.add( # type: ignore
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata, # type: ignore
            ids=ids
        )

        print(f"Added {len(texts)} documents to the vector store.")

if __name__ == "__main__":
    corpus_dir = Path("./physics_corpus")
    pdf_files = list(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the specified directory.")
    
    extractor = DataExtractor()
    preprocessor = DataPreprocessor()

    all_chunks = []
    all_metadata = []

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.name}")

        # Extract text and equations from the PDF
        extracted_content = extractor.extract_text_from_pdf(str(pdf_file))

        if not extracted_content:
            continue
        
        # Preprocess and chunk the text
        chunks = preprocessor.chunk_text(extracted_content['text'])

        # Prepare metadata for each chunk
        for i, chunk in enumerate(chunks):
            metadata = {
                'source_file': pdf_file.name,
                'chunk_id': i,
                'has_equations': len(chunk['equations']) > 0,
                'equations': chunk['equations'],
            }
            all_metadata.append(metadata)
        
        all_chunks.extend(chunks)

        # Add the chunks to the vector store
        vector_store = VectorStore()
        vector_store.add_documents(all_chunks, all_metadata)