from typing import List, Dict
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
import re
from pathlib import Path
import os

os.chdir(Path(__file__).parent.parent)  # Change to project root directory

class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def chunk_text(self, text: str, segment_size: int = 100000, chunk_size: int = 512) -> List[Dict]:
        print("Cleaning and chunking text...")
        # Basic cleaning
        text = re.sub(r'\\s+', ' ', text).strip()
        
        all_chunks = []
        
        for i in range(0, len(text), segment_size):
            segment = text[i:i + segment_size]
            doc = self.nlp(segment)
            
            current_chunk = ""
            for sent in doc.sents:
                if len(current_chunk.split()) + len(sent.text.split()) > chunk_size and current_chunk:
                    all_chunks.append({'text': current_chunk.strip()})
                    current_chunk = ""
                current_chunk += sent.text + " "
            
            if current_chunk:
                all_chunks.append({'text': current_chunk.strip()})
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

class VectorStore:
    def __init__(self, db_path: str = "./data/physics_vectordb"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="physics_corpus")

    def add_documents(self, chunks: List[Dict], metadatas: List[Dict]):
        if not chunks: return
        texts = [chunk['text'] for chunk in chunks]
        ids = [f"{meta['source_file']}_{meta['chunk_id']}" for meta in metadatas]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # Add to collection in batches
        for i in range(0, len(texts), 1000):
            self.collection.add(
                embeddings=embeddings[i:i+1000],
                documents=texts[i:i+1000],
                metadatas=metadatas[i:i+1000], #type: ignore
                ids=ids[i:i+1000]
            )
        print("-> Documents added to vector store.")

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        formatted_results = []
        for i in range(len(results['documents'][0])): # type: ignore
            formatted_results.append({
                'text': results['documents'][0][i], # type: ignore
                'metadata': results['metadatas'][0][i], # type: ignore
                'distance': results['distances'][0][i] if results['distances'] else None
            })
        
        return formatted_results

if __name__ == "__main__":
    text_dir = Path("./data/processed_text")
    txt_files = list(text_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError("No processed text files found. Run 'extract_text.py' first.")

    preprocessor = DataPreprocessor()
    vector_store = VectorStore()

    for txt_file in txt_files:
        print(f"--- Processing file: {txt_file.name} ---")
        
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = preprocessor.chunk_text(content)
        
        metadatas = [{
            'source_file': txt_file.name,
            'chunk_id': i,
        } for i, chunk in enumerate(chunks)]
        
        vector_store.add_documents(chunks, metadatas)
        

    print("All text files have been processed and embedded.")