import os
from dotenv import load_dotenv
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocess import VectorStore

load_dotenv()
openai = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
system_prompt = str(os.getenv("SYSTEM_PROMPT"))

class ContentSummarizer:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.length_specifications = {
            'short': '100 words',
            'medium': '150-200 words',
            'long': '300-500 words'
        }

    def summarize_document(self, documents: List[Dict], query: str, summary_length: str = 'medium') -> str:
        
        combined_text = "\n\n".join([doc['text'] for doc in documents])

        try:
            response = openai.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                {"role": "system", "content": system_prompt.format(query, length=self.length_specifications.get(summary_length), context=combined_text)},
                {"role": "user", "content": query}
                ]
            )
            summary = str(response.choices[0].message.content).strip()
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Error: Unable to generate summary at this time."

class SearchSystem:
    def __init__(self):
        self.summarizer = ContentSummarizer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = chromadb.PersistentClient(path="./physics_vectordb").get_collection(name="physics_corpus")
        
    def search_and_summarize(self, query: str, n_results: int = 5, summary_length: str = 'medium') -> Dict:
        search_results = VectorStore().search(query, n_results)

        summary = self.summarizer.summarize_document(search_results, query, summary_length)
        return {
            "query": query,
            "summary": summary,
            'source_documents': search_results,
            'n_sources': len(search_results)
        }

class SystemEvaluator:
    def __init__(self):
        self.search_system = SearchSystem()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_search_accuracy(self, test_queries: List[Dict]) -> Dict:
        results = []
        
        for test_case in test_queries:
            
            search_results = VectorStore().search(test_case['query'])

        return {}