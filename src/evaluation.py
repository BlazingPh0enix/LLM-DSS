import chromadb
import openai
from rouge_score import rouge_scorer
import json
from pathlib import Path
import random
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

DB_PATH = "./data/physics_vectordb"
COLLECTION_NAME = "physics_corpus"
TEST_SET_FILE = Path("./test_set.json")
TEST_SET_SIZE = 20 
RETRIEVAL_K = 5 

class Evaluator:
    def __init__(self, search_and_summarize_instance):
        self.search_summarizer = search_and_summarize_instance
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def create_test_set(self):
        if TEST_SET_FILE.exists():
            print(f"Test set already exists at {TEST_SET_FILE}. Loading from file.")
            with open(TEST_SET_FILE, "r") as f:
                return json.load(f)

        print("Creating a new test set...")
        # Get a random sample of documents from the database
        total_docs = self.collection.count()
        random_ids = [f"chunk_{i}" for i in random.sample(range(total_docs), TEST_SET_SIZE)]
        
        documents = self.collection.get(ids=random_ids, include=["documents"])['documents']
        
        test_set = []
        for doc_text in documents: # type: ignore
            if not doc_text:
                continue
            
            print(f"-> Generating query for document snippet: '{doc_text[:100]}...'")
            try:
                # Use GPT to generate a query
                response = openai.chat.completions.create(
                    model="gpt-5-nano-2025-08-07",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Based on the following text, generate a single, concise question that this text could answer."},
                        {"role": "user", "content": doc_text}
                    ],
                    max_tokens=50,
                    temperature=0.5
                )
                query = str(response.choices[0].message.content).strip()
                test_set.append({
                    "expected_document": doc_text,
                    "generated_query": query
                })
            except Exception as e:
                print(f"   - Failed to generate query: {e}")

        # Save the generated test set to a file
        with open(TEST_SET_FILE, "w") as f:
            json.dump(test_set, f, indent=4)
        
        print(f"\nTest set created and saved to {TEST_SET_FILE}")
        return test_set

    def evaluate_retrieval(self, test_set: list) -> dict:
        print("\n--- Evaluating Search Retrieval ---")
        hits = 0
        total = len(test_set)

        for item in test_set:
            query = item['generated_query']
            expected_doc = item['expected_document']
            
            # Use the search function from the main class
            retrieved_docs = self.search_summarizer.search(query, top_n=RETRIEVAL_K)
            
            # Check if the text of the expected document is in the retrieved results
            retrieved_texts = [doc['text'] for doc in retrieved_docs]
            if expected_doc in retrieved_texts:
                hits += 1
        
        hit_rate = (hits / total) * 100 if total > 0 else 0
        results = {
            "retrieval_hit_rate_at_k": RETRIEVAL_K,
            "accuracy_percentage": f"{hit_rate:.2f}%"
        }
        print(f"Retrieval Hit Rate @{RETRIEVAL_K}: {hit_rate:.2f}%")
        return results

    def evaluate_summaries(self, test_set: list) -> dict:
        """
        Evaluates summary quality using ROUGE scores.
        """
        print("\n--- Evaluating Summary Quality ---")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        total_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        total = len(test_set)

        for item in test_set:
            query = item['generated_query']
            reference_summary = item['expected_document'] # The original doc is our reference
            
            # Use the summarization function from the main class
            generated_summary = self.search_summarizer.summarize(query, "2-3")

            if generated_summary.startswith("I could not find"):
                print(f"Skipping summary evaluation for query: '{query}' (no answer found)")
                total -= 1
                continue

            scores = scorer.score(reference_summary, generated_summary)
            total_scores['rouge1'] += scores['rouge1'].fmeasure
            total_scores['rouge2'] += scores['rouge2'].fmeasure
            total_scores['rougeL'] += scores['rougeL'].fmeasure

        avg_scores = {key: (value / total) * 100 for key, value in total_scores.items()} if total > 0 else {}
        
        print(f"Average ROUGE-1 F-score: {avg_scores.get('rouge1', 0):.2f}%")
        print(f"Average ROUGE-2 F-score: {avg_scores.get('rouge2', 0):.2f}%")
        print(f"Average ROUGE-L F-score: {avg_scores.get('rougeL', 0):.2f}%")
        return avg_scores