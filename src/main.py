import os
from dotenv import load_dotenv
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from rouge_score import rouge_scorer
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent

load_dotenv(project_root / ".env")
openai = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

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
            # format the system prompt using keyword args that match the placeholders
            system_prompt = f"""You are a highly intelligent AI assistant specializing in physics. Your primary task is to synthesize information from a given context to provide a clear and concise answer to a user's query.

            Your Instructions:
            1. You will be provided with a user's query and a block of text labeled "CONTEXT".
            2. You must answer the user's query based ONLY on the information available in the CONTEXT. Do not use any prior knowledge or external information.
            3. Synthesize the information from the CONTEXT into a single, coherent paragraph. Do not use bullet points or lists.
            4. The final answer must strictly adhere to the requested summary length.
            5. If the CONTEXT does not contain enough information to answer the query, you must respond with the exact phrase: "I could not find a definitive answer in the provided documents."

            ---

            User Query: "{query}"
            Requested Length: "{self.length_specifications[summary_length]}"
            CONTEXT:
            "{combined_text}"

            ---"""

            response = openai.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
                ]
            )
            summary = str(response.choices[0].message.content).strip()
            return summary
        except KeyError as ke:
            print(f"Prompt formatting error: missing key {ke}")
            return "Error: prompt formatting failed."
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Error: Unable to generate summary at this time."

class SearchSystem:
    def __init__(self, db_path: str = str(project_root / "data/vectordb")):
        self.vector_store = chromadb.PersistentClient(path=db_path)
        self.collection = self.vector_store.get_or_create_collection(name="physics_corpus")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = ContentSummarizer()

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_n,
            include=["documents", "metadatas"]
        )
        
        formatted_results = []
        if results and results['documents']:
            for i, doc_text in enumerate(results['documents'][0]):
                formatted_results.append({
                    "text": doc_text,
                    "source": results['metadatas'][0][i].get('source_file', 'Unknown') #type: ignore
                })
        return formatted_results

    def search_and_summarize(self, query: str, summary_length: str = 'medium') -> Dict:
        documents = []
        summary = "Error: Unable to generate summary at this time."
        try:
            documents = self.search(query)
            if not documents:
                summary = "Could not find any relevant documents for the query."
            else:
                summary = self.summarizer.summarize_document(documents, query, summary_length)
        except Exception as e:
            print(f"Error during summarization: {e}")
        
        return {
            'query': query,
            'summary': summary,
            'source_documents': documents
        }

class Evaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.collection = chromadb.PersistentClient(path="./data/physics_vectordb").get_collection(name="physics_corpus")

    def evaluate_search_accuracy(self, test_queries: List[Dict]) -> Dict:
        results = []
        
        for test_case in test_queries:
            search_results = SearchSystem().search(test_case['query'])
            
            # Simple relevance check based on expected topics
            relevance_scores = []
            for result in search_results:
                score = self._calculate_relevance(
                    result['text'], 
                    test_case["expected_topics"]
                )
                relevance_scores.append(score)
            
            results.append({
                'query': test_case["query"],
                'avg_relevance': np.mean(relevance_scores),
                'top_relevance': max(relevance_scores) if relevance_scores else 0
            })
        
        return {
            'overall_accuracy': np.mean([r['avg_relevance'] for r in results]),
            'detailed_results': results
        }
    
    def _calculate_relevance(self, text: str, expected_topics: List[str]) -> float:
        text_lower = text.lower()
        topic_matches = sum(1 for topic in expected_topics if topic.lower() in text_lower)
        return topic_matches / len(expected_topics)
    
    def evaluate_summary_quality(self, reference_summaries: List[str], 
                                generated_summaries: List[str]) -> Dict:
        rouge_scores = []
        
        for ref, gen in zip(reference_summaries, generated_summaries):
            scores = self.rouge_scorer.score(ref, gen)
            rouge_scores.append(scores)
        
        # Average scores
        avg_scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            avg_scores[metric] = {
                'precision': np.mean([s[metric].precision for s in rouge_scores]),
                'recall': np.mean([s[metric].recall for s in rouge_scores]),
                'fmeasure': np.mean([s[metric].fmeasure for s in rouge_scores])
            }
        
        return avg_scores

if __name__ == "__main__":
    search_and_summarizer = SearchSystem()
    queries = [
        {'query': "What are the Newton's Laws of Motion?", 'expected_topics': ['Newton\'s Laws of Motion', 'First Law', 'Second Law', 'Third Law'], 'expected_summary': "Newton's three laws of motion are fundamental principles of classical mechanics that describe the relationship between an object and the forces acting upon it. The first law, the law of inertia, states that an object will remain at rest or in uniform motion in a straight line unless acted upon by an external force. The second law quantitatively describes how the net force acting on an object is equal to the product of its mass and acceleration (F=ma), meaning a greater force is required to accelerate a heavier object. The third law states that for every action, there is an equal and opposite reaction, explaining that forces always occur in pairs. Together, these laws provide a comprehensive framework for understanding the motion of everyday objects and are essential for fields ranging from engineering to celestial mechanics."},
        {'query': 'What is the principle of conservation of energy?', 'expected_topics': ['Conservation of Energy', 'Potential Energy', 'Kinetic Energy'], 'expected_summary': "The principle of conservation of energy is a foundational concept in physics, stating that the total energy of an isolated system remains constant over time. Energy cannot be created or destroyed; it can only be transformed from one form to another. For instance, in a simple pendulum, potential energy at its highest point is converted into kinetic energy as it swings downwards, and back into potential energy as it rises. This principle applies across all scales, from subatomic particles to galactic clusters, and encompasses various forms of energy, including kinetic, potential, thermal, chemical, and nuclear. It is a powerful tool for analyzing physical systems, as it allows scientists and engineers to predict the behavior of a system without needing to know the intricate details of the processes involved."},
        {'query': 'What is the Theory of Relativity?', 'expected_topics': ['Theory of Relativity', 'Physics', 'Albert Einstein'], 'expected_summary': "Albert Einstein's theory of special relativity, published in 1905, revolutionized our understanding of space and time. It is based on two postulates: first, that the laws of physics are the same for all observers in uniform motion, and second, that the speed of light in a vacuum is constant for all observers, regardless of their motion or the motion of the light source. These seemingly simple ideas lead to extraordinary consequences, including time dilation, where time passes more slowly for a moving observer, and length contraction, where objects appear shorter in their direction of motion. The theory also established the famous mass-energy equivalence, expressed by the equation E=mc², demonstrating that mass and energy are interchangeable."},
        {'query': 'How does the photoelectric effect demonstrate the particle nature of light?', 'expected_topics': ['Photoelectric Effect', 'Particle Nature of Light', 'Wave-Particle Duality', 'Photons'], 'expected_summary': "The photoelectric effect is a phenomenon where electrons are emitted from a material when it is exposed to light of a sufficiently high frequency. This effect provided crucial evidence for the particle nature of light, as the classical wave theory could not explain its observed characteristics. Specifically, it was found that the emission of electrons depends on the light's frequency, not its intensity; below a certain threshold frequency, no electrons are emitted, no matter how bright the light. Albert Einstein explained this by proposing that light consists of discrete packets of energy called photons. The energy of a photon is directly proportional to its frequency, and an electron is only ejected if it absorbs a single photon with enough energy to overcome the material's binding force."},
    ]

    for query in queries:
        result = search_and_summarizer.search_and_summarize(query['query'])
        search_accuracy = Evaluator().evaluate_search_accuracy([query])
        summary_quality = Evaluator().evaluate_summary_quality([query['expected_summary']], [result['summary']])
        print(f"\nQuery: {result['query']}")
        print(f"Summary: {result['summary']}")
        print(f"Number of source documents: {result['n_sources']}")
        print(f"Search Accuracy: {search_accuracy['overall_accuracy']}")
        print(f"Summary Quality: {summary_quality}")
        print("-" * 40)