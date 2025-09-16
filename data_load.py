# NOTE: This script is intended to be run once to set up necessary directories and download required models and datasets.

import nltk
import sys
import subprocess
from pathlib import Path
import urllib.request

# Ensure necessary directories exist
directories = [
    "physics_corpus",
    'models',
    "physics_vectordb"
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Download necessary NLTK and spaCy models
def download_nltk_spacy_models():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    except Exception as e:
        print(f"An error occurred while downloading nltk and spaCy models: {e}")

# Download the physics corpus
def download_corpus():
    # Define the directory to save the corpus
    corpus_dir = Path("physics_corpus")

    # URLs of the textbooks to download
    textbook_urls = {
        "High School Physics": "https://assets.openstax.org/oscms-prodcms/media/documents/Physics-WEB_Sab7RrQ.pdf",
        "College Physics Vol 1": "https://assets.openstax.org/oscms-prodcms/media/documents/University_Physics_Volume_1_-_WEB.pdf",
        "College Physics Vol 2": "https://assets.openstax.org/oscms-prodcms/media/documents/University_Physics_Volume_2_-_WEB.pdf",
        "College Physics Vol 3": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume3-WEB.pdf", 
    }

    # Download each textbook and save it to the corpus directory
    for title, url in textbook_urls.items():
        response = urllib.request.urlopen(url)
        pdf_data = response.read()
        pdf_path = corpus_dir / f"{title}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)

if __name__ == "__main__":
    download_nltk_spacy_models()
    download_corpus()