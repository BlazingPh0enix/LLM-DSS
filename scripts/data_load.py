# NOTE: This script is intended to be run once to set up necessary directories and download required models and datasets.

import nltk
import sys
import subprocess
from pathlib import Path
import urllib.request
import os

os.chdir(Path(__file__).parent.parent)  # Change to project root directory

# Data directory
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Ensure necessary directories exist
directories = [
    "physics_corpus",
    "physics_vectordb",
    'processed_text',
]

for directory in directories:
    dir_path = Path(data_dir / directory)
    dir_path.mkdir(exist_ok=True)

# Download necessary NLTK and spaCy models
def download_nltk_spacy_models():
    try:
        print("Installing necessary NLTK models")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK models installed successfully")
        print("Installing necessary spaCy models")
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
        print("spaCy models installed successfully")
    except Exception as e:
        print(f"An error occurred while downloading nltk and spaCy models: {e}")

# Download the physics corpus
def download_corpus():
    # Define the directory to save the corpus
    corpus_dir = Path("data/physics_corpus")

    # URLs of the textbooks to download
    textbook_urls = {
        "High School Physics": "https://assets.openstax.org/oscms-prodcms/media/documents/Physics-WEB_Sab7RrQ.pdf",
        "College Physics Vol 1": "https://assets.openstax.org/oscms-prodcms/media/documents/University_Physics_Volume_1_-_WEB.pdf",
        "College Physics Vol 2": "https://assets.openstax.org/oscms-prodcms/media/documents/University_Physics_Volume_2_-_WEB.pdf",
        "College Physics Vol 3": "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume3-WEB.pdf", 
    }

    # Download each textbook and save it to the corpus directory
    for title, url in textbook_urls.items():
        print(f"Downloading {title}...")
        response = urllib.request.urlopen(url)
        pdf_data = response.read()
        pdf_path = corpus_dir / f"{title}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)
        print(f"Saved {title} to {pdf_path}")

if __name__ == "__main__":
    download_nltk_spacy_models()
    download_corpus()