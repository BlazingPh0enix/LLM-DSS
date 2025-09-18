# Physics Knowledge Base Search System

A sophisticated document search and summarization system powered by Large Language Models (LLMs) and vector databases. The system provides intelligent search capabilities across physics textbooks and generates comprehensive, structured summaries for user queries.

## Features

- **Semantic Search**: Advanced vector-based search across physics knowledge base
- **AI-Powered Summarization**: Structured responses with configurable length options
- **Multiple Summary Formats**: Short (100 words), Medium (150-200 words), Long (300-500 words)
- **Source Attribution**: Direct references to source documents with metadata
- **Professional UI**: Clean, responsive Streamlit interface with dark/light mode support
- **Real-time Processing**: Fast retrieval and processing with detailed logging

## Architecture

The system consists of several key components:

- **Vector Database**: ChromaDB for semantic document storage and retrieval
- **Embedding Model**: SentenceTransformers for text vectorization
- **Language Model**: OpenAI GPT for intelligent summarization
- **Web Interface**: Streamlit application for user interaction
- **Document Processing**: Automated PDF text extraction and chunking

## Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Sufficient CPU resources (for text extraction and processing)
- At least 4GB RAM recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/BlazingPh0enix/LLM-DSS.git
cd LLM-DSS
```

### 2. Set Up Python Environment

Using venv:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

Using conda:
```bash
conda create -n llm-dss python=3.10
conda activate llm-dss
```

### 3. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using uv (recommended for faster installation):
```bash
pip install uv
uv pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Data Setup and Preprocessing

After cloning the repository, you must run the following scripts in order to set up the vector database:

### 1. Download Source Documents

```bash
python scripts/data_load.py
```

This script will:
- Create necessary directory structure
- Download required NLTK and spaCy models
- Download physics textbooks from OpenStax

### 2. Extract Text from PDFs

```bash
python scripts/extract_text.py
```

**Note**: This process is CPU-intensive and may take significant time depending on your system specifications. The script processes multiple large PDF files and extracts text content.

### 3. Process and Vectorize Documents

```bash
python scripts/data_preprocess.py
```

This script will:
- Clean and chunk the extracted text
- Generate embeddings for all text chunks
- Store vectors in ChromaDB database
- Create searchable indices

## Running the Application

### Start the Streamlit Server

```bash
streamlit run src/app.py
```

The application will be available at:
- Local URL: `http://localhost:8501`
- Network URL: `http://[your-ip]:8501`

### Using the Application

1. **Enter Query**: Type your physics-related question in the text input
2. **Select Length**: Choose desired summary length (Short/Medium/Long)
3. **Submit**: Click "Search & Analyze" to process your query
4. **Review Results**: Examine the structured summary and source references

## Project Structure

```
LLM-DSS/
├── src/
│   ├── app.py              # Streamlit web application
│   ├── main.py             # Core search and summarization logic
│   └── evaluation.py       # System evaluation metrics
├── scripts/
│   ├── data_load.py        # Download documents and models
│   ├── extract_text.py     # PDF text extraction
│   └── data_preprocess.py  # Text processing and vectorization
├── data/
│   ├── physics_corpus/     # Raw PDF documents
│   ├── processed_text/     # Extracted text files
│   └── physics_vectordb/   # Vector database storage
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Configuration

### Summary Length Options

- **Short**: Approximately 100 words
- **Medium**: 150-200 words (default)
- **Long**: 300-500 words

### Customization

The system can be customized by modifying:

- **Database Path**: Change vector database location in `src/main.py`
- **Model Selection**: Update embedding model in `scripts/data_preprocess.py`
- **Chunk Size**: Adjust text chunking parameters in `scripts/data_preprocess.py`
- **UI Styling**: Modify CSS in `src/app.py`

## API Dependencies

This system requires:

- **OpenAI API**: For text summarization (GPT model)
- **Internet Connection**: For initial model downloads and API calls

## Performance Considerations

- **Initial Setup**: First-time setup requires downloading ~2GB of textbooks and models
- **Processing Time**: Text extraction and vectorization can take 30-60 minutes
- **Query Response**: Typical query processing takes 15-20 seconds
- **Memory Usage**: Approximately 2-4GB RAM during operation, mainly for data extraction and preprocessing.

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Verify API key in `.env` file
2. **Module Not Found**: Ensure all dependencies are installed
3. **Database Empty**: Complete all preprocessing steps in order
4. **Slow Performance**: Increase system RAM or reduce chunk size

### Logging

The system provides detailed logging output in the terminal when running. Monitor the console for:
- Database connection status
- Document processing progress
- Search and summarization timing
- Error messages and warnings

## Development

### Running Tests

```bash
python -m src.main
```

### Code Structure

- `ContentSummarizer`: Handles AI-powered text summarization
- `SearchSystem`: Manages vector search and query processing
- `Evaluator`: Provides system performance metrics

## Contributing

For bug reports or feature requests, please create an issue in the repository.

## Support

For technical support or questions about setup, refer to the troubleshooting section or check the terminal output for detailed error messages.