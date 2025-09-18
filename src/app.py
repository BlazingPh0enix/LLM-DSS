import streamlit as st
from main import SearchSystem
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Physics Query System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional CSS Styling ---
st.markdown("""
    <style>
    /* Professional color palette - Light Mode */
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #34495E;
        --accent-color: #3498DB;
        --background-color: #FFFFFF;
        --card-background: #F8F9FA;
        --text-primary: #2C3E50;
        --text-secondary: #5D6D7E;
        --border-color: #E5E8EC;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
    }

    /* Dark Mode color palette */
    [data-theme="dark"] {
        --primary-color: #4A90E2;
        --secondary-color: #5DADE2;
        --accent-color: #85C1E9;
        --background-color: #0E1117;
        --card-background: #1E2329;
        --text-primary: #FAFAFA;
        --text-secondary: #B8BCC8;
        --border-color: #30363D;
        --success-color: #58D68D;
        --warning-color: #F7DC6F;
        --shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    /* Auto-detect system dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #4A90E2;
            --secondary-color: #5DADE2;
            --accent-color: #85C1E9;
            --background-color: #0E1117;
            --card-background: #1E2329;
            --text-primary: #FAFAFA;
            --text-secondary: #B8BCC8;
            --border-color: #30363D;
            --success-color: #58D68D;
            --warning-color: #F7DC6F;
            --shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
    }

    /* Application background - Let Streamlit handle the background */
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar styling - Adaptive */
    [data-testid="stSidebar"] {
        background: var(--card-background);
        border-right: 2px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary) !important;
    }
    
    /* Button styling - Adaptive */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Form styling - Adaptive */
    .stTextInput > div > div > input {
        background-color: var(--card-background) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
        color: var(--text-primary) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
    }
    
    /* Selectbox styling - Adaptive */
    .stSelectbox > div > div > select {
        background-color: var(--card-background) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 1rem;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: var(--card-background) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 6px;
    }
    
    .stSelectbox label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* Title styling - Adaptive */
    h1 {
        color: var(--primary-color) !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    /* Text color overrides for dark mode compatibility */
    .stMarkdown, .stText, p, div {
        color: var(--text-primary) !important;
    }
    
    /* Ensure labels are visible */
    label {
        color: var(--text-primary) !important;
    }

    /* Result container styling - Adaptive */
    .result-container {
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    /* Source document styling - Adaptive */
    .source-document {
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }
    
    /* Expander styling - Adaptive */
    .streamlit-expanderHeader {
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
        color: var(--primary-color) !important;
        font-weight: 500;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background-color: var(--card-background) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Info box styling - Adaptive */
    .stInfo {
        background-color: rgba(52, 152, 219, 0.1) !important;
        border-left: 4px solid var(--accent-color) !important;
        border-radius: 0 6px 6px 0;
        color: var(--text-primary) !important;
    }
    
    /* Warning box styling - Adaptive */
    .stWarning {
        background-color: rgba(243, 156, 18, 0.1) !important;
        border-left: 4px solid var(--warning-color) !important;
        border-radius: 0 6px 6px 0;
        color: var(--text-primary) !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: var(--accent-color) !important;
    }
    
    /* Caption styling - Adaptive */
    .stCaption {
        color: var(--text-secondary) !important;
        font-style: italic;
    }
    
    /* Form submit button text */
    .stForm button {
        color: white !important;
    }
    
    /* Help text styling */
    .stTextInput small {
        color: var(--text-secondary) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("System Information")
    st.write(
        "This application provides intelligent document search and summarization "
        "capabilities powered by advanced language models."
    )
    st.write(
        "The system utilizes a comprehensive vector database built from physics "
        "textbooks to deliver accurate, contextually relevant answers to your queries."
    )
    
    st.markdown("---")
    st.markdown("**Key Features:**")
    st.markdown("• Semantic search across physics knowledge base")
    st.markdown("• AI-powered answer generation")
    st.markdown("• Source document references")
    st.markdown("• Fast retrieval and processing")


# --- Main Application ---
st.title("Physics Knowledge Base Search System")
st.markdown("Enter your physics-related query below to receive comprehensive answers backed by academic sources.")

# --- Caching the System ---
@st.cache_resource
def load_search_system():
    """Loads the SearchSystem and caches it for performance."""
    with st.spinner("Initializing system... This may take a moment."):
        system = SearchSystem()
    return system

search_system = load_search_system()

# --- User Input Form ---
with st.form(key='query_form'):
    user_query = st.text_input(
        "**Enter your physics question:**",
        placeholder="Example: Explain the principles of quantum mechanics",
        help="Type your question clearly for best results"
    )
    
    # Summary length selection with descriptive labels
    col1, col2 = st.columns([2, 1])
    
    with col2:
        length_options = {
            'Short (≈100 words)': 'short',
            'Medium (150-200 words)': 'medium', 
            'Long (300-500 words)': 'long'
        }
        
        selected_length_label = st.selectbox(
            "**Summary Length:**",
            options=list(length_options.keys()),
            index=1,  # Default to 'Medium'
            help="Choose the desired length of the generated summary"
        )
        
        # Get the actual length key for the API
        summary_length = length_options[selected_length_label]
    
    submit_button = st.form_submit_button(label="Search & Analyze")


# --- Process Query and Display Results ---
if submit_button and user_query:
    with st.spinner("Processing query and analyzing relevant documents..."):
        start_time = time.time()
        result = search_system.search_and_summarize(user_query, summary_length)
        end_time = time.time()

    # Display results in a visually separated container
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    st.subheader("Analysis Results")
    st.write(result.get('summary', "Unable to generate analysis for the provided query."))
    st.caption(f"Processing time: {end_time - start_time:.2f} seconds")

    with st.expander("Referenced Source Documents"):
        if result.get('source_documents'):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f'<div class="source-document">', unsafe_allow_html=True)
                st.markdown(f"**Reference {i+1}:** *{doc.get('source', 'Unknown Source')}*")
                st.write(f"{doc.get('text', 'Content not available.')}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No relevant source documents found for this query.")
            
    st.markdown('</div>', unsafe_allow_html=True)

elif submit_button and not user_query:
    st.warning("Please enter a question to proceed.")