import streamlit as st
from main import SearchSystem
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Physics Query System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a refined look ---
st.markdown("""
    <style>
    /* Define a primary color variable */
    :root {
        --primary-color: #008080; /* Dark Teal */
        --background-color: #f4f6f8;
        --widget-background-color: #ffffff;
        --text-color: #31333F;
        --border-color: #e6e6e6;
    }

    /* General Styles */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background-color: var(--widget-background-color);
        border-right: 1px solid var(--border-color);
    }
    
    /* --- UNIFIED COLOR SCHEME --- */
    /* Style for the submit button */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #006666; /* Darker shade for hover */
        color: white;
    }
    
    /* Style for expander headers to match the button */
    .st-emotion-cache-1f1d6gn {
        color: var(--primary-color);
    }

    /* Result container styling */
    .result-container {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        background-color: var(--widget-background-color);
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }

    /* Source document styling */
    .source-document {
        border-top: 1px solid var(--border-color);
        padding-top: 15px;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("About the System")
    st.write(
        "This application is a Document Search and Summarization system "
        "powered by Large Language Models (LLMs)."
    )
    st.write(
        "It uses a vector database created from physics textbooks to find relevant "
        "information and generate a concise summary to answer your questions."
    )
    st.info("Project by Four Junctions")


# --- Main Application ---
st.title("AI-Powered Physics Search System")
st.markdown("Ask a question about a physics topic, and the system will search its knowledge base to provide a summarized answer.")

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
        "**Enter your question:**",
        placeholder="e.g., What is the significance of the Higgs Boson?",
    )
    submit_button = st.form_submit_button(label="Generate Answer")


# --- Process Query and Display Results ---
if submit_button and user_query:
    with st.spinner("Searching documents and generating summary..."):
        start_time = time.time()
        result = search_system.search_and_summarize(user_query)
        end_time = time.time()

    # Display results in a visually separated container
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    st.subheader("Generated Summary")
    st.write(result.get('summary', "No summary could be generated."))
    st.caption(f"Time taken: {end_time - start_time:.2f} seconds")

    with st.expander("View Retrieved Sources"):
        if result.get('source_documents'):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f'<div class="source-document">', unsafe_allow_html=True)
                st.markdown(f"**Source {i+1} from:** *{doc.get('source', 'Unknown')}*")
                st.info(f"_{doc.get('text', 'No text available.')}_")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No source documents were found for this query.")
            
    st.markdown('</div>', unsafe_allow_html=True)

elif submit_button and not user_query:
    st.warning("Please enter a question.")