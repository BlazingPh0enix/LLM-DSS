import streamlit as st
import os
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from main import SearchSystem, SystemEvaluator

# Page configuration
st.set_page_config(
    page_title="Physics Search System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .equation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_search_system():
    """Load and cache the search system"""
    return SearchSystem(
        data_dir="./physics_textbooks"
    )

@st.cache_data
def setup_system_cache():
    """Setup system with caching"""
    system = load_search_system()
    if not system.documents_processed:
        system.setup_system()
    return True

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Physics Textbook Search System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Search through OpenStax Physics textbooks with AI-powered semantic search and summarization.
    The system preserves mathematical equations and provides contextual physics explanations.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System setup
        st.subheader("System Status")
        if st.button("🚀 Initialize System"):
            with st.spinner("Processing physics textbooks..."):
                try:
                    setup_system_cache()
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Search parameters
        st.subheader("Search Parameters")
        n_results = st.slider("Number of results", 1, 10, 5)
        summary_length = st.selectbox(
            "Summary length", 
            ["short", "medium", "long"],
            index=1
        )
        
        # File upload
        st.subheader("📚 Add New Textbooks")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files ready to process")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search interface
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        st.subheader("🔍 Search Physics Concepts")
        
        # Query input
        query = st.text_input(
            "Enter your physics question:",
            placeholder="e.g., What is Newton's second law of motion?",
            help="Ask about any physics concept, equation, or principle"
        )
        
        # Quick query suggestions
        st.write("💡 **Quick suggestions:**")
        suggestions = [
            "What is electromagnetic induction?",
            "Explain energy conservation",
            "How do electric fields work?",
            "What is quantum mechanics?",
            "Describe wave-particle duality"
        ]
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(f"💭 {suggestion.split('?')[0]}?", key=f"suggestion_{i}"):
                query = suggestion
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Search execution
        if query:
            try:
                system = load_search_system()
                
                with st.spinner("🔍 Searching through physics textbooks..."):
                    result = system.search_and_summarize(
                        query=query,
                        n_results=n_results,
                        summary_length=summary_length
                    )
                
                # Display results
                st.markdown("---")
                st.subheader("📖 Summary")
                
                # Summary card
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.write(result['summary'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Source documents
                st.subheader("📚 Source Documents")
                
                for i, doc in enumerate(result['source_documents']):
                    with st.expander(f"📄 Source {i+1}: {doc['metadata']['source_file']}"):
                        st.write(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                        
                        # Show equations if present
                        if doc['metadata'].get('equations'):
                            st.write("**Equations found:**")
                            for eq in doc['metadata']['equations']:
                                st.markdown(f'<div class="equation-box">{eq}</div>', 
                                          unsafe_allow_html=True)
                        
                        # Metadata
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.metric("Chunk ID", doc['metadata']['chunk_id'])
                        with col_meta2:
                            st.metric("Has Equations", 
                                    "✅ Yes" if doc['metadata']['has_equations'] else "❌ No")
            
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                st.info("💡 Make sure the system is initialized and PDF files are available.")
    
    with col2:
        # Statistics and information
        st.subheader("📊 System Statistics")
        
        try:
            system = load_search_system()
            if system.vector_db.collection:
                # Get collection stats
                collection_info = system.vector_db.collection.count()
                
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📚 Documents Processed", collection_info)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Recent queries (mock data for demo)
                st.subheader("🕒 Recent Popular Queries")
                recent_queries = [
                    "Newton's laws",
                    "Electromagnetic waves", 
                    "Quantum mechanics",
                    "Energy conservation",
                    "Electric fields"
                ]
                
                for query in recent_queries:
                    st.write(f"• {query}")
                
        except:
            st.info("Initialize system to see statistics")
        
        # System information
        st.subheader("ℹ️ About")
        st.write("""
        **Features:**
        - 🔍 Semantic search through physics textbooks
        - 🧮 Equation-preserving text extraction  
        - 🤖 AI-powered summarization
        - 📊 Quality evaluation metrics
        - 🎯 Physics-specific optimizations
        
        **Supported Content:**
        - OpenStax Physics textbooks
        - Mathematical equations & formulas
        - Physics principles & concepts
        - Problem-solving examples
        """)

# Additional pages
def evaluation_page():
    """Evaluation and analytics page"""
    st.title("📊 System Evaluation")
    
    try:
        system = load_search_system()
        evaluator = SystemEvaluator(system)
        
        tab1, tab2, tab3 = st.tabs(["Search Accuracy", "Summary Quality", "Performance"])
        
        with tab1:
            st.subheader("🎯 Search Accuracy Evaluation")
            
            if st.button("Run Search Accuracy Test"):
                with st.spinner("Running evaluation..."):
                    test_queries = evaluator.create_test_queries(20)
                    results = evaluator.evaluate_search_accuracy(test_queries)
                    
                    # Display overall accuracy
                    st.metric("Overall Search Accuracy", 
                            f"{results['overall_accuracy']:.3f}")
                    
                    # Detailed results
                    df = pd.DataFrame(results['detailed_results'])
                    st.dataframe(df)
                    
                    # Visualization
                    fig = px.histogram(df, x='avg_relevance', 
                                     title='Distribution of Relevance Scores')
                    st.plotly_chart(fig)
        
        with tab2:
            st.subheader("📝 Summary Quality Analysis")
            st.info("Upload reference summaries to evaluate quality using ROUGE scores")
            
            # File upload for reference summaries
            ref_file = st.file_uploader("Upload reference summaries (JSON)", type=['json'])
            if ref_file:
                # Process reference summaries and evaluate
                st.success("Reference summaries loaded!")
        
        with tab3:
            st.subheader("⚡ Performance Metrics")
            
            # Mock performance data
            performance_data = {
                'Metric': ['Search Time', 'Summarization Time', 'Total Processing', 'Memory Usage'],
                'Value': [0.8, 2.3, 3.1, 256],
                'Unit': ['seconds', 'seconds', 'seconds', 'MB']
            }
            
            df_perf = pd.DataFrame(performance_data)
            st.dataframe(df_perf)
            
            # Performance chart
            fig = go.Figure(data=[
                go.Bar(x=df_perf['Metric'], y=df_perf['Value'])
            ])
            fig.update_layout(title='System Performance Metrics')
            st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")

def admin_page():
    """Admin panel for system management"""
    st.title("⚙️ System Administration")
    
    tab1, tab2, tab3 = st.tabs(["Document Management", "System Config", "Logs"])
    
    with tab1:
        st.subheader("📚 Document Management")
        
        # Document processing status
        pdf_dir = Path("./physics_textbooks")
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            
            st.write(f"📁 Found {len(pdf_files)} PDF files:")
            for pdf in pdf_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {pdf.name}")
                with col2:
                    st.write(f"{pdf.stat().st_size / (1024*1024):.1f} MB")
                with col3:
                    if st.button("🔄 Reprocess", key=f"reprocess_{pdf.name}"):
                        st.info(f"Reprocessing {pdf.name}...")
        
        # Bulk operations
        st.subheader("🔧 Bulk Operations")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Vector Database"):
                st.warning("This will delete all processed documents!")
        with col2:
            if st.button("🔄 Rebuild Database"):
                st.info("Rebuilding entire database...")
    
    with tab2:
        st.subheader("⚙️ System Configuration")
        
        # Configuration options
        st.write("**Model Settings:**")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "sentence-transformers/all-roberta-large-v1"]
        )
        
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        overlap_size = st.slider("Chunk Overlap", 0, 200, 50)
        
        st.write("**API Settings:**")
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        )
        
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved!")
    
    with tab3:
        st.subheader("📋 System Logs")
        
        # Mock log data
        log_data = [
            {"timestamp": "2024-01-15 10:30:15", "level": "INFO", "message": "System initialized"},
            {"timestamp": "2024-01-15 10:31:22", "level": "INFO", "message": "PDF processed: university_physics_vol1.pdf"},
            {"timestamp": "2024-01-15 10:32:45", "level": "WARNING", "message": "Equation extraction partial for page 45"},
            {"timestamp": "2024-01-15 10:35:12", "level": "INFO", "message": "Search query processed: Newton's laws"}
        ]
        
        df_logs = pd.DataFrame(log_data)
        st.dataframe(df_logs, use_container_width=True)
        
        # Log level filter
        log_level = st.selectbox("Filter by log level", ["ALL", "INFO", "WARNING", "ERROR"])

# Navigation
def main_app():
    """Main application with navigation"""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.selectbox("Select Page", [
            "🔍 Search",
            "📊 Evaluation", 
            "⚙️ Admin Panel"
        ])
    
    # Route to appropriate page
    if page == "🔍 Search":
        main()
    elif page == "📊 Evaluation":
        evaluation_page()
    elif page == "⚙️ Admin Panel":
        admin_page()

if __name__ == "__main__":
    main_app()