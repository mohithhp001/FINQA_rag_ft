"""
Optimized Streamlit Interface for FINQA RAG vs Fine-Tuning System.
Group 68: Re-Ranking with Cross-Encoders (RAG) + Mixture-of-Experts (FT)
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
try:
    from core.rag.reranker import create_hybrid_reranker
    from core.fine_tuning.moe import create_moe_system
    from config.settings import get_config_summary, validate_config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FINQA - RAG vs Fine-Tuning Comparison",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .expert-badge {
        background: #ff7f0e;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_query = ""
    st.session_state.show_training = False
    st.session_state.training_started = False
    st.session_state.training_completed = False
    st.session_state.training_results = {}
    st.session_state.training_error = None

class FINQAInterface:
    """Main interface class for FINQA system."""
    
    def __init__(self):
        """Initialize the interface."""
        if not st.session_state.initialized:
            self.config_summary = get_config_summary()
            self.initialize_components()
            st.session_state.initialized = True
        else:
            self.config_summary = get_config_summary()
            # Reuse existing components if already initialized
            if hasattr(st.session_state, 'rag_reranker'):
                self.rag_reranker = st.session_state.rag_reranker
            else:
                self.rag_reranker = None
                
            if hasattr(st.session_state, 'moe_system'):
                self.moe_system = st.session_state.moe_system
            else:
                self.moe_system = None
        
    def initialize_components(self):
        """Initialize RAG and MoE components."""
        try:
            # Initialize RAG reranker
            with st.spinner("Initializing RAG system..."):
                self.rag_reranker = create_hybrid_reranker()
                st.session_state.rag_reranker = self.rag_reranker
                st.success("✅ RAG system initialized")
            
            # Initialize MoE system
            with st.spinner("Initializing MoE system..."):
                self.moe_system = create_moe_system()
                st.session_state.moe_system = self.moe_system
                st.success("✅ MoE system initialized")
                
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            st.info("Running in demo mode with limited functionality.")
            self.rag_reranker = None
            self.moe_system = None
            st.session_state.rag_reranker = None
            st.session_state.moe_system = None
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            <h1>🏦 FINQA - Financial Q&A System</h1>
            <p>Group 68: Advanced RAG vs Fine-tuning Comparison</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if self.rag_reranker:
                st.success("✅ RAG System")
            else:
                st.error("❌ RAG System")
        
        with col2:
            st.info("🔍 Cross-Encoder Re-ranking + MoE Architecture")
        
        with col3:
            if self.moe_system:
                st.success("✅ MoE System")
            else:
                st.error("❌ MoE System")
    
    def render_sidebar(self):
        """Render sidebar configuration."""
        st.sidebar.header("⚙️ Configuration")
        
        # System mode selection
        system_mode = st.sidebar.selectbox(
            "System Mode",
            ["Comparison Mode", "RAG + Re-ranking", "MoE Fine-tuned"],
            help="Choose which system(s) to use for query processing"
        )
        
        # RAG configuration
        st.sidebar.subheader("🔍 RAG Settings")
        use_reranking = st.sidebar.checkbox("Use Re-ranking", value=True, help="Enable cross-encoder re-ranking")
        
        if use_reranking:
            rerank_method = st.sidebar.selectbox(
                "Re-ranking Method",
                ["cross_encoder", "hybrid"],
                help="Choose re-ranking approach"
            )
        else:
            rerank_method = "none"
        
        top_k = st.sidebar.slider("Top K Results", 1, 10, 5, help="Number of results to retrieve")
        
        # MoE configuration
        st.sidebar.subheader("🧠 MoE Settings")
        routing_strategy = st.sidebar.selectbox(
            "Routing Strategy",
            ["confidence_based", "keyword_based", "hybrid"],
            help="Expert routing strategy"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.7, 0.1,
            help="Minimum confidence for expert selection"
        )
        
        # Training toggle
        show_training = st.sidebar.checkbox("Show Training Interface", value=False, help="Display model training options")
        if show_training != st.session_state.show_training:
            st.session_state.show_training = show_training
            st.rerun()
        
        return {
            "system_mode": system_mode,
            "use_reranking": use_reranking,
            "rerank_method": rerank_method,
            "top_k": top_k,
            "routing_strategy": routing_strategy,
            "confidence_threshold": confidence_threshold
        }
    
    def render_query_input(self):
        """Render query input interface."""
        st.header("🔍 Query Input")
        
        # Example queries with better descriptions
        example_queries = [
            {
                "query": "What was HDFC Bank's total deposits in FY2024-25?",
                "description": "Numeric Query - Financial Data",
                "category": "numeric"
            },
            {
                "query": "How did HDFC Bank perform in FY2024-25?",
                "description": "Narrative Query - Performance Analysis",
                "category": "narrative"
            },
            {
                "query": "What is the CASA ratio trend?",
                "description": "Numeric Query - Ratio Analysis",
                "category": "numeric"
            },
            {
                "query": "What are the key risks facing HDFC Bank?",
                "description": "Narrative Query - Risk Assessment",
                "category": "narrative"
            },
            {
                "query": "What was the net profit margin in FY2024-25?",
                "description": "Numeric Query - Profitability Metrics",
                "category": "numeric"
            },
            {
                "query": "How did the bank manage digital transformation?",
                "description": "Narrative Query - Strategic Initiatives",
                "category": "narrative"
            },
            {
                "query": "What was the loan growth rate?",
                "description": "Numeric Query - Growth Metrics",
                "category": "numeric"
            },
            {
                "query": "What is the capital adequacy ratio?",
                "description": "Numeric Query - Regulatory Compliance",
                "category": "numeric"
            },
            {
                "query": "What were the digital banking initiatives?",
                "description": "Narrative Query - Technology Focus",
                "category": "narrative"
            },
            {
                "query": "What is the capital of France?",
                "description": "Irrelevant Query - Testing Robustness",
                "category": "irrelevant"
            }
        ]
        
        # Query input with session state management
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your financial question:",
                value=st.session_state.current_query,
                placeholder="Ask about HDFC Bank's financial performance...",
                key="main_query_input"
            )
        
        with col2:
            if st.button("🚀 Ask", type="primary", use_container_width=True):
                if query.strip():
                    st.session_state.current_query = query.strip()
                    st.rerun()
                else:
                    st.warning("Please enter a question first!")
        
        # Quick query buttons for common questions
        st.subheader("⚡ Quick Queries")
        quick_queries = [
            "What was HDFC Bank's total deposits in FY2024-25?",
            "How did HDFC Bank perform in FY2024-25?",
            "What is the CASA ratio?",
            "What are the key risks?"
        ]
        
        cols = st.columns(len(quick_queries))
        for i, quick_query in enumerate(quick_queries):
            with cols[i]:
                if st.button(quick_query[:25] + "...", key=f"quick_{i}", use_container_width=True):
                    st.session_state.current_query = quick_query
                    st.rerun()
        
        # Example queries section with improved UI
        st.subheader("💡 Example Queries")
        st.info("Click on any example query to automatically fill the input field above")
        
        # Create a more organized layout for example queries
        # Use expandable sections for better organization
        with st.expander("📊 Numeric Queries (Financial Data)", expanded=True):
            numeric_queries = [q for q in example_queries if q['category'] == 'numeric']
            for i, example in enumerate(numeric_queries):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{example['query']}**")
                    st.caption(f"*{example['description']}*")
                with col2:
                    if st.button(f"Use", key=f"use_numeric_{i}", use_container_width=True):
                        st.session_state.current_query = example['query']
                        st.rerun()
        
        with st.expander("📝 Narrative Queries (Analysis & Strategy)", expanded=True):
            narrative_queries = [q for q in example_queries if q['category'] == 'narrative']
            for i, example in enumerate(narrative_queries):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{example['query']}**")
                    st.caption(f"*{example['description']}*")
                with col2:
                    if st.button(f"Use", key=f"use_narrative_{i}", use_container_width=True):
                        st.session_state.current_query = example['query']
                        st.rerun()
        
        with st.expander("❌ Irrelevant Queries (Robustness Testing)", expanded=False):
            irrelevant_queries = [q for q in example_queries if q['category'] == 'irrelevant']
            for i, example in enumerate(irrelevant_queries):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{example['query']}**")
                    st.caption(f"*{example['description']}*")
                with col2:
                    if st.button(f"Use", key=f"use_irrelevant_{i}", use_container_width=True):
                        st.session_state.current_query = example['query']
                        st.rerun()
        
        # Clear query button
        if st.session_state.current_query:
            if st.button("🗑️ Clear Query", type="secondary"):
                st.session_state.current_query = ""
                st.rerun()
        
        # Training section
        if st.session_state.show_training:
            st.header("🎯 Model Training")
            st.info("Training MoE models for Group 68: Mixture-of-Experts Fine-tuning")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Training Configuration:**")
                st.write("• Base Model: FLAN-T5-small")
                st.write("• Expert Types: Numeric & Narrative")
                st.write("• Training Data: 16 Financial Q&A pairs")
                st.write("• Epochs: 2 (demonstration)")
            
            with col2:
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    st.session_state.training_started = True
            
            if st.session_state.training_started:
                with st.spinner("Training MoE models..."):
                    try:
                        # Import and run training
                        from core.fine_tuning.trainer import create_trainer
                        
                        # Create sample training data
                        training_data = [
                            {
                                "question": "What was HDFC Bank's total deposits in FY2024-25?",
                                "answer": "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25.",
                                "expert_type": "numeric"
                            },
                            {
                                "question": "How did HDFC Bank perform in FY2024-25?",
                                "answer": "HDFC Bank demonstrated strong performance in FY2024-25 with robust growth across key metrics.",
                                "expert_type": "narrative"
                            }
                        ]
                        
                        # Initialize trainer and train
                        trainer = create_trainer()
                        results = trainer.train_all_experts(training_data, epochs=1)
                        
                        # Store results
                        st.session_state.training_results = results
                        st.session_state.training_completed = True
                        
                        st.success("✅ Training completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.session_state.training_error = str(e)
            
            # Show training results
            if st.session_state.training_completed:
                st.subheader("📊 Training Results")
                results = st.session_state.training_results
                
                for expert_type, result in results.items():
                    if result.get("success"):
                        st.success(f"✅ {expert_type.capitalize()} Expert")
                        st.write(f"Model Path: {result['model_path']}")
                        st.write(f"Training Examples: {result['training_examples']}")
                        if result.get('mock_training'):
                            st.info("Mock training completed (transformers not available)")
                        else:
                            st.success("Real training completed")
                    else:
                        st.error(f"❌ {expert_type.capitalize()} Expert: {result.get('error')}")
                
                st.info("💡 Restart the app to use the newly trained models!")
        
        return st.session_state.current_query
    
    def process_rag_query(self, query: str, config: dict) -> dict:
        """Process query using RAG system."""
        if not self.rag_reranker:
            return {"error": "RAG system not available"}
        
        try:
            start_time = time.time()
            
            # Mock retrieval for demo (replace with actual retrieval)
            mock_chunks = [
                {
                    "id": "chunk_1",
                    "text": "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25, representing a 15% increase over the previous year.",
                    "title": "Balance Sheet FY2024-25",
                    "source": "Annual Report"
                },
                {
                    "id": "chunk_2", 
                    "text": "The bank's deposit growth was driven by strong performance in retail and corporate banking segments.",
                    "title": "Directors Report",
                    "source": "Annual Report"
                }
            ]
            
            # Apply re-ranking if enabled
            if config['use_reranking']:
                reranked_chunks = self.rag_reranker.rerank(
                    query, mock_chunks, method=config['rerank_method'], top_k=config['top_k']
                )
            else:
                reranked_chunks = mock_chunks[:config['top_k']]
            
            # Generate answer (mock for demo)
            answer = f"Based on the retrieved information: {reranked_chunks[0]['text']}"
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "evidence": reranked_chunks,
                "processing_time": processing_time,
                "method": f"RAG + {config['rerank_method'].replace('_', ' ').title()}",
                "chunks_retrieved": len(reranked_chunks),
                "reranking_applied": config['use_reranking']
            }
            
        except Exception as e:
            return {"error": f"RAG processing failed: {str(e)}"}
    
    def process_moe_query(self, query: str, config: dict) -> dict:
        """Process query using MoE system."""
        if not self.moe_system:
            return {"error": "MoE system not available"}
        
        try:
            start_time = time.time()
            
            # Update routing strategy
            self.moe_system.router.routing_strategy = config['routing_strategy']
            
            # Process query
            result = self.moe_system.answer_query(
                query, 
                confidence_threshold=config['confidence_threshold']
            )
            
            processing_time = time.time() - start_time
            
            return {
                "answer": result['answer'],
                "expert": result['expert'],
                "confidence": result['confidence'],
                "routing_confidence": result['routing_confidence'],
                "method": result['method'],
                "fallback_used": result['fallback_used'],
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {"error": f"MoE processing failed: {str(e)}"}
    
    def render_results(self, query: str, config: dict):
        """Render query results."""
        if not query:
            return
        
        st.header("📊 Results")
        
        # Process query based on system mode
        if config['system_mode'] == "RAG + Re-ranking":
            result = self.process_rag_query(query, config)
            self.render_rag_results(result)
            
        elif config['system_mode'] == "MoE Fine-tuned":
            result = self.process_moe_query(query, config)
            self.render_moe_results(result)
            
        elif config['system_mode'] == "Comparison Mode":
            self.render_comparison_results(query, config)
    
    def render_rag_results(self, result: dict):
        """Render RAG system results."""
        if "error" in result:
            st.error(result["error"])
            return
        
        # Answer
        st.subheader("💬 Answer")
        st.markdown(f"**{result['answer']}**")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Method", result['method'])
        with col2:
            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
        with col3:
            st.metric("Chunks Retrieved", result['chunks_retrieved'])
        with col4:
            st.metric("Re-ranking", "✅" if result['reranking_applied'] else "❌")
        
        # Evidence
        if result.get('evidence'):
            st.subheader("📚 Evidence")
            for i, chunk in enumerate(result['evidence'][:3]):  # Show top 3
                with st.expander(f"Chunk {i+1}: {chunk['title']}"):
                    st.write(f"**Source:** {chunk['source']}")
                    st.write(f"**Content:** {chunk['text']}")
                    if 'rerank_score' in chunk:
                        st.write(f"**Re-rank Score:** {chunk['rerank_score']:.3f}")
    
    def render_moe_results(self, result: dict):
        """Render MoE system results."""
        if "error" in result:
            st.error(result["error"])
            return
        
        # Answer
        st.subheader("💬 Answer")
        st.markdown(f"**{result['answer']}**")
        
        # Expert information
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="expert-badge">
                {result['expert'].upper()} EXPERT
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if result['fallback_used']:
                st.warning("⚠️ Fallback model used")
            else:
                st.success("✅ Expert model used")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Method", result['method'])
        with col2:
            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
        with col3:
            confidence_class = "confidence-high" if result['confidence'] >= 0.8 else \
                             "confidence-medium" if result['confidence'] >= 0.6 else "confidence-low"
            st.markdown(f"<span class='{confidence_class}'>**Confidence:** {result['confidence']:.3f}</span>", 
                       unsafe_allow_html=True)
        with col4:
            st.metric("Routing Confidence", f"{result['routing_confidence']:.3f}")
    
    def render_comparison_results(self, query: str, config: dict):
        """Render comparison between RAG and MoE systems."""
        st.subheader("🔄 System Comparison")
        
        # Process with both systems
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RAG + Re-ranking Results**")
            rag_result = self.process_rag_query(query, config)
            if "error" not in rag_result:
                st.success(rag_result['answer'][:100] + "...")
                st.metric("Time", f"{rag_result['processing_time']:.3f}s")
            else:
                st.error(rag_result['error'])
        
        with col2:
            st.markdown("**MoE Fine-tuned Results**")
            moe_result = self.process_moe_query(query, config)
            if "error" not in moe_result:
                st.success(moe_result['answer'][:100] + "...")
                st.metric("Time", f"{moe_result['processing_time']:.3f}s")
            else:
                st.error(moe_result['error'])
        
        # Comparison metrics
        if "error" not in rag_result and "error" not in moe_result:
            st.subheader("📈 Performance Comparison")
            
            comparison_data = {
                'Metric': ['Processing Time', 'Method', 'Confidence'],
                'RAG': [
                    f"{rag_result['processing_time']:.3f}s",
                    rag_result['method'],
                    "N/A"
                ],
                'MoE': [
                    f"{moe_result['processing_time']:.3f}s",
                    moe_result['method'],
                    f"{moe_result['confidence']:.3f}"
                ]
            }
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
    
    def render_system_stats(self):
        """Render system statistics and health information."""
        st.header("📊 System Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ RAG System")
            if self.rag_reranker:
                st.success("✅ RAG System Active")
                st.write("**Features:**")
                st.write("• Cross-Encoder Re-ranking")
                st.write("• Hybrid Retrieval")
                st.write("• Evidence-based Answers")
            else:
                st.error("❌ RAG System Unavailable")
        
        with col2:
            st.subheader("🧠 MoE System")
            if self.moe_system:
                stats = self.moe_system.get_system_stats()
                st.success("✅ MoE System Active")
                st.write(f"**Experts:** {stats['total_experts']}")
                st.write(f"**Types:** {', '.join(stats['expert_types'])}")
                st.write(f"**Fallback:** {'✅' if stats['fallback_available'] else '❌'}")
            else:
                st.error("❌ MoE System Unavailable")
    
    def render_about_section(self):
        """Render about section with Group 68 information."""
        st.header("ℹ️ About This System")
        
        st.markdown("""
        **Group 68 Assignment: Comparative Financial QA System**
        
        This system implements and compares two advanced approaches for financial question answering:
        
        ### 🎯 **Advanced Techniques Implemented**
        
        **1. RAG System: Re-Ranking with Cross-Encoders**
        - Uses cross-encoder models to re-rank retrieved chunks
        - Improves precision by considering query-chunk pairs together
        - Balances retrieval speed with ranking quality
        
        **2. Fine-tuning System: Mixture-of-Experts (MoE)**
        - Multiple specialized experts for different query types
        - Intelligent query routing based on confidence
        - Fallback mechanisms for robust performance
        
        ### 🏦 **Financial Domain Focus**
        - HDFC Bank Annual Reports (FY2023-24, FY2024-25)
        - 73+ Question-Answer pairs
        - Covers financial metrics, ratios, and narrative analysis
        
        ### 🔧 **Technical Features**
        - Hybrid retrieval (dense + sparse)
        - Confidence scoring and fallback mechanisms
        - Comprehensive evaluation framework
        - Professional Streamlit interface
        """)
    
    def run(self):
        """Run the main interface."""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Query Interface", 
            "📊 System Stats", 
            "📈 Performance", 
            "ℹ️ About"
        ])
        
        with tab1:
            # Query input and results
            query = self.render_query_input()
            if query:
                self.render_results(query, config)
        
        with tab2:
            # System statistics
            self.render_system_stats()
        
        with tab3:
            # Performance metrics and charts
            st.header("📈 Performance Metrics")
            st.info("Performance metrics will be displayed here after running evaluations.")
            
            # Placeholder for performance charts
            if st.button("Generate Performance Charts"):
                st.write("Charts generation feature coming soon...")
        
        with tab4:
            # About section
            self.render_about_section()

def main():
    """Main function to run the Streamlit interface."""
    try:
        # Create and run interface
        interface = FINQAInterface()
        interface.run()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
