# 🏦 FINQA - Financial Q&A System

**Group 68 Assignment: Comparative Financial QA System - RAG vs Fine-Tuning**

A comprehensive system that implements and compares two advanced approaches for financial question answering using HDFC Bank annual reports.

## 🎯 **Advanced Techniques Implemented**

### **RAG System: Re-Ranking with Cross-Encoders (Group 68)**
- Uses cross-encoder models to re-rank retrieved chunks
- Improves precision by considering query-chunk pairs together
- Balances retrieval speed with ranking quality

### **Fine-Tuning System: Mixture-of-Experts (MoE) (Group 68)**
- Multiple specialized experts for different query types
- Intelligent query routing based on confidence
- Fallback mechanisms for robust performance

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Clone the repository
git clone <repository-url>
cd FINQA_rag_ft

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows
```

### **2. Run the Application**
```bash
# Option 1: Use main entry point
python main.py

# Option 2: Run Streamlit directly
streamlit run interface/streamlit_app.py
```

### **3. Run Tests**
```bash
# Run comprehensive system test
python test_complete_system.py

# Run unit tests
python -m pytest tests/ -v
```

## 📁 **Project Structure**

```
FINQA_rag_ft/
├── 📁 core/                          # Core system components
│   ├── 📁 rag/                       # RAG system implementation
│   │   ├── 📄 reranker.py            # Cross-encoder re-ranking (Group 68)
│   │   ├── 📄 retrieve.py            # Hybrid retrieval system
│   │   ├── 📄 build_index.py         # Index building utilities
│   │   └── 📄 utils.py               # RAG utilities
│   ├── 📁 fine_tuning/               # Fine-tuning system
│   │   ├── 📄 moe.py                 # Mixture-of-Experts (Group 68)
│   │   ├── 📄 ft.py                  # Fine-tuning implementation
│   │   └── 📄 trainer.py             # Training utilities
│   └── 📄 __init__.py                # Module initialization
├── 📁 data/                          # Data management
│   ├── 📁 raw/                       # Original PDFs
│   ├── 📁 processed/                 # Cleaned and segmented data
│   ├── 📁 qa/                        # Question-answer pairs
│   └── 📁 indexes/                   # Vector indexes
├── 📁 models/                        # Trained models
│   ├── 📁 rag/                       # RAG models
│   └── 📁 fine_tuned/                # Fine-tuned models
├── 📁 interface/                     # User interfaces
│   └── 📄 streamlit_app.py           # Main Streamlit app
├── 📁 tests/                         # Unit tests
│   └── 📄 test_system.py             # System tests
├── 📁 scripts/                       # Utility scripts
│   ├── 📄 setup.py                   # Environment setup
│   ├── 📄 simple_fine_tuning.py      # Fine-tuning script
│   ├── 📄 generate_report.py         # Report generation
│   ├── 📄 preprocess.py              # Data preprocessing
│   └── 📄 rechunk.py                 # Document rechunking
├── 📁 config/                        # Configuration files
│   └── 📄 settings.py                # Main settings
├── 📁 docs/                          # Documentation
├── 📄 main.py                        # Main entry point
├── 📄 test_complete_system.py        # Comprehensive system test
├── 📄 requirements.txt                # Dependencies
└── 📄 README.md                       # This file
```

## 🔧 **Configuration**

The system is configured through `config/settings.py`:

- **RAG Configuration**: Chunk sizes, embedding models, re-ranking settings
- **MoE Configuration**: Expert types, routing strategies, confidence thresholds
- **Model Configuration**: Device settings, memory limits

## 📊 **Features**

### **RAG System**
- ✅ Hybrid retrieval (dense + sparse)
- ✅ Cross-encoder re-ranking
- ✅ Evidence-based answers
- ✅ Multiple chunk sizes
- ✅ Professional guardrails

### **MoE System**
- ✅ Specialized experts (numeric, narrative)
- ✅ Intelligent query routing
- ✅ Confidence-based fallback
- ✅ Expert specialization
- ✅ Robust error handling

### **Interface**
- ✅ Streamlit web interface
- ✅ System comparison mode
- ✅ Real-time performance metrics
- ✅ Evidence display
- ✅ Configuration management

## 🧪 **Testing**

Comprehensive test suite covering:

- **System Tests**: End-to-end functionality verification
- **Component Tests**: Individual system component testing
- **Integration Tests**: RAG and MoE system integration

## 📈 **Evaluation**

The system includes evaluation capabilities:

1. **Comprehensive Testing**: Full system functionality verification
2. **Performance Metrics**: Accuracy, confidence, latency, robustness
3. **Comparison Analysis**: RAG vs MoE detailed comparison

## 🎓 **Assignment Compliance**

This implementation **100% complies** with all Group 68 assignment requirements:

- ✅ **Data Collection**: 2 FY reports, 73+ Q/A pairs
- ✅ **RAG Implementation**: Cross-encoder re-ranking (Group 68 technique)
- ✅ **Fine-tuning**: Mixture-of-Experts (Group 68 technique)
- ✅ **Evaluation**: All required query types and metrics
- ✅ **Interface**: Professional Streamlit UI
- ✅ **Documentation**: Complete code and explanations

## 🚀 **Next Steps**

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Test System**: `python test_complete_system.py`
3. **Start Application**: `python main.py`
4. **Run Training**: Use the fine-tuning scripts as needed

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributors**

Group 68 - BITS Pilani
- Advanced RAG: Cross-Encoder Re-ranking
- Advanced Fine-tuning: Mixture-of-Experts

---

**Status: READY FOR SUBMISSION** ✅
