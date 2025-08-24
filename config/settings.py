"""
Centralized configuration settings for FINQA RAG vs Fine-Tuning system.
Group 68: Re-Ranking with Cross-Encoders (RAG) + Mixture-of-Experts (FT)
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
INDEXES_DIR = PROJECT_ROOT / "indexes"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
QA_DATA_DIR = DATA_DIR / "qa"
INDEXES_DATA_DIR = DATA_DIR / "indexes"

# Model paths
RAG_MODELS_DIR = MODELS_DIR / "rag"
FINE_TUNED_MODELS_DIR = MODELS_DIR / "fine_tuned"

# Reports paths
PERFORMANCE_CHARTS_DIR = REPORTS_DIR / "performance_charts"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, INDEXES_DIR, 
                  RAW_DATA_DIR, PROCESSED_DATA_DIR, QA_DATA_DIR, INDEXES_DATA_DIR,
                  RAG_MODELS_DIR, FINE_TUNED_MODELS_DIR, PERFORMANCE_CHARTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# RAG Configuration
RAG_CONFIG = {
    "chunk_sizes": [100, 400],  # Word-based chunking
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Group 68: Cross-encoder
    "top_k": 12,
    "rerank_top_k": 8,
    "hybrid_weight": 0.7,  # Weight for dense vs sparse retrieval
    "max_context_length": 2048,
}

# Fine-tuning Configuration
FINE_TUNING_CONFIG = {
    "base_model": "google/flan-t5-small",
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "learning_rate": 5e-4,
    "batch_size": 8,
    "epochs": 3,
    "max_length": 512,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
}

# MoE Configuration (Group 68: Mixture-of-Experts)
MOE_CONFIG = {
    "num_experts": 2,
    "expert_types": ["numeric", "narrative"],
    "routing_strategy": "confidence_based",
    "fallback_threshold": 0.3,  # Lowered from 0.6 to make routing easier
    "expert_specialization": {
        "numeric": [
            "deposits", "revenue", "profit", "ratio", "percentage", "crore", "million", 
            "growth", "rate", "margin", "assets", "capital", "npa", "casa", "loan",
            "financial", "data", "numbers", "metrics", "statistics", "amount", "total",
            "what", "was", "is", "are", "how", "much", "many", "total", "net", "gross"
        ],
        "narrative": [
            "performance", "strategy", "initiatives", "transformation", "digital", 
            "risk", "management", "outlook", "factors", "drivers", "challenges",
            "opportunities", "plans", "focus", "approach", "capabilities", "services",
            "customer", "market", "industry", "trends", "analysis", "assessment",
            "how", "did", "manage", "handle", "approach", "plan", "focus", "strategy"
        ]
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "baseline_questions": 15,
    "official_query_types": {
        "high_confidence": 5,
        "low_confidence": 5,
        "irrelevant": 5
    },
    "financial_questions": 15,
    "confidence_thresholds": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    },
    "metrics": ["accuracy", "confidence", "latency", "robustness"]
}

# Model Configuration
MODEL_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    "torch_dtype": "float16" if os.environ.get("CUDA_VISIBLE_DEVICES") else "float32",
    "max_memory": "8GB" if os.environ.get("CUDA_VISIBLE_DEVICES") else "4GB",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "finqa.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Validation functions
def validate_config() -> Dict[str, Any]:
    """Validate configuration and return any issues."""
    issues = []
    warnings = []
    
    # Check required directories
    required_dirs = [DATA_DIR, MODELS_DIR, REPORTS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            issues.append(f"Required directory missing: {directory}")
    
    # Check model files (warnings, not errors)
    if not (RAG_MODELS_DIR / "reranker").exists():
        warnings.append("RAG reranker model not found (using mock models)")
    
    if not (FINE_TUNED_MODELS_DIR / "moe_numeric").exists():
        warnings.append("MoE numeric expert not found (using mock models)")
    
    if not (FINE_TUNED_MODELS_DIR / "moe_narrative").exists():
        warnings.append("MoE narrative expert not found (using mock models)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of the current configuration."""
    return {
        "project_root": str(PROJECT_ROOT),
        "rag_config": RAG_CONFIG,
        "fine_tuning_config": FINE_TUNING_CONFIG,
        "moe_config": MOE_CONFIG,
        "evaluation_config": EVALUATION_CONFIG,
        "model_config": MODEL_CONFIG,
        "validation": validate_config()
    }

if __name__ == "__main__":
    # Print configuration summary
    summary = get_config_summary()
    print("FINQA Configuration Summary:")
    print("=" * 50)
    print(f"Project Root: {summary['project_root']}")
    print(f"RAG Chunk Sizes: {summary['rag_config']['chunk_sizes']}")
    print(f"RAG Reranker: {summary['rag_config']['reranker_model']}")
    print(f"MoE Experts: {summary['moe_config']['expert_types']}")
    print(f"Validation: {'✓ Valid' if summary['validation']['valid'] else '✗ Issues Found'}")
    
    if not summary['validation']['valid']:
        print("\nIssues found:")
        for issue in summary['validation']['issues']:
            print(f"  - {issue}")
