#!/usr/bin/env python3
"""
FINQA - Financial Q&A System
Main entry point for the application.

Group 68: RAG vs Fine-tuning Comparison
Advanced Techniques: Cross-Encoder Re-ranking + Mixture-of-Experts
"""

import sys
import subprocess
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import get_config_summary, validate_config

def main():
    """Main entry point."""
    print("🏦 FINQA - Financial Q&A System")
    print("=" * 50)
    
    # Show configuration
    config = get_config_summary()
    print(f"Project Root: {config['project_root']}")
    print(f"RAG Reranker: {config['rag_config']['reranker_model']}")
    print(f"MoE Experts: {config['moe_config']['expert_types']}")
    
    # Validate system
    validation = config['validation']
    if validation['valid']:
        print("✅ System Valid - Ready to run!")
    else:
        print("❌ System Issues Found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
        print("\nPlease fix issues before running the application.")
        return 1
    
    print("\n🚀 Starting Streamlit interface...")
    print("Access the application at: http://localhost:8501")
    print("\nTo stop the application, press Ctrl+C in this terminal.")
    
    # Get the path to the Streamlit app
    streamlit_app_path = Path(__file__).parent / "interface" / "streamlit_app.py"
    
    # Run Streamlit using subprocess to avoid import issues
    try:
        # Set environment variable to avoid tokenizer parallelism warnings
        env = os.environ.copy()
        env['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--server.port", "8501",
            "--server.headless", "true"
        ], env=env, check=True)
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
        return 0
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
