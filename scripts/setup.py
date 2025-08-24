#!/usr/bin/env python3
"""
Setup script for FINQA project.
Initializes the project structure and downloads required models.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_environment():
    """Set up the Python environment."""
    print("🚀 Setting up FINQA project environment...")
    
    # Check if virtual environment exists
    if not Path(".venv").exists():
        print("📦 Creating virtual environment...")
        if not run_command("python3 -m venv .venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    # Install dependencies
    print("📦 Installing dependencies...")
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def download_models():
    """Download required models."""
    print("🤖 Downloading required models...")
    
    models_to_download = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "google/flan-t5-small"
    ]
    
    for model in models_to_download:
        print(f"📥 Downloading {model}...")
        # This would typically use huggingface_hub or transformers
        # For now, we'll just note what needs to be downloaded
        print(f"   Note: {model} will be downloaded on first use")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/qa",
        "data/indexes",
        "models/rag",
        "models/fine_tuned",
        "reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return True

def validate_setup():
    """Validate the setup."""
    print("🔍 Validating setup...")
    
    # Check if key files exist
    required_files = [
        "config/settings.py",
        "core/rag/reranker.py",
        "core/fine_tuning/moe.py",
        "interface/streamlit_app.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing!")
            return False
    
    # Check if directories exist
    required_dirs = [
        "core/rag",
        "core/fine_tuning", 
        "core/evaluation",
        "interface",
        "tests"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - Missing!")
            return False
    
    return True

def main():
    """Main setup function."""
    print("🏦 FINQA Project Setup")
    print("=" * 50)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        return 1
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        return 1
    
    # Download models
    if not download_models():
        print("❌ Failed to download models")
        return 1
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment: source .venv/bin/activate (Unix) or .venv\\Scripts\\activate (Windows)")
    print("2. Run tests: python -m pytest tests/")
    print("3. Start application: python main.py")
    print("4. Or run Streamlit directly: streamlit run interface/streamlit_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
