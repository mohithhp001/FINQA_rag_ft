#!/usr/bin/env python3
"""
Comprehensive test script for FINQA system.
Tests all components: RAG, MoE, Training, and Integration.
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_configuration():
    """Test configuration system."""
    print("🧪 Testing Configuration System...")
    try:
        from config.settings import get_config_summary
        config = get_config_summary()
        print("✅ Configuration loaded successfully")
        print(f"   Project Root: {config['project_root']}")
        print(f"   RAG Reranker: {config['rag_config']['reranker_model']}")
        print(f"   MoE Experts: {config['moe_config']['expert_types']}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system components."""
    print("\n🧪 Testing RAG System...")
    try:
        from core.rag.reranker import create_hybrid_reranker
        
        # Test reranker creation
        reranker = create_hybrid_reranker()
        print("✅ RAG Reranker created successfully")
        
        # Test reranking
        test_query = "What was HDFC Bank's total deposits in FY2024-25?"
        test_chunks = [
            {"id": "1", "text": "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25.", "title": "Balance Sheet", "source": "Annual Report"},
            {"id": "2", "text": "The bank's deposit growth was strong during the fiscal year.", "title": "Directors Report", "source": "Annual Report"}
        ]
        
        reranked = reranker.rerank(test_query, test_chunks, method="cross_encoder", top_k=2)
        print(f"✅ Reranking test passed: {len(reranked)} chunks reranked")
        
        return True
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_moe_system():
    """Test MoE system components."""
    print("\n🧪 Testing MoE System...")
    try:
        from core.fine_tuning.moe import create_moe_system
        
        # Test MoE system creation
        moe = create_moe_system()
        print("✅ MoE System created successfully")
        
        # Test query processing
        test_query = "What was HDFC Bank's total deposits in FY2024-25?"
        result = moe.answer_query(test_query, confidence_threshold=0.7)
        
        print(f"✅ MoE query processing test passed")
        print(f"   Answer: {result['answer'][:50]}...")
        print(f"   Expert: {result['expert']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ MoE system test failed: {e}")
        return False

def test_training_system():
    """Test training system."""
    print("\n🧪 Testing Training System...")
    try:
        from core.fine_tuning.trainer import create_trainer
        
        # Test trainer creation
        trainer = create_trainer()
        print("✅ Trainer created successfully")
        
        # Test training status
        status = trainer.get_training_status()
        print(f"✅ Training status retrieved: {status['trained_models']}/{status['total_models']} models")
        
        return True
    except Exception as e:
        print(f"❌ Training system test failed: {e}")
        return False

def test_streamlit_integration():
    """Test Streamlit app integration."""
    print("\n🧪 Testing Streamlit Integration...")
    try:
        # Test if we can import the Streamlit app
        from interface.streamlit_app import FINQAInterface
        
        # Test system creation
        system = FINQAInterface()
        print("✅ Streamlit app system created successfully")
        
        # Test configuration
        config = system.config_summary
        print(f"✅ Configuration accessible: {len(config)} config sections")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end functionality."""
    print("\n🧪 Testing End-to-End Functionality...")
    try:
        from core.fine_tuning.moe import create_moe_system
        from core.rag.reranker import create_hybrid_reranker
        
        # Test both systems
        moe = create_moe_system()
        rag = create_hybrid_reranker()
        
        # Test query
        query = "What was HDFC Bank's total deposits in FY2024-25?"
        
        # Test MoE
        moe_result = moe.answer_query(query, confidence_threshold=0.7)
        
        # Test RAG
        test_chunks = [{"id": "1", "text": "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25.", "title": "Balance Sheet", "source": "Annual Report"}]
        rag_result = rag.rerank(query, test_chunks, method="cross_encoder", top_k=1)
        
        print("✅ End-to-end test passed")
        print(f"   MoE Answer: {moe_result['answer'][:50]}...")
        print(f"   RAG Chunks: {len(rag_result)} chunks retrieved")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏦 FINQA Complete System Test Suite")
    print("=" * 60)
    print("Group 68: RAG vs Fine-tuning Comparison System")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("RAG System", test_rag_system),
        ("MoE System", test_moe_system),
        ("Training System", test_training_system),
        ("Streamlit Integration", test_streamlit_integration),
        ("End-to-End", test_end_to_end)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total Test Time: {total_time:.2f}s")
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is fully operational.")
        print("\n🚀 Next Steps:")
        print("1. Open http://localhost:8501 in your browser")
        print("2. Test the MoE Fine-tuned system")
        print("3. Train models using the training interface")
        print("4. Compare RAG vs Fine-tuning performance")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
