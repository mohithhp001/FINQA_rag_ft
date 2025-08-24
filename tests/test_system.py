"""
Simple system tests for FINQA components.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class TestSystemComponents(unittest.TestCase):
    """Test basic system components."""
    
    def test_config_import(self):
        """Test configuration system import."""
        try:
            from config.settings import get_config_summary
            config = get_config_summary()
            self.assertIsInstance(config, dict)
            self.assertIn('project_root', config)
            print("✅ Configuration system working")
        except Exception as e:
            self.fail(f"Configuration import failed: {e}")
    
    def test_rag_reranker_import(self):
        """Test RAG reranker import."""
        try:
            from core.rag.reranker import create_reranker
            reranker = create_reranker()
            self.assertIsNotNone(reranker)
            print("✅ RAG reranker working")
        except Exception as e:
            self.fail(f"RAG reranker import failed: {e}")
    
    def test_moe_system_import(self):
        """Test MoE system import."""
        try:
            from core.fine_tuning.moe import create_moe_system
            moe = create_moe_system()
            self.assertIsNotNone(moe)
            print("✅ MoE system working")
        except Exception as e:
            self.fail(f"MoE system import failed: {e}")
    
    def test_rag_functionality(self):
        """Test basic RAG functionality."""
        try:
            from core.rag.reranker import create_reranker
            
            reranker = create_reranker()
            
            # Test query and chunks
            query = "What was HDFC Bank's total deposits?"
            chunks = [
                {"id": "1", "text": "HDFC Bank reported total deposits of ₹2,000,000 crore."},
                {"id": "2", "text": "The bank's deposit growth was strong."}
            ]
            
            # Test reranking
            result = reranker.rerank_chunks(query, chunks, top_k=2)
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 2)
            
            print("✅ RAG functionality working")
            
        except Exception as e:
            self.fail(f"RAG functionality test failed: {e}")
    
    def test_moe_functionality(self):
        """Test basic MoE functionality."""
        try:
            from core.fine_tuning.moe import create_moe_system
            
            moe = create_moe_system()
            
            # Test query
            query = "What was HDFC Bank's total deposits?"
            
            # Test answer generation
            result = moe.answer_query(query)
            self.assertIsInstance(result, dict)
            self.assertIn('answer', result)
            self.assertIn('expert', result)
            self.assertIn('confidence', result)
            
            print("✅ MoE functionality working")
            
        except Exception as e:
            self.fail(f"MoE functionality test failed: {e}")
    
    def test_system_stats(self):
        """Test system statistics."""
        try:
            from core.fine_tuning.moe import create_moe_system
            
            moe = create_moe_system()
            stats = moe.get_system_stats()
            
            self.assertIsInstance(stats, dict)
            self.assertIn('total_experts', stats)
            self.assertIn('expert_types', stats)
            
            print("✅ System statistics working")
            
        except Exception as e:
            self.fail(f"System statistics test failed: {e}")

if __name__ == "__main__":
    # Run tests
    print("🧪 Running FINQA System Tests...")
    print("=" * 50)
    
    unittest.main(verbosity=2)
