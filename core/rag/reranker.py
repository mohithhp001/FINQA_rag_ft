"""
Simplified Cross-Encoder Re-Ranking System for RAG (Group 68 Advanced Technique).
This is a demonstration version that shows the concept without requiring complex dependencies.
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAG_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockCrossEncoder:
    """Mock cross-encoder for demonstration purposes."""
    
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        logger.info(f"Mock CrossEncoder initialized: {model_name}")
    
    def predict(self, pairs):
        """Mock prediction that returns realistic scores."""
        import random
        # Simulate realistic cross-encoder scores
        scores = []
        for query, text in pairs:
            # Simple heuristic scoring based on content
            base_score = 0.5
            if any(word in text.lower() for word in query.lower().split()):
                base_score += 0.2
            if len(text) > 100:
                base_score += 0.1
            # Add some randomness to simulate real model behavior
            score = base_score + random.uniform(-0.1, 0.1)
            scores.append(max(0.1, min(0.9, score)))
        return scores

class CrossEncoderReranker:
    """
    Cross-Encoder based re-ranking system for improving RAG retrieval quality.
    
    Group 68 Advanced Technique: Re-Ranking with Cross-Encoders
    - Uses cross-encoder models to re-rank retrieved chunks
    - Improves precision by considering query-chunk pairs together
    - Balances retrieval speed with ranking quality
    """
    
    def __init__(self, 
                 model_name: str = None,
                 device: str = None,
                 max_length: int = 512):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model to use
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.model_name = model_name or RAG_CONFIG["reranker_model"]
        self.device = device or MODEL_CONFIG["device"]
        self.max_length = max_length
        
        logger.info(f"Initializing CrossEncoderReranker with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Try to import real CrossEncoder, fallback to mock
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length
                )
                logger.info("Real CrossEncoder model loaded successfully")
            except ImportError:
                logger.warning("sentence_transformers not available, using mock encoder")
                self.model = MockCrossEncoder(self.model_name, self.device, self.max_length)
                
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            logger.info("Falling back to mock encoder")
            self.model = MockCrossEncoder(self.model_name, self.device, self.max_length)
    
    def rerank_chunks(self, 
                      query: str, 
                      chunks: List[Dict], 
                      top_k: int = None) -> List[Tuple[Dict, float]]:
        """
        Re-rank retrieved chunks using cross-encoder scoring.
        
        Args:
            query: User query
            chunks: List of retrieved chunks with metadata
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk, score) tuples, sorted by score
        """
        if not chunks:
            logger.warning("No chunks provided for re-ranking")
            return []
        
        top_k = top_k or RAG_CONFIG["rerank_top_k"]
        logger.info(f"Re-ranking {len(chunks)} chunks for query: {query[:50]}...")
        
        try:
            # Prepare query-chunk pairs for cross-encoder
            pairs = []
            for chunk in chunks:
                # Create query-chunk pair
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    pairs.append([query, chunk_text])
                else:
                    logger.warning(f"Chunk missing text: {chunk.get('id', 'unknown')}")
            
            if not pairs:
                logger.error("No valid query-chunk pairs found")
                return []
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Pair chunks with scores and sort
            chunk_scores = list(zip(chunks, scores))
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            top_results = chunk_scores[:top_k]
            
            logger.info(f"Re-ranking complete. Top score: {top_results[0][1]:.4f}")
            return top_results
            
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            # Fallback: return original chunks with default scores
            return [(chunk, 0.5) for chunk in chunks[:top_k]]
    
    def rerank_with_metadata(self, 
                            query: str, 
                            chunks: List[Dict], 
                            top_k: int = None,
                            include_metadata: bool = True) -> List[Dict]:
        """
        Re-rank chunks and return with enhanced metadata.
        
        Args:
            query: User query
            chunks: List of retrieved chunks
            top_k: Number of top chunks to return
            include_metadata: Whether to include reranking metadata
            
        Returns:
            List of re-ranked chunks with scores and metadata
        """
        reranked = self.rerank_chunks(query, chunks, top_k)
        
        if not include_metadata:
            return [chunk for chunk, _ in reranked]
        
        # Enhance chunks with reranking information
        enhanced_chunks = []
        for i, (chunk, score) in enumerate(reranked):
            enhanced_chunk = chunk.copy()
            enhanced_chunk['rerank_score'] = float(score)
            enhanced_chunk['rerank_rank'] = i + 1
            enhanced_chunk['rerank_confidence'] = self._score_to_confidence(score)
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert numerical score to confidence level."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"

class HybridReranker:
    """
    Hybrid reranking system combining cross-encoder with other ranking methods.
    Provides fallback mechanisms and performance optimization.
    """
    
    def __init__(self, 
                 cross_encoder_model: str = None,
                 use_cache: bool = True,
                 cache_size: int = 1000):
        """
        Initialize hybrid reranker.
        
        Args:
            cross_encoder_model: Cross-encoder model name
            use_cache: Whether to use caching for performance
            cache_size: Maximum cache size
        """
        self.cross_encoder = CrossEncoderReranker(cross_encoder_model)
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.cache = {}
        
        logger.info("HybridReranker initialized")
    
    def rerank(self, 
               query: str, 
               chunks: List[Dict], 
               method: str = "cross_encoder",
               top_k: int = None) -> List[Dict]:
        """
        Re-rank chunks using specified method with fallback.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            method: Reranking method
            top_k: Number of top results
            
        Returns:
            Re-ranked chunks
        """
        try:
            if method == "cross_encoder":
                return self.cross_encoder.rerank_with_metadata(query, chunks, top_k)
            elif method == "hybrid":
                return self._hybrid_rerank(query, chunks, top_k)
            else:
                logger.warning(f"Unknown reranking method: {method}, using cross-encoder")
                return self.cross_encoder.rerank_with_metadata(query, chunks, top_k)
        except Exception as e:
            logger.error(f"Reranking failed with method {method}: {e}")
            # Fallback: return original chunks
            return chunks[:top_k] if top_k else chunks
    
    def _hybrid_rerank(self, 
                       query: str, 
                       chunks: List[Dict], 
                       top_k: int = None) -> List[Dict]:
        """
        Hybrid reranking combining multiple approaches.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            top_k: Number of top results
            
        Returns:
            Hybrid re-ranked chunks
        """
        # First pass: cross-encoder reranking
        cross_encoder_results = self.cross_encoder.rerank_chunks(query, chunks, top_k)
        
        # Second pass: apply additional ranking factors
        enhanced_results = []
        for chunk, score in cross_encoder_results:
            enhanced_chunk = chunk.copy()
            enhanced_chunk['rerank_score'] = score
            
            # Apply additional ranking factors
            enhanced_chunk['final_score'] = self._calculate_hybrid_score(
                chunk, score, query
            )
            enhanced_results.append(enhanced_chunk)
        
        # Sort by final hybrid score
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return enhanced_results[:top_k] if top_k else enhanced_results
    
    def _calculate_hybrid_score(self, 
                               chunk: Dict, 
                               cross_encoder_score: float, 
                               query: str) -> float:
        """
        Calculate hybrid score combining multiple factors.
        
        Args:
            chunk: Chunk data
            cross_encoder_score: Cross-encoder score
            query: User query
            
        Returns:
            Hybrid score
        """
        # Base score from cross-encoder
        score = cross_encoder_score
        
        # Length penalty (prefer appropriately sized chunks)
        chunk_length = len(chunk.get('text', ''))
        if chunk_length < 50:
            score *= 0.9  # Penalty for very short chunks
        elif chunk_length > 1000:
            score *= 0.95  # Slight penalty for very long chunks
        
        # Source quality bonus
        source = chunk.get('source', '').lower()
        if 'balance sheet' in source or 'income statement' in source:
            score *= 1.1  # Bonus for financial statements
        elif 'directors report' in source:
            score *= 1.05  # Bonus for official reports
        
        # Recency bonus (if available)
        if 'year' in chunk.get('title', '').lower():
            if '2025' in chunk.get('title', ''):
                score *= 1.05  # Bonus for current year
        
        return min(1.0, score)  # Cap at 1.0

# Utility functions
def create_reranker(model_name: str = None, 
                   device: str = None) -> CrossEncoderReranker:
    """Factory function to create a reranker instance."""
    return CrossEncoderReranker(model_name, device)

def create_hybrid_reranker(model_name: str = None) -> HybridReranker:
    """Factory function to create a hybrid reranker instance."""
    return HybridReranker(model_name)

if __name__ == "__main__":
    # Test the reranker
    print("Testing CrossEncoderReranker...")
    
    # Create test data
    test_query = "What was HDFC Bank's total deposits in FY2024-25?"
    test_chunks = [
        {"id": "1", "text": "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25."},
        {"id": "2", "text": "The bank's deposit growth was strong during the fiscal year."},
        {"id": "3", "text": "Total deposits increased by 15% compared to the previous year."}
    ]
    
    try:
        reranker = create_reranker()
        reranked = reranker.rerank_chunks(test_query, test_chunks, top_k=2)
        
        print(f"Query: {test_query}")
        print("Re-ranked results:")
        for chunk, score in reranked:
            print(f"  Score: {score:.4f}, Text: {chunk['text'][:60]}...")
            
    except Exception as e:
        print(f"Error testing reranker: {e}")
        print("This is expected if models are not downloaded yet.")
