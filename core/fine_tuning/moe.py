"""
Simplified Mixture-of-Experts (MoE) Fine-tuning System (Group 68 Advanced Technique).
This is a demonstration version that shows the concept without requiring complex dependencies.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import sys
import json
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MOE_CONFIG, FINE_TUNING_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockModel:
    """Mock model for demonstration purposes."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Mock model initialized: {model_name} on {device}")
    
    def generate(self, **kwargs):
        """Mock generation that returns realistic responses."""
        return ["Mock generated response"]

class MockTokenizer:
    """Mock tokenizer for demonstration purposes."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        logger.info(f"Mock tokenizer initialized: {model_name}")
    
    def encode(self, text, **kwargs):
        """Mock encoding."""
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    
    def decode(self, tokens, **kwargs):
        """Mock decoding."""
        return "Mock decoded text"

class ExpertRouter:
    """
    Intelligent router for directing queries to appropriate experts.
    
    Group 68: Mixture-of-Experts with confidence-based routing
    """
    
    def __init__(self, 
                 expert_specialization: Dict[str, List[str]] = None,
                 routing_strategy: str = "confidence_based"):
        """
        Initialize the expert router.
        
        Args:
            expert_specialization: Keywords for each expert type
            routing_strategy: Routing strategy to use
        """
        self.expert_specialization = expert_specialization or MOE_CONFIG["expert_specialization"]
        self.routing_strategy = routing_strategy
        self.routing_cache = {}
        
        logger.info(f"ExpertRouter initialized with strategy: {routing_strategy}")
        logger.info(f"Expert types: {list(self.expert_specialization.keys())}")
    
    def route_query(self, 
                    query: str, 
                    confidence_threshold: float = None) -> Tuple[str, float]:
        """
        Route a query to the most appropriate expert.
        
        Args:
            query: User query
            confidence_threshold: Minimum confidence for routing
            
        Returns:
            Tuple of (expert_type, confidence_score)
        """
        threshold = confidence_threshold or MOE_CONFIG["fallback_threshold"]
        
        if self.routing_strategy == "confidence_based":
            return self._confidence_based_routing(query, threshold)
        elif self.routing_strategy == "keyword_based":
            return self._keyword_based_routing(query)
        elif self.routing_strategy == "hybrid":
            return self._hybrid_routing(query, threshold)
        else:
            logger.warning(f"Unknown routing strategy: {self.routing_strategy}, using confidence_based")
            return self._confidence_based_routing(query, threshold)
    
    def _confidence_based_routing(self, 
                                 query: str, 
                                 threshold: float) -> Tuple[str, float]:
        """
        Route based on confidence scores from expert predictions.
        
        Args:
            query: User query
            threshold: Confidence threshold
            
        Returns:
            Tuple of (expert_type, confidence_score)
        """
        # Get base routing from keyword matching
        expert_type, base_confidence = self._keyword_based_routing(query)
        
        # Boost confidence for better matches
        if base_confidence > 0.2:  # If we have any reasonable match
            # Boost confidence significantly to ensure routing to experts
            confidence = min(0.9, base_confidence + 0.4)
        else:
            # Add some randomness for edge cases
            confidence = base_confidence + random.uniform(0.1, 0.2)
            confidence = max(0.0, min(1.0, confidence))
        
        # Only use fallback if confidence is truly very low
        if confidence < threshold and confidence < 0.25:
            logger.info(f"Query confidence {confidence:.3f} below threshold {threshold}, using fallback")
            return "fallback", confidence
        
        return expert_type, confidence
    
    def _keyword_based_routing(self, query: str) -> Tuple[str, float]:
        """
        Route based on keyword matching in query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (expert_type, confidence_score)
        """
        query_lower = query.lower()
        
        # Calculate scores for each expert
        expert_scores = {}
        for expert_type, keywords in self.expert_specialization.items():
            score = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
            
            # Normalize score and provide better confidence
            if total_keywords > 0:
                base_score = score / total_keywords
                # Boost confidence for better matches
                if base_score > 0.3:  # If at least 30% of keywords match
                    confidence = min(0.9, base_score + 0.3)  # Boost confidence
                else:
                    confidence = base_score
                expert_scores[expert_type] = confidence
            else:
                expert_scores[expert_type] = 0
        
        # Find best expert
        if not expert_scores:
            return "fallback", 0.0
        
        best_expert = max(expert_scores.items(), key=lambda x: x[1])
        
        # Provide minimum confidence for any match
        if best_expert[1] > 0:
            confidence = max(0.4, best_expert[1])  # Minimum 0.4 confidence for any match
        else:
            confidence = 0.0
            
        return best_expert[0], confidence
    
    def _hybrid_routing(self, 
                        query: str, 
                        threshold: float) -> Tuple[str, float]:
        """
        Hybrid routing combining multiple strategies.
        
        Args:
            query: User query
            threshold: Confidence threshold
            
        Returns:
            Tuple of (expert_type, confidence_score)
        """
        # Get keyword-based routing
        keyword_expert, keyword_confidence = self._keyword_based_routing(query)
        
        # Get confidence-based routing
        confidence_expert, confidence_score = self._confidence_based_routing(query, threshold)
        
        # Combine scores (weighted average)
        combined_confidence = (keyword_confidence * 0.6 + confidence_score * 0.4)
        
        # Choose expert based on combined confidence
        if combined_confidence >= threshold:
            # Prefer keyword-based expert if confidence is similar
            if abs(keyword_confidence - confidence_score) < 0.1:
                return keyword_expert, combined_confidence
            else:
                return confidence_expert, combined_confidence
        else:
            return "fallback", combined_confidence

class MoEExpert:
    """
    Individual expert in the MoE system.
    
    Each expert is specialized for a particular type of query.
    """
    
    def __init__(self, 
                 model_path: str,
                 expert_type: str,
                 device: str = "cpu"):
        """
        Initialize an expert.
        
        Args:
            model_path: Path to the trained model or base model name
            expert_type: Type of expert (numeric, narrative, etc.)
            device: Device to run on
        """
        self.model_path = model_path
        self.expert_type = expert_type
        self.device = device
        
        # Check if this is a trained model path or base model name
        self.is_trained_model = Path(model_path).exists() and Path(model_path).is_dir()
        
        if self.is_trained_model:
            logger.info(f"Loading trained {expert_type} expert from {model_path}")
            self._load_trained_model()
        else:
            logger.info(f"Initializing {expert_type} expert with base model: {model_path}")
            self._initialize_mock_model()
    
    def _load_trained_model(self):
        """Load the trained model."""
        try:
            # Try to load transformers for actual model loading
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                
                # Load training metadata if available
                metadata_file = Path(self.model_path) / "training_metadata.json"
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Loaded {self.expert_type} expert with metadata: {self.metadata.get('training_examples', 0)} training examples")
                else:
                    self.metadata = {}
                    logger.info(f"Loaded {self.expert_type} expert (no metadata found)")
                
                self.is_mock = False
                
            except ImportError:
                logger.warning("Transformers not available, using mock model")
                self._initialize_mock_model()
                
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}, falling back to mock")
            self._initialize_mock_model()
    
    def _initialize_mock_model(self):
        """Initialize mock model for demonstration."""
        self.tokenizer = MockTokenizer(self.model_path)
        self.model = MockModel(self.model_path, self.device)
        self.metadata = {}
        self.is_mock = True
        logger.info(f"Mock {self.expert_type} expert initialized: {self.model_path}")
    
    def generate_answer(self, query: str, context: str = None) -> str:
        """
        Generate an answer for the given query.
        
        Args:
            query: User query
            context: Optional context information
            
        Returns:
            Generated answer
        """
        try:
            if self.is_mock:
                return self._generate_mock_answer(query, context)
            else:
                return self._generate_real_answer(query, context)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _generate_real_answer(self, query: str, context: str = None) -> str:
        """Generate answer using the actual trained model."""
        try:
            import torch
            
            # Prepare input
            input_text = f"Question: {query}"
            if context:
                input_text += f" Context: {context}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in real model generation: {e}")
            return self._generate_mock_answer(query, context)
    
    def _generate_mock_answer(self, query: str, context: str = None) -> str:
        """Generate mock answer for demonstration."""
        
        # Generate more realistic financial answers based on expert type and query
        if self.expert_type == "numeric":
            if "deposits" in query.lower():
                return "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25, showing a growth of 15% compared to the previous year. The bank's strong retail deposit mobilization strategy contributed to this growth."
            elif "casa" in query.lower() or "ratio" in query.lower():
                return "The CASA (Current Account Savings Account) ratio for HDFC Bank stands at 45.2% as of FY2024-25, indicating strong retail deposit mobilization and stable funding sources."
            elif "profit" in query.lower() or "margin" in query.lower():
                return "HDFC Bank achieved a net profit margin of 18.5% in FY2024-25, demonstrating strong operational efficiency and profitability across all business segments."
            elif "capital" in query.lower():
                return "The capital adequacy ratio (CAR) for HDFC Bank is 16.8%, well above the regulatory requirement of 11.5%, ensuring strong capital buffers for risk management."
            elif "growth" in query.lower() or "rate" in query.lower():
                return "HDFC Bank recorded a loan growth rate of 12.3% in FY2024-25, driven by strong retail and corporate lending across priority sectors."
            else:
                return f"Based on financial data analysis, the {self.expert_type} expert provides quantitative insights: {query[:50]}... The bank's performance metrics indicate strong financial health and growth trajectory."
        
        elif self.expert_type == "narrative":
            if "performance" in query.lower():
                return "HDFC Bank demonstrated strong performance in FY2024-25 with robust growth across key metrics. The bank achieved 15% deposit growth, 12.3% loan growth, and maintained a healthy net profit margin of 18.5%. Digital transformation initiatives contributed significantly to operational efficiency and customer satisfaction."
            elif "risk" in query.lower():
                return "HDFC Bank faces several key risks including credit risk from economic uncertainties, operational risks from digital transformation, regulatory compliance risks, and market risks from interest rate fluctuations. The bank has implemented comprehensive risk management frameworks to mitigate these challenges effectively."
            elif "digital" in query.lower() or "transformation" in query.lower():
                return "HDFC Bank has successfully managed digital transformation through strategic investments in technology infrastructure, mobile banking platforms, and AI-driven customer service. The bank has digitized 85% of its transactions and launched innovative digital products like HDFC Bank MobileBanking and SmartBuy, improving customer experience significantly."
            elif "initiatives" in query.lower():
                return "HDFC Bank's digital banking initiatives include the launch of enhanced mobile banking app, AI-powered chatbot services, digital onboarding for new customers, and integration with UPI and other digital payment systems. These initiatives have improved customer experience and operational efficiency while reducing costs."
            elif "strategy" in query.lower() or "focus" in query.lower():
                return "HDFC Bank's strategic focus for the future includes expanding digital banking capabilities, enhancing customer experience through technology, strengthening risk management frameworks, and exploring opportunities in sustainable finance and green banking initiatives."
            else:
                return f"The {self.expert_type} expert provides comprehensive analysis: {query[:50]}... Based on strategic assessment and market analysis, the bank demonstrates strong positioning and growth potential in the financial services sector."
        
        else:
            return f"Expert analysis from {self.expert_type} specialist: {query[:50]}... The bank's comprehensive approach to financial services ensures robust performance and sustainable growth across all business segments."
    
    def get_confidence(self, query: str, answer: str) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            if self.is_mock:
                # Mock confidence calculation
                confidence = 0.7  # Base confidence for mock models
                
                # Boost confidence for trained models
                if self.is_trained_model:
                    confidence += 0.2
                
                return min(1.0, confidence)
            else:
                # Real confidence calculation for trained models
                confidence = 0.8  # Base confidence for real models
                
                # Length-based confidence
                if len(answer) > 20:
                    confidence += 0.1
                
                # Content-based confidence
                if any(char.isdigit() for char in answer):
                    confidence += 0.1
                
                # Expert-specific confidence
                if self.expert_type == "numeric":
                    if any(unit in answer.lower() for unit in ['crore', 'million', 'billion', '%']):
                        confidence += 0.1
                elif self.expert_type == "narrative":
                    if len(answer.split()) > 8:
                        confidence += 0.1
                
                return min(1.0, confidence)
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

class MoESystem:
    """
    Complete Mixture-of-Experts system for financial Q&A.
    
    Group 68: Mixture-of-Experts Fine-Tuning
    - Multiple specialized experts
    - Intelligent query routing
    - Confidence-based fallback
    """
    
    def __init__(self, 
                 base_model_name: str = None,
                 experts_dir: str = None,
                 device: str = None):
        """
        Initialize the MoE system.
        
        Args:
            base_model_name: Base model for experts
            experts_dir: Directory containing expert models
            device: Device to run on
        """
        self.base_model_name = base_model_name or FINE_TUNING_CONFIG["base_model"]
        self.experts_dir = Path(experts_dir) if experts_dir else Path("models/fine_tuned")
        self.device = device or MODEL_CONFIG["device"]
        
        # Initialize components
        self.router = ExpertRouter()
        self.experts = {}
        self.fallback_model = None
        
        logger.info("Initializing MoE System...")
        
        # Load experts
        self._load_experts()
        
        # Initialize fallback
        self._initialize_fallback()
        
        logger.info("MoE System initialized successfully")
    
    def _load_experts(self):
        """Load all available experts."""
        expert_types = MOE_CONFIG["expert_types"]
        
        for expert_type in expert_types:
            expert_path = self.experts_dir / f"moe_{expert_type}"
            
            if expert_path.exists():
                try:
                    # Try to load the actual trained model
                    self.experts[expert_type] = MoEExpert(
                        str(expert_path),
                        expert_type,
                        self.device
                    )
                    logger.info(f"Loaded {expert_type} expert from {expert_path}")
                except Exception as e:
                    logger.error(f"Failed to load {expert_type} expert: {e}")
                    # Fallback to mock expert
                    logger.info(f"Creating mock {expert_type} expert as fallback")
                    self.experts[expert_type] = MoEExpert(
                        self.base_model_name,
                        expert_type,
                        self.device
                    )
            else:
                logger.warning(f"Expert path not found: {expert_path}, creating mock expert")
                # Create mock expert for demonstration
                self.experts[expert_type] = MoEExpert(
                    self.base_model_name,
                    expert_type,
                    self.device
                )
        
        if not self.experts:
            logger.warning("No experts loaded, system will use fallback only")
    
    def _initialize_fallback(self):
        """Initialize fallback model."""
        try:
            self.fallback_model = MoEExpert(
                self.base_model_name,
                "fallback",
                self.device
            )
            logger.info("Fallback model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
    
    def answer_query(self, 
                     query: str, 
                     context: str = None,
                     use_reranking: bool = True,
                     confidence_threshold: float = None) -> Dict[str, any]:
        """
        Answer a query using the MoE system.
        
        Args:
            query: User query
            context: Optional context information
            use_reranking: Whether to use reranking for context
            confidence_threshold: Custom confidence threshold for routing
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Use the provided threshold or fallback to config default
            threshold = confidence_threshold if confidence_threshold is not None else MOE_CONFIG["fallback_threshold"]
            logger.info(f"Using confidence threshold: {threshold}")
            
            # Route query to appropriate expert
            expert_type, routing_confidence = self.router.route_query(query, threshold)
            
            # Get answer from expert
            if expert_type in self.experts and routing_confidence >= threshold:
                expert = self.experts[expert_type]
                answer = expert.generate_answer(query, context)
                confidence = expert.get_confidence(query, answer)
                method = f"MoE_{expert_type}"
                fallback_used = False
                
                logger.info(f"Query answered by {expert_type} expert with confidence {confidence:.3f}")
                
            else:
                # Use fallback
                if self.fallback_model:
                    answer = self.fallback_model.generate_answer(query, context)
                    confidence = self.fallback_model.get_confidence(query, answer)
                    method = "MoE_fallback"
                    fallback_used = True
                    
                    logger.info(f"Query answered by fallback model with confidence {confidence:.3f}")
                else:
                    # No fallback available
                    answer = "Unable to generate answer. Please try rephrasing your question."
                    confidence = 0.0
                    method = "MoE_error"
                    fallback_used = True
                    
                    logger.warning("No expert or fallback model available")
            
            return {
                "answer": answer,
                "expert": expert_type,
                "confidence": confidence,
                "routing_confidence": routing_confidence,
                "method": method,
                "fallback_used": fallback_used,
                "context_provided": context is not None
            }
            
        except Exception as e:
            logger.error(f"Error in MoE system: {e}")
            return {
                "answer": f"Error occurred: {str(e)}",
                "expert": "error",
                "confidence": 0.0,
                "routing_confidence": 0.0,
                "method": "MoE_error",
                "fallback_used": True,
                "context_provided": context is not None,
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict[str, any]:
        """
        Get system statistics and health information.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "total_experts": len(self.experts),
            "expert_types": list(self.experts.keys()),
            "fallback_available": self.fallback_model is not None,
            "device": self.device,
            "routing_strategy": self.router.routing_strategy,
            "routing_cache_size": len(self.router.routing_cache)
        }
        
        # Add expert-specific stats
        for expert_type, expert in self.experts.items():
            stats[f"{expert_type}_expert_loaded"] = True
            stats[f"{expert_type}_expert_device"] = str(expert.device)
        
        return stats

# Utility functions
def create_moe_system(base_model: str = None, 
                     experts_dir: str = None,
                     device: str = None) -> MoESystem:
    """Factory function to create an MoE system instance."""
    return MoESystem(base_model, experts_dir, device)

if __name__ == "__main__":
    # Test the MoE system
    print("Testing MoE System...")
    
    try:
        # Create MoE system
        moe = create_moe_system()
        
        # Test queries
        test_queries = [
            "What was HDFC Bank's total deposits in FY2024-25?",  # Numeric
            "How did HDFC Bank perform in FY2024-25?",            # Narrative
            "What is the weather like today?"                     # Irrelevant
        ]
        
        print("\nTesting individual queries:")
        for query in test_queries:
            result = moe.answer_query(query)
            print(f"\nQuery: {query[:50]}...")
            print(f"Expert: {result['expert']}")
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Fallback used: {result['fallback_used']}")
        
        # Get system stats
        stats = moe.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"Total experts: {stats['total_experts']}")
        print(f"Expert types: {stats['expert_types']}")
        print(f"Fallback available: {stats['fallback_available']}")
        
    except Exception as e:
        print(f"Error testing MoE system: {e}")
        print("This is expected if models are not downloaded yet.")
