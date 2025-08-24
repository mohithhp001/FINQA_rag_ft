#!/usr/bin/env python3
"""
Simple Fine-tuning for FINQA MoE Models
Group 68: Mixture-of-Experts Fine-tuning Implementation

This script performs actual fine-tuning with compatibility fixes.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_data():
    """Create training data for fine-tuning."""
    
    training_data = [
        # Numeric queries
        {
            "question": "What was HDFC Bank's total deposits in FY2024-25?",
            "answer": "HDFC Bank reported total deposits of ₹2,000,000 crore in FY2024-25, showing a growth of 15% compared to the previous year.",
            "type": "numeric"
        },
        {
            "question": "What is the CASA ratio for HDFC Bank?",
            "answer": "The CASA ratio for HDFC Bank stands at 45.2% as of FY2024-25, indicating strong retail deposit mobilization.",
            "type": "numeric"
        },
        {
            "question": "What was the net profit margin in FY2024-25?",
            "answer": "HDFC Bank achieved a net profit margin of 18.5% in FY2024-25, demonstrating strong operational efficiency.",
            "type": "numeric"
        },
        {
            "question": "What is the capital adequacy ratio?",
            "answer": "The capital adequacy ratio for HDFC Bank is 16.8%, well above the regulatory requirement of 11.5%.",
            "type": "numeric"
        },
        {
            "question": "What was the loan growth rate?",
            "answer": "HDFC Bank recorded a loan growth rate of 12.3% in FY2024-25, driven by strong retail and corporate lending.",
            "type": "numeric"
        },
        
        # Narrative queries
        {
            "question": "How did HDFC Bank perform in FY2024-25?",
            "answer": "HDFC Bank demonstrated strong performance in FY2024-25 with robust growth across key metrics and digital transformation initiatives.",
            "type": "narrative"
        },
        {
            "question": "What are the key risks facing HDFC Bank?",
            "answer": "HDFC Bank faces several key risks including credit risk, operational risks, and regulatory compliance risks.",
            "type": "narrative"
        },
        {
            "question": "How did the bank manage digital transformation?",
            "answer": "HDFC Bank has successfully managed digital transformation through strategic investments in technology infrastructure.",
            "type": "narrative"
        },
        {
            "question": "What were the digital banking initiatives?",
            "answer": "HDFC Bank's digital banking initiatives include enhanced mobile banking app, AI-powered chatbot services, and UPI integration.",
            "type": "narrative"
        },
        {
            "question": "What is the bank's strategic focus for the future?",
            "answer": "HDFC Bank's strategic focus includes expanding digital banking capabilities and enhancing customer experience through technology.",
            "type": "narrative"
        }
    ]
    
    return training_data

def train_with_simple_approach():
    """Train models using a simpler, more compatible approach."""
    
    print("🏦 FINQA Simple Fine-tuning Training")
    print("=" * 50)
    
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForSeq2SeqLM,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq
        )
        from datasets import Dataset
        import torch
        import json
        
        # Configuration
        base_model = "google/flan-t5-small"
        models_dir = Path("models/fine_tuned")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training dataset
        training_data = create_training_data()
        print(f"📊 Created training dataset with {len(training_data)} examples")
        
        # Load base model and tokenizer
        print(f"📥 Loading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Train numeric expert
        print("\n🔧 Training numeric expert...")
        numeric_data = [item for item in training_data if item["type"] == "numeric"]
        
        # Prepare dataset
        questions = [item["question"] for item in numeric_data]
        answers = [item["answer"] for item in numeric_data]
        
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers
        })
        
        # Tokenize function
        def tokenize_function(examples):
            inputs = tokenizer(
                examples["question"],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            targets = tokenizer(
                examples["answer"],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": targets["input_ids"]
            }
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments (simplified)
        training_args = TrainingArguments(
            output_dir=str(models_dir / "moe_numeric"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        print("🚀 Starting training for numeric expert...")
        trainer.train()
        
        # Save the model
        numeric_dir = models_dir / "moe_numeric"
        trainer.save_model(str(numeric_dir))
        tokenizer.save_pretrained(str(numeric_dir))
        
        # Save metadata
        metadata = {
            "expert_type": "numeric",
            "training_examples": len(numeric_data),
            "base_model": base_model,
            "training_notes": "Model fine-tuned for numeric financial queries"
        }
        
        with open(numeric_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Numeric expert trained and saved to {numeric_dir}")
        
        # Train narrative expert
        print("\n🔧 Training narrative expert...")
        narrative_data = [item for item in training_data if item["type"] == "narrative"]
        
        # Prepare dataset
        questions = [item["question"] for item in narrative_data]
        answers = [item["answer"] for item in narrative_data]
        
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers
        })
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Update training arguments
        training_args.output_dir = str(models_dir / "moe_narrative")
        
        # Create new trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        print("🚀 Starting training for narrative expert...")
        trainer.train()
        
        # Save the model
        narrative_dir = models_dir / "moe_narrative"
        trainer.save_model(str(narrative_dir))
        tokenizer.save_pretrained(str(narrative_dir))
        
        # Save metadata
        metadata["expert_type"] = "narrative"
        metadata["training_examples"] = len(narrative_data)
        metadata["training_notes"] = "Model fine-tuned for narrative financial queries"
        
        with open(narrative_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Narrative expert trained and saved to {narrative_dir}")
        
        print("\n🎉 All MoE expert models trained successfully!")
        print("You can now use the MoE Fine-tuned system with real trained models.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Main function."""
    print("🏦 FINQA MoE Model Fine-tuning")
    print("=" * 50)
    
    success = train_with_simple_approach()
    
    if success:
        print("\n🚀 Next Steps:")
        print("1. Restart the Streamlit app")
        print("2. Test the MoE Fine-tuned system")
        print("3. Verify that real trained models are loaded")
    else:
        print("\n❌ Training failed. Check the error above.")

if __name__ == "__main__":
    main()
