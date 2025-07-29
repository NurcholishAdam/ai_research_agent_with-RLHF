# -*- coding: utf-8 -*-
"""
Reward Model Training for RLHF - Phase 6
Fine-tunes transformer models on human feedback for reward prediction
"""

from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    DistilBertModel, RobertaModel,
    Trainer, TrainingArguments,
    get_linear_schedule_with_warmup
)
import numpy as np
from dataclasses import dataclass
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from rlhf.feedback_system import FeedbackCollector, FeedbackPair, FeedbackType

@dataclass
class RewardModelConfig:
    """Configuration for reward model training"""
    model_name: str = "distilbert-base-uncased"  # or "roberta-base"
    max_length: int = 512
    hidden_size: int = 768
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50

class RewardModel(nn.Module):
    """Transformer-based reward model for RLHF"""
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        if "distilbert" in config.model_name.lower():
            self.transformer = DistilBertModel.from_pretrained(config.model_name)
        elif "roberta" in config.model_name.lower():
            self.transformer = RobertaModel.from_pretrained(config.model_name)
        else:
            self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Reward head
        self.dropout = nn.Dropout(config.dropout)
        self.reward_head = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights"""
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and get reward score
        pooled_output = self.dropout(pooled_output)
        reward = self.reward_head(pooled_output)
        
        return reward.squeeze(-1)  # Remove last dimension

class FeedbackDataset(Dataset):
    """Dataset for training reward model on feedback pairs"""
    
    def __init__(self, feedback_pairs: List[FeedbackPair], 
                 tokenizer, max_length: int = 512):
        self.pairs = feedback_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare training examples
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict[str, Any]]:
        """Prepare training examples from feedback pairs"""
        examples = []
        
        for pair in self.pairs:
            # Create preference label
            if pair.preference == "a":
                preferred_text = pair.output_a.content
                rejected_text = pair.output_b.content
            elif pair.preference == "b":
                preferred_text = pair.output_b.content
                rejected_text = pair.output_a.content
            else:  # tie
                # For ties, we can skip or treat as neutral
                continue
            
            # Add context (research question)
            context = f"Research Question: {pair.output_a.research_question}\n\n"
            
            examples.append({
                "preferred": context + preferred_text,
                "rejected": context + rejected_text,
                "feedback_type": pair.feedback_type.value,
                "confidence": pair.confidence
            })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize preferred and rejected texts
        preferred_encoding = self.tokenizer(
            example["preferred"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            example["rejected"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "preferred_input_ids": preferred_encoding["input_ids"].squeeze(0),
            "preferred_attention_mask": preferred_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "confidence": torch.tensor(example["confidence"], dtype=torch.float)
        }

class RewardModelTrainer:
    """Trainer for reward model using contrastive learning"""
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = RewardModel(config).to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger("RewardModelTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Training history
        self.training_history = []
    
    def bradley_terry_loss(self, preferred_rewards: torch.Tensor, 
                          rejected_rewards: torch.Tensor,
                          confidence_weights: torch.Tensor = None) -> torch.Tensor:
        """Bradley-Terry-Luce model loss for preference learning"""
        
        # Compute log probability that preferred is better than rejected
        logits = preferred_rewards - rejected_rewards
        loss = -F.logsigmoid(logits)
        
        # Apply confidence weighting if provided
        if confidence_weights is not None:
            loss = loss * confidence_weights
        
        return loss.mean()
    
    def ranking_loss(self, preferred_rewards: torch.Tensor,
                    rejected_rewards: torch.Tensor,
                    margin: float = 1.0) -> torch.Tensor:
        """Ranking loss with margin"""
        loss = F.relu(margin - (preferred_rewards - rejected_rewards))
        return loss.mean()
    
    def train_model(self, feedback_pairs: List[FeedbackPair],
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the reward model on feedback pairs"""
        
        if len(feedback_pairs) < 10:
            raise ValueError("Need at least 10 feedback pairs for training")
        
        # Split data
        train_pairs, val_pairs = train_test_split(
            feedback_pairs, 
            test_size=validation_split,
            random_state=42
        )
        
        # Create datasets
        train_dataset = FeedbackDataset(train_pairs, self.tokenizer, self.config.max_length)
        val_dataset = FeedbackDataset(val_pairs, self.tokenizer, self.config.max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                preferred_rewards = self.model(
                    batch["preferred_input_ids"],
                    batch["preferred_attention_mask"]
                )
                
                rejected_rewards = self.model(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )
                
                # Compute loss
                loss = self.bradley_terry_loss(
                    preferred_rewards,
                    rejected_rewards,
                    batch["confidence"]
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    self.logger.info(
                        f"Step {global_step}, Loss: {loss.item():.4f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
            
            # Validation
            val_loss, val_accuracy = self._validate(val_loader)
            avg_train_loss = epoch_loss / num_batches
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model("best_reward_model.pt")
            
            # Record training history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        return {
            "training_history": self.training_history,
            "best_val_loss": best_val_loss,
            "total_steps": global_step,
            "model_path": "best_reward_model.pt"
        }
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                preferred_rewards = self.model(
                    batch["preferred_input_ids"],
                    batch["preferred_attention_mask"]
                )
                
                rejected_rewards = self.model(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )
                
                # Compute loss
                loss = self.bradley_terry_loss(
                    preferred_rewards,
                    rejected_rewards,
                    batch["confidence"]
                )
                
                total_loss += loss.item()
                
                # Compute accuracy (preferred should have higher reward)
                correct = (preferred_rewards > rejected_rewards).float()
                correct_predictions += correct.sum().item()
                total_predictions += len(correct)
        
        self.model.train()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def predict_reward(self, text: str, context: str = "") -> float:
        """Predict reward for a given text"""
        self.model.eval()
        
        # Prepare input
        full_text = f"{context}\n\n{text}" if context else text
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            reward = self.model(
                encoding["input_ids"],
                encoding["attention_mask"]
            )
        
        return reward.item()
    
    def compare_outputs(self, text_a: str, text_b: str, context: str = "") -> Dict[str, Any]:
        """Compare two outputs and return preference prediction"""
        reward_a = self.predict_reward(text_a, context)
        reward_b = self.predict_reward(text_b, context)
        
        # Compute preference probability using Bradley-Terry model
        logit = reward_a - reward_b
        prob_a_preferred = torch.sigmoid(torch.tensor(logit)).item()
        
        return {
            "reward_a": reward_a,
            "reward_b": reward_b,
            "preference_a_prob": prob_a_preferred,
            "preference_b_prob": 1 - prob_a_preferred,
            "predicted_preference": "a" if reward_a > reward_b else "b",
            "confidence": abs(prob_a_preferred - 0.5) * 2  # Scale to 0-1
        }
    
    def _save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "tokenizer": self.tokenizer,
            "training_history": self.training_history
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.config = checkpoint["config"]
        self.training_history = checkpoint.get("training_history", [])
        
        self.logger.info(f"Model loaded from {path}")

class RewardModelManager:
    """Manages reward model training and inference"""
    
    def __init__(self, config: RewardModelConfig = None):
        self.config = config or RewardModelConfig()
        self.trainer = RewardModelTrainer(self.config)
        self.feedback_collector = FeedbackCollector()
        
        # Model performance tracking
        self.performance_history = []
    
    def train_from_feedback(self, feedback_type: FeedbackType = None,
                          min_pairs: int = 50) -> Dict[str, Any]:
        """Train reward model from collected feedback"""
        
        # Get training pairs from feedback collector
        if self.feedback_collector.mongo_store:
            training_pairs = self.feedback_collector.mongo_store.get_training_pairs(
                feedback_type=feedback_type,
                limit=1000
            )
        else:
            training_pairs = [
                pair for pair in self.feedback_collector.pairs_memory
                if feedback_type is None or pair.feedback_type == feedback_type
            ]
        
        if len(training_pairs) < min_pairs:
            raise ValueError(f"Need at least {min_pairs} training pairs, got {len(training_pairs)}")
        
        # Train the model
        training_results = self.trainer.train_model(training_pairs)
        
        # Record performance
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type.value if feedback_type else "all",
            "training_pairs": len(training_pairs),
            "best_val_loss": training_results["best_val_loss"],
            "total_steps": training_results["total_steps"]
        }
        
        self.performance_history.append(performance_record)
        
        return {
            "training_results": training_results,
            "performance_record": performance_record,
            "model_ready": True
        }
    
    def evaluate_agent_output(self, output_content: str, 
                            research_question: str = "") -> Dict[str, Any]:
        """Evaluate agent output using trained reward model"""
        
        context = f"Research Question: {research_question}" if research_question else ""
        
        try:
            reward_score = self.trainer.predict_reward(output_content, context)
            
            # Normalize reward to 0-1 scale (approximate)
            normalized_score = torch.sigmoid(torch.tensor(reward_score)).item()
            
            return {
                "raw_reward": reward_score,
                "normalized_score": normalized_score,
                "quality_assessment": self._interpret_score(normalized_score),
                "model_confidence": min(abs(reward_score), 2.0) / 2.0  # Rough confidence estimate
            }
            
        except Exception as e:
            return {
                "error": f"Reward prediction failed: {str(e)}",
                "raw_reward": 0.0,
                "normalized_score": 0.5,
                "quality_assessment": "unknown"
            }
    
    def _interpret_score(self, score: float) -> str:
        """Interpret normalized reward score"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "average"
        elif score >= 0.2:
            return "poor"
        else:
            return "very_poor"
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get reward model performance statistics"""
        
        if not self.performance_history:
            return {"status": "no_training_history"}
        
        latest_performance = self.performance_history[-1]
        
        return {
            "model_config": {
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "learning_rate": self.config.learning_rate
            },
            "latest_performance": latest_performance,
            "total_training_sessions": len(self.performance_history),
            "training_history": self.performance_history[-5:]  # Last 5 sessions
        }

def get_reward_model_manager() -> RewardModelManager:
    """Get the global reward model manager instance"""
    return RewardModelManager()