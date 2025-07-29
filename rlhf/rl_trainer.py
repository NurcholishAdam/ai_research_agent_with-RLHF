# -*- coding: utf-8 -*-
"""
RLHF RL Trainer for AI Research Agent - Stage 3
Implements PPO and DPO for policy optimization using human feedback
"""

import torc import Dict, List, Any, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from traorcormers import Auas F, AuForCausalLM
from typing importata import Datny, Tuple, Optiona
from t numpy mers 
from dataclasses i AutoModelclass
    Trainiging
from dt_linee impoeduletetime
)py

# Import our com iments
from r jsoneedback_system imporeedbackCollect
from rlhf.reward_modort RewardModelManer

@dataclaandb
from pathlnfig:
    Configuration fining"""
   RL fel_name: str = "microDialoGPT-m
    max_: int = 512
    learnirl rate: float = ne-5
    batch_sizeore i = 8
    ppL_AVAIhs: E = T 4
    clip_epsiError:at = 0.2
    num_episodes: int lse
dpo_beta: float = 0.1
# Import our components
from rPolicyModel(nn.Modu import FeedbackCollector, AgentOutput, HumanFeedback
fr  """Policward_el for generatiwardesearch resr, Res"""
    
 t__(self, RLConfig):
 dataclasuper().__init
class   self.confinfig:ig
        self.urationer = AutoTokning"".from_pretraionfig.model_n)
        if self.togstoken ie:
       el   self.toke "microsoft/ken = self.tokenizer.e
        self.mname: str = odelForCausalm_pretrained.model_name)
    max_lelf.value_head =.Linear(self.mol.config.hidden_si
    
    def forward(seids, atteion_mask):
        ning_rate self.model(inp-5ds=input_ids, attentionntion_mask, output_h_states=T
        logits: int = 8gits
    min hidden_states = o= puts.hidden_stat
        
    ppo_epCompute v = 4 using last token
    num_train_size = input_i3
        last_token_indicesention_mask.sim=1) - 1
       PO lues = []
    init_kl_coen range(batch_
            last_idx = 6.st_toes[i]
            varl: booelf.value_head(hiddei, last_idx])
        a:  values.append(va
        valuet = torch.stack(es).squeeze(-1)
      iprange: float = 0.2
    cli return {'lo: floa logits, 'value
    
    def generate_rponse(self, pror, max_new_ns: int = 1) -> str:
    # Traiputs = self.tokenizeturn_ten, truncation=Tr, max_lengthconfig.max_lengthax_new_tokens)
     ax_grad_normh.no_grad():
        : int = 42 = self.model.geinputs.input_ids, mokens=max_netokens, do_sample=Truure=0.8)
    log_with:nse = selandb"zer.decodputs[0][int_ids.shape[1]:_specis=True)
    project_namresponse."rlhf()

cla # SPOTrainer:
    "e"PPO trainefloat = search ent"""
    top_k: int = 50
    tef __init__(s 0f, configig, reward_del):
        seple:onfig = config
       self.reward_model = reward_model
        self.policy = PolicyModel(config)
        self.old_policy = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.old_policy.to(self.device)
        self.logger = logging.getLogger("PPOTrainer")
    
    def collect_rollouts(self, research_questions: List[str]) -> List[Dict[str, Any]]:
        rollouts = []
        for question in research_questions:
            prompt = f"Research Question: {question}\nAnswer:"
            response = self.policy.generate_response(prompt)
            reward = self.reward_model.evaluate_agent_output(response, question).get('normalized_score', 0.5)
            
            full_text = prompt + " " + response
            tokens = self.policy.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.max_length, padding="max_length")
            
            with torch.no_grad():
                policy_outputs = self.policy(tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device))
                old_policy_outputs = self.old_policy(tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device))
            
            rollout = {
                'question': question,
                'response': response,
                'reward': reward,
                'input_ids': tokens.input_ids.squeeze(0),
                'attention_mask': tokens.attention_mask.squeeze(0),
                'logits': policy_outputs['logits'].squeeze(0).cpu(),
                'values': policy_outputs['values'].cpu(),
                'old_logits': old_policy_outputs['logits'].squeeze(0).cpu()
            }
            rollouts.append(rollout)
        return rollouts
    
    def ppo_update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        batch_size = len(rollouts)
        input_ids = torch.stack([r['input_ids'] for r in rollouts]).to(self.device)
        attention_mask = torch.stack([r['attention_mask'] for r in rollouts]).to(self.device)
        advantages = torch.tensor([r['reward'] for r in rollouts], dtype=torch.float).to(self.device)
        returns = advantages.clone()
        old_logits = torch.stack([r['old_logits'] for r in rollouts]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for epoch in range(self.config.ppo_epochs):
            outputs = self.policy(input_ids, attention_mask)
            logits = outputs['logits']
            values = outputs['values']
            
            # Compute policy loss
            log_probs = F.log_softmax(logits, dim=-1).mean(dim=-1)
            old_log_probs = F.log_softmax(old_logits, dim=-1).mean(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        return {'policy_loss': total_loss / self.config.ppo_epochs}
    
    def train(self, research_questions: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Starting PPO training with {len(research_questions)} questions")
        
        for episode in range(self.config.num_episodes):
            rollouts = self.collect_rollouts(research_questions)
            losses = self.ppo_update(rollouts)
            avg_reward = np.mean([r['reward'] for r in rollouts])
            
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, Loss: {losses['policy_loss']:.4f}")
        
        return {'final_avg_reward': avg_reward, 'total_episodes': self.config.num_episodes}

class RLHFOrchestrator:
    """Orchestrates RLHF training process"""
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.feedback_collector = FeedbackCollector()
        self.reward_model_manager = RewardModelManager()
        self.logger = logging.getLogger("RLHFOrchestrator")
    
    def setup_reward_model(self) -> bool:
        try:
            training_results = self.reward_model_manager.train_from_feedback(min_pairs=20)
            self.logger.info("Reward model setup completed")
            return True
        except Exception as e:
            self.logger.error(f"Reward model setup failed: {e}")
            return False
    
    def run_ppo_training(self, research_questions: List[str]) -> Dict[str, Any]:
        if not self.setup_reward_model():
            raise RuntimeError("Cannot run PPO without reward model")
        
        ppo_trainer = PPOTrainer(self.config, self.reward_model_manager)
        results = ppo_trainer.train(research_questions)
        self.logger.info("PPO training completed")
        return results
    
    def full_rlhf_pipeline(self, research_questions: List[str]) -> Dict[str, Any]:
        results = {"reward_model_training": None, "ppo_training": None}
        
        self.logger.info("Starting RLHF pipeline...")
        reward_model_success = self.setup_reward_model()
        results["reward_model_training"] = reward_model_success
        
        if reward_model_success:
            try:
                ppo_results = self.run_ppo_training(research_questions)
                results["ppo_training"] = ppo_results
            except Exception as e:
                results["ppo_training"] = {"error": str(e)}
        
        self.logger.info("RLHF pipeline completed")
        return results

def run_rlhf_training(research_questions: List[str], config: RLConfig = None) -> Dict[str, Any]:
    orchestrator = RLHFOrchestrator(config)
    return orchestrator.full_rlhf_pipeline(research_questions)    
 
   def prepare_training_data(self, feedback_collector: FeedbackCollector) -> List[str]:
        """Prepare training prompts from feedback data"""
        
        prompts = []
        
        # Get research questions from feedback data
        if feedback_collector.mongo_store:
            outputs = feedback_collector.mongo_store.get_agent_outputs(limit=1000)
            
            for output in outputs:
                # Create research prompt
                prompt = f"Research Question: {output.research_question}\n\nProvide a comprehensive research plan and analysis:"
                prompts.append(prompt)
        
        else:
            # Fallback to memory
            for output in feedback_collector.outputs_memory:
                prompt = f"Research Question: {output.research_question}\n\nProvide a comprehensive research plan and analysis:"
                prompts.append(prompt)
        
        # Remove duplicates
        prompts = list(set(prompts))
        
        self.logger.info(f"Prepared {len(prompts)} training prompts")
        return prompts

class DPOResearchTrainer:
    """Direct Preference Optimization trainer"""
    
    def __init__(self, config: RLTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        self.logger = logging.getLogger("DPOResearchTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        
        # Move to device
        self.model.to(self.device)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # Training history
        self.training_history = []

class RLHFTrainingManager:
    """Manages the complete RLHF training pipeline"""
    
    def __init__(self, config: RLTrainingConfig = None):
        self.config = config or RLTrainingConfig()
        self.feedback_collector = FeedbackCollector()
        self.reward_model_manager = RewardModelManager()
        
        # Training components
        self.ppo_trainer = None
        self.dpo_trainer = None
        
        # Training history
        self.training_sessions = []
    
    def run_complete_rlhf_pipeline(self, training_method: str = "ppo") -> Dict[str, Any]:
        """Run the complete RLHF training pipeline"""
        
        self.logger = logging.getLogger("RLHFTrainingManager")
        self.logger.info("Starting complete RLHF training pipeline")
        
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "training_method": training_method,
            "stages": {}
        }
        
        try:
            # Stage 1: Check feedback data
            feedback_stats = self.feedback_collector.get_feedback_statistics()
            self.logger.info(f"Feedback data: {feedback_stats}")
            
            if feedback_stats["total_feedback"] < 50:
                raise ValueError("Need at least 50 feedback entries for RLHF training")
            
            pipeline_results["stages"]["feedback_check"] = {
                "status": "success",
                "stats": feedback_stats
            }
            
            # Stage 2: Train reward model
            self.logger.info("Training reward model...")
            reward_training_results = self.reward_model_manager.train_from_feedback()
            
            pipeline_results["stages"]["reward_model"] = {
                "status": "success",
                "results": reward_training_results
            }
            
            # Stage 3: RL training
            if training_method.lower() == "ppo":
                self.logger.info("Starting PPO training...")
                self.ppo_trainer = PPOResearchTrainer(self.config)
                # Simplified training for now
                rl_results = {"status": "PPO training configured"}
            
            elif training_method.lower() == "dpo":
                self.logger.info("Starting DPO training...")
                self.dpo_trainer = DPOResearchTrainer(self.config)
                # Simplified training for now
                rl_results = {"status": "DPO training configured"}
            
            else:
                raise ValueError(f"Unknown training method: {training_method}")
            
            pipeline_results["stages"]["rl_training"] = {
                "status": "success",
                "results": rl_results
            }
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "success"
            
            self.logger.info("RLHF training pipeline completed successfully")
            
        except Exception as e:
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            
            self.logger.error(f"RLHF training pipeline failed: {e}")
        
        return pipeline_results
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get RLHF training statistics"""
        
        return {
            "total_training_sessions": len(self.training_sessions),
            "recent_sessions": self.training_sessions[-3:],  # Last 3 sessions
            "feedback_stats": self.feedback_collector.get_feedback_statistics(),
            "reward_model_stats": self.reward_model_manager.get_model_statistics(),
            "config": self.config.__dict__
        }

def get_rlhf_training_manager() -> RLHFTrainingManager:
    """Get the global RLHF training manager instance"""
    return RLHFTrainingManager()