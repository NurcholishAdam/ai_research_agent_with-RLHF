# -*- coding: utf-8 -*-
"""
RLHF Feedback System for AI Research Agent - Phase 6
Captures agent outputs and human ratings for reinforcement learning
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import uuid
import logging
from enum import Enum

# Database imports
try:
    import pymongo
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

class FeedbackType(Enum):
    QUALITY = "quality"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    USEFULNESS = "usefulness"
    OVERALL = "overall"

class FeedbackRating(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    TERRIBLE = 1

@dataclass
class AgentOutput:
    """Represents an agent output for feedback collection"""
    id: str
    session_id: str
    research_question: str
    output_type: str  # "plan", "finding", "hypothesis", "final_answer"
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    agent_confidence: float
    context: Dict[str, Any]

@dataclass
class HumanFeedback:
    """Represents human feedback on agent output"""
    id: str
    output_id: str
    feedback_type: FeedbackType
    rating: FeedbackRating
    comment: Optional[str]
    timestamp: datetime
    user_id: str
    session_context: Dict[str, Any]

@dataclass
class FeedbackPair:
    """Training pair for reward model"""
    id: str
    output_a: AgentOutput
    output_b: AgentOutput
    preference: str  # "a", "b", or "tie"
    feedback_type: FeedbackType
    confidence: float
    timestamp: datetime
    user_id: str

class FeedbackLogger:
    """Logs agent outputs and human feedback"""
    
    def __init__(self, log_file: str = "rlhf_feedback.log"):
        self.logger = logging.getLogger("RLHF_Feedback")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_agent_output(self, output: AgentOutput):
        """Log agent output for feedback collection"""
        log_data = {
            "type": "agent_output",
            "data": asdict(output)
        }
        # Convert datetime to string for JSON serialization
        log_data["data"]["timestamp"] = output.timestamp.isoformat()
        
        self.logger.info(f"AGENT_OUTPUT: {json.dumps(log_data)}")
    
    def log_human_feedback(self, feedback: HumanFeedback):
        """Log human feedback"""
        log_data = {
            "type": "human_feedback",
            "data": asdict(feedback)
        }
        # Convert enums and datetime to strings
        log_data["data"]["feedback_type"] = feedback.feedback_type.value
        log_data["data"]["rating"] = feedback.rating.value
        log_data["data"]["timestamp"] = feedback.timestamp.isoformat()
        
        self.logger.info(f"HUMAN_FEEDBACK: {json.dumps(log_data)}")
    
    def log_feedback_pair(self, pair: FeedbackPair):
        """Log feedback pair for training"""
        log_data = {
            "type": "feedback_pair",
            "data": {
                "id": pair.id,
                "output_a_id": pair.output_a.id,
                "output_b_id": pair.output_b.id,
                "preference": pair.preference,
                "feedback_type": pair.feedback_type.value,
                "confidence": pair.confidence,
                "timestamp": pair.timestamp.isoformat(),
                "user_id": pair.user_id
            }
        }
        
        self.logger.info(f"FEEDBACK_PAIR: {json.dumps(log_data)}")

class MongoFeedbackStore:
    """MongoDB storage for feedback data"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 database_name: str = "rlhf_feedback"):
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB support requires pymongo: pip install pymongo")
        
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.outputs_collection = self.db.agent_outputs
        self.feedback_collection = self.db.human_feedback
        self.pairs_collection = self.db.feedback_pairs
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes"""
        # Agent outputs indexes
        self.outputs_collection.create_index("session_id")
        self.outputs_collection.create_index("timestamp")
        self.outputs_collection.create_index("output_type")
        
        # Feedback indexes
        self.feedback_collection.create_index("output_id")
        self.feedback_collection.create_index("user_id")
        self.feedback_collection.create_index("timestamp")
        self.feedback_collection.create_index("feedback_type")
        
        # Pairs indexes
        self.pairs_collection.create_index("user_id")
        self.pairs_collection.create_index("timestamp")
        self.pairs_collection.create_index("feedback_type")
    
    def store_agent_output(self, output: AgentOutput):
        """Store agent output in MongoDB"""
        doc = asdict(output)
        doc["timestamp"] = output.timestamp
        
        self.outputs_collection.insert_one(doc)
    
    def store_human_feedback(self, feedback: HumanFeedback):
        """Store human feedback in MongoDB"""
        doc = asdict(feedback)
        doc["feedback_type"] = feedback.feedback_type.value
        doc["rating"] = feedback.rating.value
        doc["timestamp"] = feedback.timestamp
        
        self.feedback_collection.insert_one(doc)
    
    def store_feedback_pair(self, pair: FeedbackPair):
        """Store feedback pair in MongoDB"""
        doc = {
            "id": pair.id,
            "output_a_id": pair.output_a.id,
            "output_b_id": pair.output_b.id,
            "output_a": asdict(pair.output_a),
            "output_b": asdict(pair.output_b),
            "preference": pair.preference,
            "feedback_type": pair.feedback_type.value,
            "confidence": pair.confidence,
            "timestamp": pair.timestamp,
            "user_id": pair.user_id
        }
        
        # Convert datetime objects
        doc["output_a"]["timestamp"] = pair.output_a.timestamp
        doc["output_b"]["timestamp"] = pair.output_b.timestamp
        
        self.pairs_collection.insert_one(doc)
    
    def get_agent_outputs(self, session_id: str = None, 
                         output_type: str = None, 
                         limit: int = 100) -> List[AgentOutput]:
        """Retrieve agent outputs"""
        query = {}
        if session_id:
            query["session_id"] = session_id
        if output_type:
            query["output_type"] = output_type
        
        cursor = self.outputs_collection.find(query).limit(limit).sort("timestamp", -1)
        
        outputs = []
        for doc in cursor:
            doc.pop("_id", None)  # Remove MongoDB ID
            outputs.append(AgentOutput(**doc))
        
        return outputs
    
    def get_feedback_for_output(self, output_id: str) -> List[HumanFeedback]:
        """Get all feedback for a specific output"""
        cursor = self.feedback_collection.find({"output_id": output_id})
        
        feedback_list = []
        for doc in cursor:
            doc.pop("_id", None)
            doc["feedback_type"] = FeedbackType(doc["feedback_type"])
            doc["rating"] = FeedbackRating(doc["rating"])
            feedback_list.append(HumanFeedback(**doc))
        
        return feedback_list
    
    def get_training_pairs(self, feedback_type: FeedbackType = None, 
                          limit: int = 1000) -> List[FeedbackPair]:
        """Get feedback pairs for training"""
        query = {}
        if feedback_type:
            query["feedback_type"] = feedback_type.value
        
        cursor = self.pairs_collection.find(query).limit(limit).sort("timestamp", -1)
        
        pairs = []
        for doc in cursor:
            doc.pop("_id", None)
            
            # Reconstruct AgentOutput objects
            output_a = AgentOutput(**doc["output_a"])
            output_b = AgentOutput(**doc["output_b"])
            
            pair = FeedbackPair(
                id=doc["id"],
                output_a=output_a,
                output_b=output_b,
                preference=doc["preference"],
                feedback_type=FeedbackType(doc["feedback_type"]),
                confidence=doc["confidence"],
                timestamp=doc["timestamp"],
                user_id=doc["user_id"]
            )
            pairs.append(pair)
        
        return pairs

class VectorFeedbackStore:
    """Vector database storage for semantic feedback search"""
    
    def __init__(self, collection_name: str = "rlhf_feedback"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("Vector storage requires chromadb: pip install chromadb")
        
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RLHF feedback storage"}
        )
    
    def store_output_with_feedback(self, output: AgentOutput, 
                                  feedback_list: List[HumanFeedback]):
        """Store output with associated feedback for semantic search"""
        
        # Calculate average rating
        if feedback_list:
            avg_rating = sum(f.rating.value for f in feedback_list) / len(feedback_list)
            feedback_summary = {
                "avg_rating": avg_rating,
                "feedback_count": len(feedback_list),
                "feedback_types": list(set(f.feedback_type.value for f in feedback_list))
            }
        else:
            feedback_summary = {
                "avg_rating": 0,
                "feedback_count": 0,
                "feedback_types": []
            }
        
        # Prepare metadata
        metadata = {
            "output_id": output.id,
            "session_id": output.session_id,
            "output_type": output.output_type,
            "agent_confidence": output.agent_confidence,
            "timestamp": output.timestamp.isoformat(),
            **feedback_summary
        }
        
        # Store in vector database
        self.collection.add(
            documents=[output.content],
            metadatas=[metadata],
            ids=[output.id]
        )
    
    def search_similar_outputs(self, query_text: str, 
                              min_rating: float = 0,
                              output_type: str = None,
                              n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for similar outputs with good feedback"""
        
        where_clause = {"avg_rating": {"$gte": min_rating}}
        if output_type:
            where_clause["output_type"] = output_type
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause
        )
        
        return results

class FeedbackCollector:
    """Main feedback collection system"""
    
    def __init__(self, use_mongodb: bool = True, use_vector_db: bool = True):
        self.logger = FeedbackLogger()
        
        # Storage backends
        self.mongo_store = None
        self.vector_store = None
        
        if use_mongodb:
            try:
                self.mongo_store = MongoFeedbackStore()
            except Exception as e:
                print(f"MongoDB not available: {e}")
        
        if use_vector_db:
            try:
                self.vector_store = VectorFeedbackStore()
            except Exception as e:
                print(f"Vector DB not available: {e}")
        
        # In-memory storage as fallback
        self.outputs_memory: List[AgentOutput] = []
        self.feedback_memory: List[HumanFeedback] = []
        self.pairs_memory: List[FeedbackPair] = []
    
    def capture_agent_output(self, session_id: str, research_question: str,
                           output_type: str, content: str, 
                           metadata: Dict[str, Any] = None,
                           agent_confidence: float = 0.5,
                           context: Dict[str, Any] = None) -> str:
        """Capture agent output for feedback"""
        
        output = AgentOutput(
            id=str(uuid.uuid4()),
            session_id=session_id,
            research_question=research_question,
            output_type=output_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now(),
            agent_confidence=agent_confidence,
            context=context or {}
        )
        
        # Log the output
        self.logger.log_agent_output(output)
        
        # Store in databases
        if self.mongo_store:
            self.mongo_store.store_agent_output(output)
        
        # Store in memory as fallback
        self.outputs_memory.append(output)
        
        return output.id
    
    def collect_human_feedback(self, output_id: str, feedback_type: FeedbackType,
                             rating: FeedbackRating, comment: str = None,
                             user_id: str = "anonymous",
                             session_context: Dict[str, Any] = None) -> str:
        """Collect human feedback on agent output"""
        
        feedback = HumanFeedback(
            id=str(uuid.uuid4()),
            output_id=output_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            timestamp=datetime.now(),
            user_id=user_id,
            session_context=session_context or {}
        )
        
        # Log the feedback
        self.logger.log_human_feedback(feedback)
        
        # Store in databases
        if self.mongo_store:
            self.mongo_store.store_human_feedback(feedback)
        
        # Store in memory
        self.feedback_memory.append(feedback)
        
        # Update vector store if available
        if self.vector_store:
            self._update_vector_store(output_id)
        
        return feedback.id
    
    def create_feedback_pair(self, output_a_id: str, output_b_id: str,
                           preference: str, feedback_type: FeedbackType,
                           confidence: float = 1.0, user_id: str = "anonymous") -> str:
        """Create a feedback pair for training"""
        
        # Get outputs
        output_a = self._get_output_by_id(output_a_id)
        output_b = self._get_output_by_id(output_b_id)
        
        if not output_a or not output_b:
            raise ValueError("One or both outputs not found")
        
        pair = FeedbackPair(
            id=str(uuid.uuid4()),
            output_a=output_a,
            output_b=output_b,
            preference=preference,
            feedback_type=feedback_type,
            confidence=confidence,
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        # Log the pair
        self.logger.log_feedback_pair(pair)
        
        # Store in databases
        if self.mongo_store:
            self.mongo_store.store_feedback_pair(pair)
        
        # Store in memory
        self.pairs_memory.append(pair)
        
        return pair.id
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback collection statistics"""
        
        if self.mongo_store:
            # Get from MongoDB
            total_outputs = self.mongo_store.outputs_collection.count_documents({})
            total_feedback = self.mongo_store.feedback_collection.count_documents({})
            total_pairs = self.mongo_store.pairs_collection.count_documents({})
        else:
            # Get from memory
            total_outputs = len(self.outputs_memory)
            total_feedback = len(self.feedback_memory)
            total_pairs = len(self.pairs_memory)
        
        # Calculate average ratings
        if self.mongo_store:
            pipeline = [
                {"$group": {
                    "_id": "$feedback_type",
                    "avg_rating": {"$avg": "$rating"},
                    "count": {"$sum": 1}
                }}
            ]
            rating_stats = list(self.mongo_store.feedback_collection.aggregate(pipeline))
        else:
            rating_stats = []
            feedback_by_type = {}
            for feedback in self.feedback_memory:
                if feedback.feedback_type not in feedback_by_type:
                    feedback_by_type[feedback.feedback_type] = []
                feedback_by_type[feedback.feedback_type].append(feedback.rating.value)
            
            for feedback_type, ratings in feedback_by_type.items():
                rating_stats.append({
                    "_id": feedback_type.value,
                    "avg_rating": sum(ratings) / len(ratings),
                    "count": len(ratings)
                })
        
        return {
            "total_outputs": total_outputs,
            "total_feedback": total_feedback,
            "total_pairs": total_pairs,
            "rating_statistics": rating_stats,
            "collection_rate": total_feedback / max(total_outputs, 1)
        }
    
    def _get_output_by_id(self, output_id: str) -> Optional[AgentOutput]:
        """Get output by ID"""
        if self.mongo_store:
            doc = self.mongo_store.outputs_collection.find_one({"id": output_id})
            if doc:
                doc.pop("_id", None)
                return AgentOutput(**doc)
        
        # Check memory
        for output in self.outputs_memory:
            if output.id == output_id:
                return output
        
        return None
    
    def _update_vector_store(self, output_id: str):
        """Update vector store with feedback"""
        if not self.vector_store:
            return
        
        output = self._get_output_by_id(output_id)
        if not output:
            return
        
        # Get feedback for this output
        if self.mongo_store:
            feedback_list = self.mongo_store.get_feedback_for_output(output_id)
        else:
            feedback_list = [f for f in self.feedback_memory if f.output_id == output_id]
        
        # Update vector store
        self.vector_store.store_output_with_feedback(output, feedback_list)

def get_feedback_collector() -> FeedbackCollector:
    """Get the global feedback collector instance"""
    return FeedbackCollector()
