#!/usr/bin/env python3
"""
Simplified RLHF Feedback System for testing without external dependencies
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

class FeedbackType(Enum):
    QUALITY = "quality"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"

class FeedbackRating(Enum):
    POOR = 1
    FAIR = 2
    GOOD = 3
    VERY_GOOD = 4
    EXCELLENT = 5

@dataclass
class ResearchOutput:
    id: str
    research_question: str
    research_result: Dict[str, Any]
    session_id: str
    timestamp: str
    metadata: Dict[str, Any] = None

@dataclass
class HumanFeedback:
    id: str
    output_id: str
    feedback_type: FeedbackType
    rating: FeedbackRating
    comments: str
    user_id: str
    timestamp: str

class SimpleFeedbackCollector:
    """Simplified feedback collector for testing"""
    
    def __init__(self):
        self.research_outputs: Dict[str, ResearchOutput] = {}
        self.feedback_data: Dict[str, HumanFeedback] = {}
        print("🎯 Simple Feedback Collector initialized")
    
    def capture_research_output(
        self,
        research_result: Dict[str, Any],
        research_question: str,
        session_id: str
    ) -> ResearchOutput:
        """Capture research output for feedback"""
        output_id = str(uuid.uuid4())
        
        output = ResearchOutput(
            id=output_id,
            research_question=research_question,
            research_result=research_result,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metadata={}
        )
        
        self.research_outputs[output_id] = output
        return output
    
    def collect_human_feedback(
        self,
        output_id: str,
        feedback_type: FeedbackType,
        rating: FeedbackRating,
        comments: str,
        user_id: str
    ) -> HumanFeedback:
        """Collect human feedback on research output"""
        feedback_id = str(uuid.uuid4())
        
        feedback = HumanFeedback(
            id=feedback_id,
            output_id=output_id,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
        
        self.feedback_data[feedback_id] = feedback
        return feedback
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        if not self.feedback_data:
            return {
                "total_feedback": 0,
                "average_quality_rating": 0,
                "feedback_by_type": {},
                "total_outputs": len(self.research_outputs)
            }
        
        feedback_list = list(self.feedback_data.values())
        total_feedback = len(feedback_list)
        
        # Calculate average rating
        total_rating = sum(f.rating.value for f in feedback_list)
        avg_rating = total_rating / total_feedback if total_feedback > 0 else 0
        
        # Group by feedback type
        feedback_by_type = {}
        for feedback in feedback_list:
            ftype = feedback.feedback_type.value
            if ftype not in feedback_by_type:
                feedback_by_type[ftype] = []
            feedback_by_type[ftype].append(feedback.rating.value)
        
        return {
            "total_feedback": total_feedback,
            "average_quality_rating": round(avg_rating, 2),
            "feedback_by_type": {
                ftype: {
                    "count": len(ratings),
                    "average": round(sum(ratings) / len(ratings), 2)
                }
                for ftype, ratings in feedback_by_type.items()
            },
            "total_outputs": len(self.research_outputs)
        }

class SimpleRewardModelManager:
    """Simplified reward model for testing"""
    
    def __init__(self):
        print("🏆 Simple Reward Model Manager initialized")
    
    def evaluate_agent_output(
        self,
        output_content: str,
        research_question: str
    ) -> Dict[str, Any]:
        """Simple evaluation based on content length and keywords"""
        
        # Simple scoring based on content characteristics
        content_length = len(output_content)
        word_count = len(output_content.split())
        
        # Basic quality indicators
        has_structure = any(marker in output_content.lower() for marker in [
            "introduction", "conclusion", "findings", "analysis", "evidence"
        ])
        
        has_citations = any(marker in output_content for marker in [
            "source:", "reference", "according to", "study shows"
        ])
        
        # Calculate normalized score (0-1)
        length_score = min(content_length / 1000, 1.0)  # Normalize to 1000 chars
        word_score = min(word_count / 200, 1.0)  # Normalize to 200 words
        structure_score = 0.3 if has_structure else 0.0
        citation_score = 0.2 if has_citations else 0.0
        
        normalized_score = (length_score * 0.3 + word_score * 0.2 + 
                          structure_score + citation_score)
        
        return {
            "normalized_score": round(normalized_score, 3),
            "content_length": content_length,
            "word_count": word_count,
            "has_structure": has_structure,
            "has_citations": has_citations,
            "quality_indicators": {
                "length_adequate": content_length > 200,
                "well_structured": has_structure,
                "properly_cited": has_citations
            }
        }

# Compatibility functions for existing code
def FeedbackCollector():
    """Factory function for backward compatibility"""
    return SimpleFeedbackCollector()

def RewardModelManager():
    """Factory function for backward compatibility"""
    return SimpleRewardModelManager()