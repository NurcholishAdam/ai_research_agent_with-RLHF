# -*- coding: utf-8 -*-
"""
Hierarchical Memory System for AI Research Agent
Implements short-term, long-term, and episodic memory with knowledge graphs
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid
import networkx as nx
from dataclasses import dataclass, asdict
from enum import Enum

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

@dataclass
class MemoryNode:
    """Represents a single memory node"""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime
    importance_score: float
    access_count: int
    last_accessed: datetime
    metadata: Dict[str, Any]
    citations: List[str]
    related_concepts: List[str]

@dataclass
class ResearchEpisode:
    """Represents a complete research episode"""
    id: str
    question: str
    start_time: datetime
    end_time: Optional[datetime]
    findings: List[Dict[str, Any]]
    final_answer: str
    knowledge_gained: List[str]
    citations_used: List[str]

class HierarchicalMemory:
    """Advanced hierarchical memory system"""
    
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.short_term_memory: List[MemoryNode] = []
        self.long_term_memory: Dict[str, MemoryNode] = {}
        self.episodic_memory: Dict[str, ResearchEpisode] = {}
        self.concept_index: Dict[str, List[str]] = {}  # concept -> memory_ids
        self.citation_index: Dict[str, List[str]] = {}  # citation -> memory_ids
        
        # Memory management parameters
        self.short_term_capacity = 20
        self.importance_threshold = 0.7
        self.decay_rate = 0.1
        
    def add_memory(self, content: str, memory_type: MemoryType = MemoryType.SHORT_TERM,
                   importance: float = 0.5, metadata: Dict[str, Any] = None,
                   citations: List[str] = None, concepts: List[str] = None) -> str:
        """Add a new memory to the system"""
        
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        
        memory_node = MemoryNode(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=now,
            importance_score=importance,
            access_count=0,
            last_accessed=now,
            metadata=metadata or {},
            citations=citations or [],
            related_concepts=concepts or []
        )
        
        # Add to appropriate memory store
        if memory_type == MemoryType.SHORT_TERM:
            self._add_to_short_term(memory_node)
        else:
            self.long_term_memory[memory_id] = memory_node
        
        # Update indices
        self._update_concept_index(memory_id, concepts or [])
        self._update_citation_index(memory_id, citations or [])
        
        # Add to knowledge graph
        self._add_to_knowledge_graph(memory_node)
        
        return memory_id
    
    def search_memory(self, query: str, memory_types: List[MemoryType] = None,
                     max_results: int = 10) -> List[Tuple[MemoryNode, float]]:
        """Search across all memory types with relevance scoring"""
        
        if memory_types is None:
            memory_types = list(MemoryType)
        
        results = []
        
        # Search short-term memory
        if MemoryType.SHORT_TERM in memory_types:
            for memory in self.short_term_memory:
                relevance = self._calculate_relevance(query, memory)
                if relevance > 0.1:
                    results.append((memory, relevance))
                    memory.access_count += 1
                    memory.last_accessed = datetime.now()
        
        # Search long-term memory
        if any(mt in memory_types for mt in [MemoryType.LONG_TERM, MemoryType.SEMANTIC]):
            for memory in self.long_term_memory.values():
                if memory.memory_type in memory_types:
                    relevance = self._calculate_relevance(query, memory)
                    if relevance > 0.1:
                        results.append((memory, relevance))
                        memory.access_count += 1
                        memory.last_accessed = datetime.now()
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def consolidate_memories(self):
        """Move important short-term memories to long-term storage"""
        
        memories_to_promote = []
        
        for memory in self.short_term_memory:
            # Calculate consolidation score
            age_factor = (datetime.now() - memory.timestamp).days
            access_factor = memory.access_count
            importance_factor = memory.importance_score
            
            consolidation_score = (importance_factor * 0.5 + 
                                 min(access_factor / 10, 1.0) * 0.3 + 
                                 min(age_factor / 7, 1.0) * 0.2)
            
            if consolidation_score > self.importance_threshold:
                memories_to_promote.append(memory)
        
        # Move to long-term memory
        for memory in memories_to_promote:
            memory.memory_type = MemoryType.LONG_TERM
            self.long_term_memory[memory.id] = memory
            self.short_term_memory.remove(memory)
        
        # Apply decay to remaining short-term memories
        self._apply_decay()
        
        return len(memories_to_promote)
    
    def create_research_episode(self, question: str) -> str:
        """Start a new research episode"""
        
        episode_id = str(uuid.uuid4())
        episode = ResearchEpisode(
            id=episode_id,
            question=question,
            start_time=datetime.now(),
            end_time=None,
            findings=[],
            final_answer="",
            knowledge_gained=[],
            citations_used=[]
        )
        
        self.episodic_memory[episode_id] = episode
        return episode_id
    
    def update_research_episode(self, episode_id: str, finding: Dict[str, Any] = None,
                               final_answer: str = None, citations: List[str] = None):
        """Update an ongoing research episode"""
        
        if episode_id not in self.episodic_memory:
            return False
        
        episode = self.episodic_memory[episode_id]
        
        if finding:
            episode.findings.append(finding)
        
        if final_answer:
            episode.final_answer = final_answer
            episode.end_time = datetime.now()
        
        if citations:
            episode.citations_used.extend(citations)
        
        return True
    
    def get_knowledge_graph_insights(self, concept: str) -> Dict[str, Any]:
        """Get insights from the knowledge graph about a concept"""
        
        if concept not in self.knowledge_graph:
            return {"error": f"Concept '{concept}' not found in knowledge graph"}
        
        # Get connected concepts
        connected = list(self.knowledge_graph.neighbors(concept))
        
        # Get centrality measures
        centrality = nx.degree_centrality(self.knowledge_graph).get(concept, 0)
        
        # Get shortest paths to other important concepts
        important_concepts = sorted(
            nx.degree_centrality(self.knowledge_graph).items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        
        paths = {}
        for imp_concept, _ in important_concepts:
            if imp_concept != concept and nx.has_path(self.knowledge_graph, concept, imp_concept):
                try:
                    path = nx.shortest_path(self.knowledge_graph, concept, imp_concept)
                    paths[imp_concept] = path
                except nx.NetworkXNoPath:
                    continue
        
        return {
            "concept": concept,
            "connected_concepts": connected,
            "centrality_score": centrality,
            "paths_to_important_concepts": paths,
            "total_connections": len(connected)
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        now = datetime.now()
        
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "episodic_count": len(self.episodic_memory),
            "knowledge_graph_nodes": self.knowledge_graph.number_of_nodes(),
            "knowledge_graph_edges": self.knowledge_graph.number_of_edges(),
            "concepts_tracked": len(self.concept_index),
            "citations_tracked": len(self.citation_index),
            "recent_episodes": len([e for e in self.episodic_memory.values() 
                                  if (now - e.start_time).days <= 7]),
            "memory_utilization": len(self.short_term_memory) / self.short_term_capacity
        }
    
    def _add_to_short_term(self, memory_node: MemoryNode):
        """Add memory to short-term storage with capacity management"""
        
        self.short_term_memory.append(memory_node)
        
        # If over capacity, remove least important memories
        if len(self.short_term_memory) > self.short_term_capacity:
            # Sort by importance and recency
            self.short_term_memory.sort(
                key=lambda m: (m.importance_score, m.last_accessed),
                reverse=True
            )
            # Keep only the most important ones
            self.short_term_memory = self.short_term_memory[:self.short_term_capacity]
    
    def _calculate_relevance(self, query: str, memory: MemoryNode) -> float:
        """Calculate relevance score between query and memory"""
        
        query_lower = query.lower()
        content_lower = memory.content.lower()
        
        # Simple keyword matching (can be enhanced with embeddings)
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        if union == 0:
            base_relevance = 0
        else:
            base_relevance = intersection / union
        
        # Boost for concept matches
        concept_boost = 0
        for concept in memory.related_concepts:
            if concept.lower() in query_lower:
                concept_boost += 0.2
        
        # Boost for recent access
        recency_boost = max(0, 0.1 - (datetime.now() - memory.last_accessed).days * 0.01)
        
        # Boost for importance
        importance_boost = memory.importance_score * 0.1
        
        return min(1.0, base_relevance + concept_boost + recency_boost + importance_boost)
    
    def _update_concept_index(self, memory_id: str, concepts: List[str]):
        """Update the concept index"""
        for concept in concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(memory_id)
    
    def _update_citation_index(self, memory_id: str, citations: List[str]):
        """Update the citation index"""
        for citation in citations:
            if citation not in self.citation_index:
                self.citation_index[citation] = []
            self.citation_index[citation].append(memory_id)
    
    def _add_to_knowledge_graph(self, memory_node: MemoryNode):
        """Add memory node to knowledge graph"""
        
        # Add concepts as nodes
        for concept in memory_node.related_concepts:
            if not self.knowledge_graph.has_node(concept):
                self.knowledge_graph.add_node(concept, type="concept")
        
        # Connect related concepts
        concepts = memory_node.related_concepts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                if not self.knowledge_graph.has_edge(concept1, concept2):
                    self.knowledge_graph.add_edge(concept1, concept2, weight=1)
                else:
                    # Strengthen existing connection
                    self.knowledge_graph[concept1][concept2]['weight'] += 1
    
    def _apply_decay(self):
        """Apply decay to short-term memories"""
        
        for memory in self.short_term_memory:
            age_days = (datetime.now() - memory.timestamp).days
            decay_factor = max(0.1, 1.0 - (age_days * self.decay_rate))
            memory.importance_score *= decay_factor