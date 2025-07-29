# -*- coding: utf-8 -*-
"""
Knowledge Graph Visualizer for AI Research Agent
Creates visual representations of research knowledge and concept relationships
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os

class KnowledgeGraphVisualizer:
    """Visualizes knowledge graphs and research relationships"""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        
    def visualize_concept_network(self, concepts: List[str], 
                                save_path: str = "knowledge_graph.png") -> str:
        """Create a visual network of concepts and their relationships"""
        
        if not self.memory_manager:
            return "Error: No memory manager provided"
        
        # Get knowledge graph from memory manager
        kg = self.memory_manager.hierarchical_memory.knowledge_graph
        
        if not kg.nodes():
            return "No knowledge graph data available"
        
        # Create subgraph with specified concepts
        if concepts:
            # Find concepts that exist in the graph
            existing_concepts = [c for c in concepts if c in kg.nodes()]
            if not existing_concepts:
                return f"None of the specified concepts found in knowledge graph: {concepts}"
            
            # Get subgraph including neighbors
            nodes_to_include = set(existing_concepts)
            for concept in existing_concepts:
                nodes_to_include.update(kg.neighbors(concept))
            
            subgraph = kg.subgraph(nodes_to_include)
        else:
            subgraph = kg
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Calculate layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes
        node_sizes = []
        node_colors = []
        for node in subgraph.nodes():
            # Size based on degree centrality
            centrality = nx.degree_centrality(subgraph).get(node, 0)
            node_sizes.append(300 + centrality * 1000)
            
            # Color based on whether it's in original concepts
            if node in concepts:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_size=node_sizes,
                              node_color=node_colors,
                              alpha=0.7)
        
        # Draw edges
        edge_weights = []
        for edge in subgraph.edges():
            weight = subgraph[edge[0]][edge[1]].get('weight', 1)
            edge_weights.append(weight)
        
        nx.draw_networkx_edges(subgraph, pos,
                              width=[w * 0.5 for w in edge_weights],
                              alpha=0.5,
                              edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos,
                               font_size=8,
                               font_weight='bold')
        
        plt.title("Knowledge Graph: Concept Relationships", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Knowledge graph saved to {save_path}"
    
    def create_research_timeline(self, episode_id: str = None,
                               save_path: str = "research_timeline.png") -> str:
        """Create a timeline visualization of research progress"""
        
        if not self.memory_manager:
            return "Error: No memory manager provided"
        
        episodes = self.memory_manager.hierarchical_memory.episodic_memory
        
        if not episodes:
            return "No research episodes found"
        
        # Use specific episode or most recent
        if episode_id and episode_id in episodes:
            episode = episodes[episode_id]
            episodes_to_plot = [episode]
        else:
            # Get most recent episodes
            sorted_episodes = sorted(episodes.values(), 
                                   key=lambda x: x.start_time, reverse=True)
            episodes_to_plot = sorted_episodes[:5]  # Last 5 episodes
        
        plt.figure(figsize=(14, 8))
        
        for i, episode in enumerate(episodes_to_plot):
            # Create timeline for this episode
            y_pos = i
            
            # Plot episode duration
            start_time = episode.start_time
            end_time = episode.end_time or datetime.now()
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            plt.barh(y_pos, duration, height=0.6, 
                    alpha=0.7, label=f"Episode {i+1}")
            
            # Add findings as markers
            for j, finding in enumerate(episode.findings):
                finding_time = j * (duration / len(episode.findings)) if episode.findings else 0
                plt.scatter(finding_time, y_pos, s=50, alpha=0.8)
            
            # Add episode info
            episode_info = f"{episode.question[:50]}..."
            plt.text(duration + 1, y_pos, episode_info, 
                    va='center', fontsize=8)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Research Episodes')
        plt.title('Research Timeline', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Research timeline saved to {save_path}"
    
    def generate_memory_report(self, save_path: str = "memory_report.json") -> str:
        """Generate a comprehensive memory system report"""
        
        if not self.memory_manager:
            return "Error: No memory manager provided"
        
        hm = self.memory_manager.hierarchical_memory
        
        # Collect statistics
        stats = hm.get_memory_statistics()
        
        # Get top concepts by centrality
        if hm.knowledge_graph.nodes():
            centrality = nx.degree_centrality(hm.knowledge_graph)
            top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_concepts = []
        
        # Get recent episodes summary
        recent_episodes = []
        for episode in hm.episodic_memory.values():
            episode_summary = {
                "id": episode.id,
                "question": episode.question,
                "start_time": episode.start_time.isoformat(),
                "findings_count": len(episode.findings),
                "citations_count": len(episode.citations_used),
                "duration_minutes": ((episode.end_time or datetime.now()) - episode.start_time).total_seconds() / 60
            }
            recent_episodes.append(episode_summary)
        
        # Compile report
        report = {
            "generated_at": datetime.now().isoformat(),
            "memory_statistics": stats,
            "top_concepts": top_concepts,
            "recent_episodes": recent_episodes[-10:],  # Last 10 episodes
            "knowledge_graph_summary": {
                "total_nodes": hm.knowledge_graph.number_of_nodes(),
                "total_edges": hm.knowledge_graph.number_of_edges(),
                "density": nx.density(hm.knowledge_graph) if hm.knowledge_graph.nodes() else 0
            }
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return f"Memory report saved to {save_path}"

def create_visualization_tools(memory_manager):
    """Create visualization tools for the research agent"""
    
    visualizer = KnowledgeGraphVisualizer(memory_manager)
    
    def visualize_concepts_tool(concepts_str: str) -> str:
        """Visualize concept relationships"""
        try:
            concepts = [c.strip() for c in concepts_str.split(',')]
            return visualizer.visualize_concept_network(concepts)
        except Exception as e:
            return f"Error creating concept visualization: {str(e)}"
    
    def create_timeline_tool(episode_id: str = "") -> str:
        """Create research timeline"""
        try:
            return visualizer.create_research_timeline(episode_id if episode_id else None)
        except Exception as e:
            return f"Error creating timeline: {str(e)}"
    
    def generate_report_tool(dummy_input: str = "") -> str:
        """Generate memory system report"""
        try:
            return visualizer.generate_memory_report()
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    from langchain.tools import Tool
    
    return [
        Tool(
            name="visualize_concepts",
            description="Create a visual network of concept relationships. Input: comma-separated list of concepts",
            func=visualize_concepts_tool
        ),
        Tool(
            name="create_research_timeline",
            description="Create a timeline visualization of research progress. Input: episode_id (optional)",
            func=create_timeline_tool
        ),
        Tool(
            name="generate_memory_report",
            description="Generate a comprehensive memory system report in JSON format",
            func=generate_report_tool
        )
    ]