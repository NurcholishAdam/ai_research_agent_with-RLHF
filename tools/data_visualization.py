# -*- coding: utf-8 -*-
"""
Data Visualization and Analysis Tools for AI Research Agent
Creates charts, graphs, and visual analysis of research data
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional
from langchain.tools import Tool
import json
import re
from datetime import datetime
import os

class DataVisualizationTool:
    """Advanced data visualization capabilities"""
    
    def __init__(self):
        self.output_dir = "research_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_research_timeline(self, data: List[Dict[str, Any]], 
                               title: str = "Research Timeline") -> str:
        """Create timeline visualization of research events"""
        try:
            if not data:
                return "No data provided for timeline visualization"
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure we have required columns
            if 'date' not in df.columns or 'event' not in df.columns:
                return "Data must contain 'date' and 'event' columns"
            
            # Convert dates
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if df.empty:
                return "No valid dates found in data"
            
            # Create timeline plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=[1] * len(df),
                mode='markers+text',
                marker=dict(size=12, color='blue'),
                text=df['event'],
                textposition="top center",
                name="Events"
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis=dict(showticklabels=False, showgrid=False),
                height=400,
                showlegend=False
            )
            
            # Save plot
            filename = f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            
            return f"Timeline visualization saved to {filepath}"
            
        except Exception as e:
            return f"Timeline visualization failed: {str(e)}"
    
    def create_concept_network(self, concepts: List[str], 
                             relationships: List[Dict[str, Any]] = None) -> str:
        """Create network visualization of concepts"""
        try:
            import networkx as nx
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for concept in concepts:
                G.add_node(concept)
            
            # Add edges from relationships
            if relationships:
                for rel in relationships:
                    if 'source' in rel and 'target' in rel:
                        weight = rel.get('weight', 1)
                        G.add_edge(rel['source'], rel['target'], weight=weight)
            else:
                # Create simple connections between consecutive concepts
                for i in range(len(concepts) - 1):
                    G.add_edge(concepts[i], concepts[i + 1])
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Extract coordinates
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            # Create edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create plot
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=list(G.nodes()),
                textposition="middle center",
                marker=dict(
                    size=30,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            ))
            
            fig.update_layout(
                title="Concept Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Concept relationships in research",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='gray', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Save plot
            filename = f"concept_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            
            return f"Concept network visualization saved to {filepath}"
            
        except Exception as e:
            return f"Concept network visualization failed: {str(e)}"
    
    def create_research_metrics_dashboard(self, metrics: Dict[str, Any]) -> str:
        """Create dashboard of research metrics"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Research Progress', 'Source Types', 'Citation Network', 'Key Metrics'),
                specs=[[{"type": "scatter"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Research Progress (if timeline data available)
            if 'timeline' in metrics:
                timeline_data = metrics['timeline']
                dates = [item['date'] for item in timeline_data]
                progress = list(range(1, len(dates) + 1))
                
                fig.add_trace(
                    go.Scatter(x=dates, y=progress, mode='lines+markers', name='Progress'),
                    row=1, col=1
                )
            
            # Source Types
            if 'sources' in metrics:
                sources = metrics['sources']
                fig.add_trace(
                    go.Pie(labels=list(sources.keys()), values=list(sources.values()), name="Sources"),
                    row=1, col=2
                )
            
            # Citation counts
            if 'citations' in metrics:
                citations = metrics['citations']
                fig.add_trace(
                    go.Bar(x=list(citations.keys()), y=list(citations.values()), name="Citations"),
                    row=2, col=1
                )
            
            # Key metric indicator
            if 'total_findings' in metrics:
                fig.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=metrics['total_findings'],
                        title={"text": "Total Findings"},
                        delta={'reference': metrics.get('previous_findings', 0)}
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Research Metrics Dashboard",
                height=600,
                showlegend=False
            )
            
            # Save plot
            filename = f"metrics_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            
            return f"Research metrics dashboard saved to {filepath}"
            
        except Exception as e:
            return f"Metrics dashboard creation failed: {str(e)}"
    
    def create_word_frequency_chart(self, text: str, top_n: int = 20) -> str:
        """Create word frequency visualization from text"""
        try:
            # Clean and tokenize text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an'}
            words = [word for word in words if word not in stop_words]
            
            # Count frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top N words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            if not top_words:
                return "No words found for frequency analysis"
            
            # Create bar chart
            words_list, frequencies = zip(*top_words)
            
            fig = go.Figure(data=[
                go.Bar(x=list(words_list), y=list(frequencies), 
                       marker_color='skyblue')
            ])
            
            fig.update_layout(
                title=f"Top {len(top_words)} Word Frequencies",
                xaxis_title="Words",
                yaxis_title="Frequency",
                xaxis_tickangle=-45
            )
            
            # Save plot
            filename = f"word_frequency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            
            return f"Word frequency chart saved to {filepath}"
            
        except Exception as e:
            return f"Word frequency visualization failed: {str(e)}"

def get_visualization_tools():
    """Get data visualization tools"""
    
    viz_tool = DataVisualizationTool()
    
    def create_timeline_tool(data_json: str) -> str:
        """Create timeline visualization from JSON data"""
        try:
            data = json.loads(data_json)
            return viz_tool.create_research_timeline(data)
        except json.JSONDecodeError:
            return "Invalid JSON format. Please provide data as JSON string with 'date' and 'event' fields."
        except Exception as e:
            return f"Timeline creation failed: {str(e)}"
    
    def create_concept_network_tool(concepts_str: str) -> str:
        """Create concept network visualization"""
        try:
            concepts = [c.strip() for c in concepts_str.split(',')]
            return viz_tool.create_concept_network(concepts)
        except Exception as e:
            return f"Concept network creation failed: {str(e)}"
    
    def create_metrics_dashboard_tool(metrics_json: str) -> str:
        """Create research metrics dashboard"""
        try:
            metrics = json.loads(metrics_json)
            return viz_tool.create_research_metrics_dashboard(metrics)
        except json.JSONDecodeError:
            return "Invalid JSON format for metrics data."
        except Exception as e:
            return f"Metrics dashboard creation failed: {str(e)}"
    
    def create_word_frequency_tool(text: str) -> str:
        """Create word frequency visualization"""
        try:
            return viz_tool.create_word_frequency_chart(text)
        except Exception as e:
            return f"Word frequency visualization failed: {str(e)}"
    
    def analyze_research_data_tool(data_description: str) -> str:
        """Analyze and suggest visualizations for research data"""
        try:
            suggestions = []
            
            if 'timeline' in data_description.lower() or 'date' in data_description.lower():
                suggestions.append("📊 Timeline visualization - Use create_timeline for chronological data")
            
            if 'concept' in data_description.lower() or 'relationship' in data_description.lower():
                suggestions.append("🕸️ Concept network - Use create_concept_network for concept relationships")
            
            if 'metric' in data_description.lower() or 'statistic' in data_description.lower():
                suggestions.append("📈 Metrics dashboard - Use create_metrics_dashboard for quantitative data")
            
            if 'text' in data_description.lower() or 'document' in data_description.lower():
                suggestions.append("📝 Word frequency - Use create_word_frequency for text analysis")
            
            if not suggestions:
                suggestions = [
                    "📊 Consider timeline visualization for chronological data",
                    "🕸️ Consider concept network for relationships",
                    "📈 Consider metrics dashboard for quantitative data",
                    "📝 Consider word frequency for text analysis"
                ]
            
            return "Visualization Suggestions:\n" + "\n".join(suggestions)
            
        except Exception as e:
            return f"Data analysis failed: {str(e)}"
    
    return [
        Tool(
            name="create_timeline_visualization",
            description="Create timeline visualization from research data. Input should be JSON with 'date' and 'event' fields.",
            func=create_timeline_tool
        ),
        Tool(
            name="create_concept_network",
            description="Create network visualization of concepts. Input should be comma-separated list of concepts.",
            func=create_concept_network_tool
        ),
        Tool(
            name="create_metrics_dashboard",
            description="Create research metrics dashboard. Input should be JSON with metrics data.",
            func=create_metrics_dashboard_tool
        ),
        Tool(
            name="create_word_frequency_chart",
            description="Create word frequency visualization from text. Input should be the text to analyze.",
            func=create_word_frequency_tool
        ),
        Tool(
            name="analyze_research_data",
            description="Analyze research data and suggest appropriate visualizations. Input should be description of your data.",
            func=analyze_research_data_tool
        )
    ]
