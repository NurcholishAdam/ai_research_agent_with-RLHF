# -*- coding: utf-8 -*-
"""
RLHF Feedback Interface - Phase 6
Web interface for collecting human feedback on agent outputs
"""

import streamlit as st
import gradio as gr
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import pandas as pd

from rlhf.feedback_system import (
    get_feedback_collector, FeedbackType, FeedbackRating,
    AgentOutput, HumanFeedback
)
from rlhf.reward_model import get_reward_model_manager

class StreamlitFeedbackInterface:
    """Streamlit interface for RLHF feedback collection"""
    
    def __init__(self):
        self.feedback_collector = get_feedback_collector()
        self.reward_model_manager = get_reward_model_manager()
        
        # Initialize session state
        if 'feedback_session' not in st.session_state:
            st.session_state.feedback_session = {
                'current_outputs': [],
                'feedback_given': [],
                'comparison_pairs': []
            }
    
    def render_feedback_collection(self):
        """Render the main feedback collection interface"""
        
        st.title("🎯 RLHF Feedback Collection")
        st.markdown("Help improve the AI Research Agent by providing feedback on its outputs!")
        
        # Sidebar with statistics
        with st.sidebar:
            self.render_feedback_statistics()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Rate Outputs", 
            "⚖️ Compare Outputs", 
            "🏆 Reward Model", 
            "📊 Analytics"
        ])
        
        with tab1:
            self.render_output_rating()
        
        with tab2:
            self.render_output_comparison()
        
        with tab3:
            self.render_reward_model_interface()
        
        with tab4:
            self.render_feedback_analytics()
    
    def render_feedback_statistics(self):
        """Render feedback statistics in sidebar"""
        st.header("📊 Feedback Stats")
        
        try:
            stats = self.feedback_collector.get_feedback_statistics()
            
            st.metric("Total Outputs", stats.get('total_outputs', 0))
            st.metric("Feedback Given", stats.get('total_feedback', 0))
            st.metric("Training Pairs", stats.get('total_pairs', 0))
            st.metric("Collection Rate", f"{stats.get('collection_rate', 0):.2%}")
            
            # Rating distribution
            rating_stats = stats.get('rating_statistics', [])
            if rating_stats:
                st.subheader("Rating Distribution")
                for stat in rating_stats:
                    st.write(f"**{stat['_id']}**: {stat['avg_rating']:.2f} ({stat['count']} ratings)")
        
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    def render_output_rating(self):
        """Render interface for rating individual outputs"""
        st.header("📝 Rate Agent Outputs")
        
        # Get recent outputs for rating
        try:
            if self.feedback_collector.mongo_store:
                recent_outputs = self.feedback_collector.mongo_store.get_agent_outputs(limit=20)
            else:
                recent_outputs = self.feedback_collector.outputs_memory[-20:]
            
            if not recent_outputs:
                st.info("No agent outputs available for rating.")
                return
            
            # Select output to rate
            output_options = [
                f"{output.output_type.title()}: {output.research_question[:50]}..." 
                for output in recent_outputs
            ]
            
            selected_idx = st.selectbox(
                "Select output to rate:",
                range(len(output_options)),
                format_func=lambda x: output_options[x]
            )
            
            selected_output = recent_outputs[selected_idx]
            
            # Display output details
            st.subheader("Output Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {selected_output.output_type}")
                st.write(f"**Question:** {selected_output.research_question}")
                st.write(f"**Timestamp:** {selected_output.timestamp}")
            
            with col2:
                st.write(f"**Agent Confidence:** {selected_output.agent_confidence:.2f}")
                st.write(f"**Session ID:** {selected_output.session_id[:8]}...")
            
            # Display content
            st.subheader("Content")
            st.text_area("Output Content", selected_output.content, height=200, disabled=True)
            
            # Feedback form
            st.subheader("Provide Feedback")
            
            col1, col2 = st.columns(2)
            
            with col1:
                feedback_type = st.selectbox(
                    "Feedback Type:",
                    [ft.value for ft in FeedbackType],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                rating = st.selectbox(
                    "Rating:",
                    [fr.value for fr in FeedbackRating],
                    format_func=lambda x: f"{x} - {FeedbackRating(x).name.title()}"
                )
            
            with col2:
                user_id = st.text_input("Your ID (optional):", value="anonymous")
                comment = st.text_area("Comments (optional):", height=100)
            
            # Submit feedback
            if st.button("Submit Feedback", type="primary"):
                try:
                    feedback_id = self.feedback_collector.collect_human_feedback(
                        output_id=selected_output.id,
                        feedback_type=FeedbackType(feedback_type),
                        rating=FeedbackRating(rating),
                        comment=comment if comment else None,
                        user_id=user_id,
                        session_context={"interface": "streamlit"}
                    )
                    
                    st.success(f"Feedback submitted successfully! ID: {feedback_id[:8]}...")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")
        
        except Exception as e:
            st.error(f"Error loading outputs: {e}")
    
    def render_output_comparison(self):
        """Render interface for comparing outputs"""
        st.header("⚖️ Compare Agent Outputs")
        
        try:
            # Get outputs for comparison
            if self.feedback_collector.mongo_store:
                available_outputs = self.feedback_collector.mongo_store.get_agent_outputs(limit=50)
            else:
                available_outputs = self.feedback_collector.outputs_memory[-50:]
            
            if len(available_outputs) < 2:
                st.info("Need at least 2 outputs for comparison.")
                return
            
            # Filter by output type for fair comparison
            output_types = list(set(output.output_type for output in available_outputs))
            selected_type = st.selectbox("Compare outputs of type:", output_types)
            
            filtered_outputs = [o for o in available_outputs if o.output_type == selected_type]
            
            if len(filtered_outputs) < 2:
                st.info(f"Need at least 2 outputs of type '{selected_type}' for comparison.")
                return
            
            # Select two outputs to compare
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Output A")
                output_a_idx = st.selectbox(
                    "Select first output:",
                    range(len(filtered_outputs)),
                    format_func=lambda x: f"{filtered_outputs[x].research_question[:40]}..."
                )
                output_a = filtered_outputs[output_a_idx]
                
                st.write(f"**Question:** {output_a.research_question}")
                st.write(f"**Confidence:** {output_a.agent_confidence:.2f}")
                st.text_area("Content A", output_a.content, height=300, disabled=True, key="content_a")
            
            with col2:
                st.subheader("Output B")
                output_b_idx = st.selectbox(
                    "Select second output:",
                    range(len(filtered_outputs)),
                    format_func=lambda x: f"{filtered_outputs[x].research_question[:40]}...",
                    key="output_b_select"
                )
                output_b = filtered_outputs[output_b_idx]
                
                st.write(f"**Question:** {output_b.research_question}")
                st.write(f"**Confidence:** {output_b.agent_confidence:.2f}")
                st.text_area("Content B", output_b.content, height=300, disabled=True, key="content_b")
            
            # Comparison form
            st.subheader("Which output is better?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feedback_type = st.selectbox(
                    "Compare based on:",
                    [ft.value for ft in FeedbackType],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="compare_type"
                )
            
            with col2:
                preference = st.radio(
                    "Preference:",
                    ["A is better", "B is better", "They're equal"],
                    key="preference"
                )
            
            with col3:
                confidence = st.slider("Confidence in preference:", 0.0, 1.0, 0.8, 0.1)
                user_id = st.text_input("Your ID:", value="anonymous", key="compare_user")
            
            # Submit comparison
            if st.button("Submit Comparison", type="primary"):
                try:
                    preference_value = {
                        "A is better": "a",
                        "B is better": "b",
                        "They're equal": "tie"
                    }[preference]
                    
                    pair_id = self.feedback_collector.create_feedback_pair(
                        output_a_id=output_a.id,
                        output_b_id=output_b.id,
                        preference=preference_value,
                        feedback_type=FeedbackType(feedback_type),
                        confidence=confidence,
                        user_id=user_id
                    )
                    
                    st.success(f"Comparison submitted successfully! ID: {pair_id[:8]}...")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error submitting comparison: {e}")
        
        except Exception as e:
            st.error(f"Error loading outputs for comparison: {e}")
    
    def render_reward_model_interface(self):
        """Render reward model training and evaluation interface"""
        st.header("🏆 Reward Model Management")
        
        # Model statistics
        try:
            model_stats = self.reward_model_manager.get_model_statistics()
            
            if model_stats.get("status") != "no_training_history":
                st.subheader("Model Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    latest = model_stats.get("latest_performance", {})
                    st.metric("Training Pairs", latest.get("training_pairs", 0))
                
                with col2:
                    st.metric("Best Val Loss", f"{latest.get('best_val_loss', 0):.4f}")
                
                with col3:
                    st.metric("Training Sessions", model_stats.get("total_training_sessions", 0))
                
                # Training history
                if model_stats.get("training_history"):
                    st.subheader("Training History")
                    history_df = pd.DataFrame(model_stats["training_history"])
                    st.dataframe(history_df)
            
            else:
                st.info("No reward model training history available.")
        
        except Exception as e:
            st.error(f"Error loading model statistics: {e}")
        
        # Training controls
        st.subheader("Train Reward Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feedback_type_filter = st.selectbox(
                "Train on feedback type:",
                ["All"] + [ft.value for ft in FeedbackType],
                format_func=lambda x: x.replace('_', ' ').title() if x != "All" else x
            )
            
            min_pairs = st.number_input("Minimum training pairs:", min_value=10, value=50)
        
        with col2:
            if st.button("Start Training", type="primary"):
                try:
                    with st.spinner("Training reward model..."):
                        filter_type = None if feedback_type_filter == "All" else FeedbackType(feedback_type_filter)
                        
                        training_results = self.reward_model_manager.train_from_feedback(
                            feedback_type=filter_type,
                            min_pairs=min_pairs
                        )
                        
                        st.success("Training completed successfully!")
                        st.json(training_results["performance_record"])
                
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        # Model evaluation
        st.subheader("Test Reward Model")
        
        test_text = st.text_area("Enter text to evaluate:", height=150)
        test_context = st.text_input("Research question context (optional):")
        
        if st.button("Evaluate Text") and test_text:
            try:
                evaluation = self.reward_model_manager.evaluate_agent_output(test_text, test_context)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Reward Score", f"{evaluation.get('normalized_score', 0):.3f}")
                
                with col2:
                    st.metric("Quality", evaluation.get('quality_assessment', 'unknown').title())
                
                with col3:
                    st.metric("Confidence", f"{evaluation.get('model_confidence', 0):.3f}")
                
                if evaluation.get('error'):
                    st.error(evaluation['error'])
            
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
    
    def render_feedback_analytics(self):
        """Render feedback analytics and insights"""
        st.header("📊 Feedback Analytics")
        
        try:
            stats = self.feedback_collector.get_feedback_statistics()
            
            # Overall metrics
            st.subheader("Overall Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Outputs", stats.get('total_outputs', 0))
            
            with col2:
                st.metric("Feedback Collected", stats.get('total_feedback', 0))
            
            with col3:
                st.metric("Comparison Pairs", stats.get('total_pairs', 0))
            
            with col4:
                collection_rate = stats.get('collection_rate', 0)
                st.metric("Collection Rate", f"{collection_rate:.1%}")
            
            # Rating analysis
            rating_stats = stats.get('rating_statistics', [])
            if rating_stats:
                st.subheader("Rating Analysis by Feedback Type")
                
                # Convert to DataFrame for better visualization
                df = pd.DataFrame(rating_stats)
                df.columns = ['Feedback Type', 'Average Rating', 'Count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(df)
                
                with col2:
                    # Simple bar chart
                    st.bar_chart(df.set_index('Feedback Type')['Average Rating'])
            
            # Recent activity
            st.subheader("Recent Activity")
            
            if self.feedback_collector.mongo_store:
                # Get recent feedback
                recent_feedback = self.feedback_collector.feedback_memory[-10:]  # Last 10
                
                if recent_feedback:
                    feedback_data = []
                    for feedback in recent_feedback:
                        feedback_data.append({
                            'Timestamp': feedback.timestamp.strftime('%Y-%m-%d %H:%M'),
                            'Type': feedback.feedback_type.value,
                            'Rating': feedback.rating.value,
                            'User': feedback.user_id
                        })
                    
                    recent_df = pd.DataFrame(feedback_data)
                    st.dataframe(recent_df)
                else:
                    st.info("No recent feedback activity.")
            else:
                st.info("Connect to MongoDB to see detailed analytics.")
        
        except Exception as e:
            st.error(f"Error loading analytics: {e}")

def create_streamlit_feedback_app():
    """Create Streamlit feedback collection app"""
    
    st.set_page_config(
        page_title="RLHF Feedback Collection",
        page_icon="🎯",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feedback-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run interface
    interface = StreamlitFeedbackInterface()
    interface.render_feedback_collection()

if __name__ == "__main__":
    create_streamlit_feedback_app()
