# -*- coding: utf-8 -*-
"""
Enhanced Research Agent with ReAct pattern and proper state management
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from llm.groq_wrapper import load_groq_llm
from memory.langmem_tools import get_memory_tools
from memory.advanced_memory_manager import get_advanced_memory_tools
from tools.research_tools_manager import get_all_research_tools
from agent.multi_agent_system import get_multi_agent_system
from agent.hypothesis_engine import get_hypothesis_engine
# Phase 7: Contextual Engineering Framework
from context_engineering import ContextOrchestrator, ResearchContext, OrchestrationStrategy, OrchestrationConfig
# RLHF Integration - Phase 6
try:
    from rlhf.feedback_system import FeedbackCollector, FeedbackType, FeedbackRating
    from rlhf.reward_model import RewardModelManager
    RLHF_AVAILABLE = True
except ImportError:
    # Fallback to simplified system for testing
    try:
        from rlhf.simple_feedback_system import (
            SimpleFeedbackCollector as FeedbackCollector,
            SimpleRewardModelManager as RewardModelManager,
            FeedbackType, FeedbackRating
        )
        RLHF_AVAILABLE = True
    except ImportError:
        # No RLHF system available
        FeedbackCollector = None
        RewardModelManager = None
        FeedbackType = None
        FeedbackRating = None
        RLHF_AVAILABLE = False
    )
import json

class AgentState(TypedDict):
    """State for the research agent - Enhanced for Phase 6 RLHF"""
    messages: List[BaseMessage]
    research_question: str
    research_plan: List[str]
    current_step: int
    findings: List[Dict[str, Any]]
    final_answer: str
    iteration_count: int
    # Phase 4 Intelligence Layer additions
    hypotheses: List[Dict[str, Any]]
    multi_agent_analysis: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    intelligence_insights: Dict[str, Any]
    # Phase 6 RLHF additions
    session_id: str
    rlhf_feedback: Dict[str, Any]
    reward_scores: Dict[str, float]
    # Phase 7 Contextual Engineering additions
    context_orchestration: Dict[str, Any]
    research_context: Dict[str, Any]

class ResearchAgent:
    def __init__(self):
        self.llm = load_groq_llm()
        self.memory_tools = get_memory_tools()
        self.advanced_memory_tools = get_advanced_memory_tools()
        self.research_tools = get_all_research_tools()
        
        # Phase 4: Intelligence Layer components
        self.multi_agent_system = get_multi_agent_system()
        self.hypothesis_engine = get_hypothesis_engine()
        
        # Phase 7: Contextual Engineering Framework
        try:
            context_config = OrchestrationConfig(
                strategy=OrchestrationStrategy.BALANCED,
                max_context_items=15,
                quality_threshold=0.7,
                processing_timeout=30,
                enable_caching=True,
                parallel_processing=True,
                optimization_goals=["quality", "relevance", "efficiency"]
            )
            self.context_orchestrator = ContextOrchestrator(context_config)
            self.context_engineering_enabled = True
            print("   🎼 Contextual Engineering Framework initialized")
        except Exception as e:
            print(f"   ⚠️ Contextual Engineering initialization failed: {e}")
            self.context_orchestrator = None
            self.context_engineering_enabled = False
        
        # Phase 6: RLHF components
        try:
            self.feedback_collector = FeedbackCollector()
            self.reward_model_manager = RewardModelManager()
            self.rlhf_enabled = True
            print("   🎯 RLHF system initialized successfully")
        except Exception as e:
            print(f"   ⚠️ RLHF system initialization failed: {e}")
            self.feedback_collector = None
            self.reward_model_manager = None
            self.rlhf_enabled = False
        
        # Combine all tool systems - Complete Arsenal!
        all_tools = (
            self.memory_tools + 
            self.advanced_memory_tools + 
            self.research_tools
        )
        self.tool_executor = ToolExecutor(all_tools)
        self.max_iterations = 10
        self.session_started = False
        
        print(f"🛠️ Research Agent initialized with {len(all_tools)} tools!")
        print("   📚 Memory tools, 🧠 Advanced memory, 🔬 Research arsenal")
        print("   🤖 Multi-agent collaboration, 🔬 Hypothesis engine")
        if self.rlhf_enabled:
            print("   🎯 RLHF feedback collection, 🏆 Reward model integration")
        else:
            print("   ⚠️ RLHF system disabled (optional dependency)")
        
    def create_research_plan(self, state: AgentState) -> AgentState:
        """Create a structured research plan with Contextual Engineering and RLHF feedback capture"""
        question = state["research_question"]
        session_id = state.get("session_id", "default_session")
        
        # Phase 7: Contextual Engineering Integration
        if self.context_engineering_enabled and self.context_orchestrator:
            try:
                # Create research context for orchestration
                research_context = ResearchContext(
                    question=question,
                    domain_hints=[],  # Could be extracted from question analysis
                    complexity_level="medium",  # Could be determined automatically
                    time_constraints=None,
                    quality_requirements=0.8,
                    user_preferences={}
                )
                
                # Orchestrate context engineering
                orchestration_result = self.context_orchestrator.orchestrate_research_context(
                    research_context=research_context
                )
                
                # Store orchestration results in state
                state["context_orchestration"] = {
                    "session_id": orchestration_result.session_id,
                    "context_items_count": len(orchestration_result.context_items),
                    "processed_items_count": len(orchestration_result.processed_context.processed_items),
                    "quality_score": orchestration_result.processed_context.quality_score,
                    "tool_recommendations": [rec.tool_name for rec in orchestration_result.tool_recommendations],
                    "execution_time": orchestration_result.execution_time
                }
                
                state["research_context"] = {
                    "question": question,
                    "complexity_level": research_context.complexity_level,
                    "quality_requirements": research_context.quality_requirements,
                    "orchestration_metadata": orchestration_result.orchestration_metadata
                }
                
                print(f"🎼 Context orchestration complete:")
                print(f"   Quality: {orchestration_result.processed_context.quality_score:.3f}")
                print(f"   Tools recommended: {len(orchestration_result.tool_recommendations)}")
                print(f"   Execution time: {orchestration_result.execution_time:.2f}s")
                
            except Exception as e:
                print(f"⚠️ Context orchestration failed: {e}")
                state["context_orchestration"] = {"error": str(e)}
                state["research_context"] = {"question": question}
        else:
            state["context_orchestration"] = {"disabled": True}
            state["research_context"] = {"question": question}
        
        # Start research session in advanced memory
        if not self.session_started:
            session_result = self.tool_executor.invoke({
                "tool": "start_research_session",
                "tool_input": question
            })
            self.session_started = True
            print(f"🧠 {session_result}")
        
        # Search existing memory for related research
        memory_search = self.tool_executor.invoke({
            "tool": "search_advanced_memory",
            "tool_input": question
        })
        
        planning_prompt = f"""
        You are a research planning expert. Given this research question: "{question}"
        
        Previous related research from memory:
        {memory_search}
        
        Create a structured research plan with 3-5 specific steps. Each step should be:
        1. Actionable and specific
        2. Build upon previous steps and existing knowledge
        3. Lead toward answering the main question
        4. Avoid duplicating what we already know
        
        Format your response as a JSON list of strings:
        ["Step 1: ...", "Step 2: ...", "Step 3: ..."]
        """
        
        response = self.llm.invoke([HumanMessage(content=planning_prompt)])
        
        try:
            # Extract JSON from response
            plan_text = response.content
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0]
            elif "[" in plan_text and "]" in plan_text:
                start = plan_text.find("[")
                end = plan_text.rfind("]") + 1
                plan_text = plan_text[start:end]
            
            research_plan = json.loads(plan_text)
        except:
            # Fallback plan
            research_plan = [
                f"Search for background information about: {question}",
                f"Identify key concepts and definitions related to: {question}",
                f"Find recent developments or research about: {question}",
                f"Synthesize findings to answer: {question}"
            ]
        
        state["research_plan"] = research_plan
        state["current_step"] = 0
        state["findings"] = []
        
        # Phase 6: RLHF Integration - Capture research plan
        if self.rlhf_enabled and self.feedback_collector:
            try:
                plan_content = "\n".join([f"{i+1}. {step}" for i, step in enumerate(research_plan)])
                
                # Capture research plan for RLHF
                plan_output = self.feedback_collector.capture_research_output(
                    research_result={
                        'research_plan': research_plan,
                        'plan_content': plan_content,
                        'step_count': len(research_plan)
                    },
                    research_question=question,
                    session_id=session_id
                )
                
                # Store for potential feedback collection
                if "rlhf_feedback" not in state:
                    state["rlhf_feedback"] = {}
                state["rlhf_feedback"]["plan_output_id"] = plan_output.id
                
                print(f"🎯 Research plan captured for RLHF: {plan_output.id[:8]}...")
                
                # Evaluate plan with reward model if available
                if self.reward_model_manager:
                    try:
                        plan_evaluation = self.reward_model_manager.evaluate_agent_output(
                            plan_content, question
                        )
                        if "reward_scores" not in state:
                            state["reward_scores"] = {}
                        state["reward_scores"]["plan"] = plan_evaluation.get("normalized_score", 0.5)
                        print(f"🏆 Plan reward score: {plan_evaluation.get('normalized_score', 0.5):.3f}")
                    except Exception as e:
                        print(f"⚠️ Reward model evaluation failed: {e}")
                        
            except Exception as e:
                print(f"⚠️ RLHF plan capture failed: {e}")
        
        return state
    
    def execute_research_step(self, state: AgentState) -> AgentState:
        """Execute the current research step with Phase 3 Research Tools Arsenal"""
        if state["current_step"] >= len(state["research_plan"]):
            return state
            
        current_step = state["research_plan"][state["current_step"]]
        
        # Phase 3: Get tool suggestions for this research step
        tool_suggestions = self.tool_executor.invoke({
            "tool": "suggest_research_tools",
            "tool_input": current_step
        })
        print(f"🛠️ {tool_suggestions}")
        
        # Search memory systems first
        basic_search = self.tool_executor.invoke({
            "tool": "search_memory",
            "tool_input": {"query": current_step}
        })
        
        advanced_search = self.tool_executor.invoke({
            "tool": "search_advanced_memory",
            "tool_input": current_step
        })
        
        # Phase 3: Execute external research based on step content
        external_research_results = []
        
        # Intelligent tool selection based on research step
        step_lower = current_step.lower()
        
        # Web search for general information
        if any(keyword in step_lower for keyword in ['search', 'find', 'information', 'about']):
            try:
                web_result = self.tool_executor.invoke({
                    "tool": "web_search",
                    "tool_input": current_step
                })
                external_research_results.append(f"Web Search: {web_result}")
                print("🌐 Web search completed")
            except Exception as e:
                print(f"⚠️ Web search failed: {e}")
        
        # Wikipedia for background/definitions
        if any(keyword in step_lower for keyword in ['background', 'definition', 'what is', 'explain']):
            try:
                wiki_result = self.tool_executor.invoke({
                    "tool": "wikipedia_search",
                    "tool_input": current_step
                })
                external_research_results.append(f"Wikipedia: {wiki_result}")
                print("📚 Wikipedia search completed")
            except Exception as e:
                print(f"⚠️ Wikipedia search failed: {e}")
        
        # arXiv for academic/research content
        if any(keyword in step_lower for keyword in ['research', 'academic', 'study', 'paper', 'scientific']):
            try:
                arxiv_result = self.tool_executor.invoke({
                    "tool": "arxiv_search",
                    "tool_input": current_step
                })
                external_research_results.append(f"arXiv: {arxiv_result}")
                print("🎓 arXiv search completed")
            except Exception as e:
                print(f"⚠️ arXiv search failed: {e}")
        
        # News search for current/recent information
        if any(keyword in step_lower for keyword in ['recent', 'current', 'latest', 'news', 'developments']):
            try:
                news_result = self.tool_executor.invoke({
                    "tool": "news_search",
                    "tool_input": current_step
                })
                external_research_results.append(f"News: {news_result}")
                print("📰 News search completed")
            except Exception as e:
                print(f"⚠️ News search failed: {e}")
        
        # Comprehensive analysis with all sources
        analysis_prompt = f"""
        Research Step: {current_step}
        
        MEMORY SOURCES:
        Basic Memory: {basic_search}
        Advanced Memory: {advanced_search}
        
        EXTERNAL RESEARCH SOURCES:
        {chr(10).join(external_research_results)}
        
        Previous Findings: {state["findings"]}
        
        Based on ALL sources (memory + external research):
        1. What key information did we discover from each source?
        2. What new concepts or relationships emerged?
        3. How do external sources complement or contradict memory?
        4. What gaps still exist after this comprehensive search?
        5. What hypotheses can we form?
        6. What are the most reliable sources and findings?
        
        Provide a structured analysis in this format:
        KEY_FINDINGS: [list key discoveries with source attribution]
        NEW_CONCEPTS: [list new concepts identified]
        SOURCE_ANALYSIS: [compare reliability and consistency of sources]
        HYPOTHESES: [list any hypotheses formed]
        GAPS: [list what's still missing]
        RESEARCH_FINDING: [concise summary to save to advanced memory with citations]
        """
        
        analysis = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Save comprehensive research finding to advanced memory
        if "RESEARCH_FINDING:" in analysis.content:
            finding_content = analysis.content.split("RESEARCH_FINDING:")[1].strip()
            if finding_content and finding_content != "[concise summary to save to advanced memory with citations]":
                save_result = self.tool_executor.invoke({
                    "tool": "save_research_finding",
                    "tool_input": f"Step {state['current_step'] + 1} - {current_step}: {finding_content}"
                })
                print(f"💾 {save_result}")
        
        # Save to basic memory for backward compatibility
        if "KEY_FINDINGS:" in analysis.content:
            key_findings = analysis.content.split("KEY_FINDINGS:")[1].split("NEW_CONCEPTS:")[0].strip()
            if key_findings and key_findings != "[list key discoveries with source attribution]":
                self.tool_executor.invoke({
                    "tool": "save_memory",
                    "tool_input": f"Research Step {state['current_step'] + 1}: {key_findings}"
                })
        
        # Record comprehensive findings with Phase 3 metadata
        finding = {
            "step": state["current_step"],
            "step_description": current_step,
            "analysis": analysis.content,
            "timestamp": "current",
            "sources_used": {
                "memory_basic": len(str(basic_search)),
                "memory_advanced": len(str(advanced_search)),
                "external_sources": len(external_research_results)
            },
            "external_research": external_research_results[:2]  # Store first 2 for reference
        }
        
        state["findings"].append(finding)
        state["current_step"] += 1
        
        return state
    
    def intelligence_analysis(self, state: AgentState) -> AgentState:
        """Phase 4: Multi-agent intelligence analysis and hypothesis generation"""
        print("🧠 Starting Intelligence Layer Analysis...")
        
        # Initialize Phase 4 state components
        if "hypotheses" not in state:
            state["hypotheses"] = []
        if "multi_agent_analysis" not in state:
            state["multi_agent_analysis"] = {}
        if "quality_assessment" not in state:
            state["quality_assessment"] = {}
        if "intelligence_insights" not in state:
            state["intelligence_insights"] = {}
        
        # Step 1: Multi-Agent Collaborative Analysis
        print("🤖 Multi-agent collaborative analysis...")
        try:
            multi_agent_results = self.multi_agent_system.collaborative_research_analysis(
                state["research_question"],
                state["research_plan"],
                state["findings"]
            )
            state["multi_agent_analysis"] = multi_agent_results
            print("✅ Multi-agent analysis completed")
        except Exception as e:
            print(f"⚠️ Multi-agent analysis failed: {e}")
            state["multi_agent_analysis"] = {"error": str(e)}
        
        # Step 2: Hypothesis Generation and Testing
        print("🔬 Generating and testing hypotheses...")
        try:
            # Generate hypotheses
            hypotheses = self.hypothesis_engine["generator"].generate_hypotheses(
                state["research_question"],
                state["findings"],
                max_hypotheses=3
            )
            
            # Test hypotheses
            if hypotheses:
                hypothesis_comparison = self.hypothesis_engine["tester"].compare_hypotheses(
                    hypotheses,
                    state["findings"]
                )
                
                state["hypotheses"] = [
                    {
                        "statement": hyp.statement,
                        "type": hyp.type.value,
                        "confidence": hyp.confidence,
                        "supporting_evidence": hyp.supporting_evidence,
                        "predictions": hyp.predictions
                    }
                    for hyp in hypotheses
                ]
                
                state["intelligence_insights"]["hypothesis_analysis"] = hypothesis_comparison
                print(f"✅ Generated and tested {len(hypotheses)} hypotheses")
            else:
                print("ℹ️ No hypotheses generated")
                
        except Exception as e:
            print(f"⚠️ Hypothesis generation failed: {e}")
            state["intelligence_insights"]["hypothesis_error"] = str(e)
        
        # Step 3: Quality Assessment
        print("📊 Performing quality assessment...")
        try:
            quality_metrics = self._assess_research_quality(state)
            state["quality_assessment"] = quality_metrics
            print("✅ Quality assessment completed")
        except Exception as e:
            print(f"⚠️ Quality assessment failed: {e}")
            state["quality_assessment"] = {"error": str(e)}
        
        return state
    
    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Phase 6 Enhanced: Synthesize with RLHF feedback capture"""
        print("🎯 Synthesizing final answer with intelligence insights and RLHF...")
        
        # Gather intelligence insights
        multi_agent_insights = state.get("multi_agent_analysis", {})
        hypotheses_info = state.get("hypotheses", [])
        quality_info = state.get("quality_assessment", {})
        session_id = state.get("session_id", "default_session")
        question = state["research_question"]
        
        # Enhanced synthesis prompt with intelligence layer
        synthesis_prompt = f"""
        Research Question: {question}
        
        Research Plan Executed:
        {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(state["research_plan"])])}
        
        Research Findings:
        {chr(10).join([f"Step {f['step']+1}: {f['analysis'][:300]}..." for f in state["findings"]])}
        
        INTELLIGENCE LAYER INSIGHTS:
        
        Multi-Agent Analysis:
        {multi_agent_insights.get('multi_agent_analysis', {}).get('synthesis', 'No multi-agent synthesis available')}
        
        Generated Hypotheses:
        {chr(10).join([f"- {hyp['statement']} (confidence: {hyp['confidence']:.2f})" for hyp in hypotheses_info[:3]])}
        
        Quality Assessment:
        Research Quality Score: {quality_info.get('overall_quality_score', 'N/A')}
        Confidence Level: {quality_info.get('confidence_assessment', 'N/A')}
        
        Based on ALL research conducted AND intelligence layer analysis, provide a comprehensive answer:
        1. Direct answer to the question
        2. Key supporting evidence with source reliability assessment
        3. Most supported hypothesis (if any)
        4. Confidence level with multi-agent validation
        5. Limitations and areas needing further research
        6. Research methodology assessment
        
        Structure your response with clear sections and evidence-based conclusions.
        """
        
        final_response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        state["final_answer"] = final_response.content
        
        # Phase 6: RLHF Integration - Capture final answer for feedback
        if self.rlhf_enabled and self.feedback_collector:
            try:
                # Capture complete research output for RLHF
                research_output = self.feedback_collector.capture_research_output(
                    research_result={
                        'final_answer': final_response.content,
                        'research_plan': state.get('research_plan', []),
                        'findings': state.get('findings', []),
                        'hypotheses': hypotheses_info,
                        'quality_assessment': quality_info,
                        'multi_agent_analysis': multi_agent_insights,
                        'research_steps': len(state.get("findings", [])),
                        'hypotheses_count': len(hypotheses_info)
                    },
                    research_question=question,
                    session_id=session_id
                )
                
                # Store for potential feedback collection
                if "rlhf_feedback" not in state:
                    state["rlhf_feedback"] = {}
                state["rlhf_feedback"]["final_output_id"] = research_output.id
                
                print(f"🎯 Final research output captured for RLHF: {research_output.id[:8]}...")
                
                # Evaluate with reward model if available
                if self.reward_model_manager:
                    try:
                        evaluation = self.reward_model_manager.evaluate_agent_output(
                            final_response.content, question
                        )
                        if "reward_scores" not in state:
                            state["reward_scores"] = {}
                        state["reward_scores"]["final_answer"] = evaluation.get("normalized_score", 0.5)
                        print(f"🏆 Final answer reward score: {evaluation.get('normalized_score', 0.5):.3f}")
                        
                        # Store detailed evaluation
                        state["rlhf_feedback"]["final_answer_evaluation"] = evaluation
                        
                    except Exception as e:
                        print(f"⚠️ Reward model evaluation failed: {e}")
                
            except Exception as e:
                print(f"⚠️ RLHF final answer capture failed: {e}")
        
        # End research session in advanced memory
        if self.session_started:
            session_end_result = self.tool_executor.invoke({
                "tool": "end_research_session",
                "tool_input": final_response.content
            })
            print(f"🎯 {session_end_result}")
            self.session_started = False
        
        return state
    
    def _assess_research_quality(self, state: AgentState) -> Dict[str, Any]:
        """Assess overall research quality and reliability"""
        
        findings = state.get("findings", [])
        multi_agent_analysis = state.get("multi_agent_analysis", {})
        
        # Calculate quality metrics
        total_findings = len(findings)
        external_sources_used = sum(1 for f in findings if f.get("external_research"))
        
        # Multi-agent confidence scores
        confidence_scores = multi_agent_analysis.get("confidence_scores", {})
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
        
        # Source diversity
        source_types = set()
        for finding in findings:
            external_research = finding.get("external_research", [])
            for research in external_research:
                if "Web Search:" in research:
                    source_types.add("web")
                elif "Wikipedia:" in research:
                    source_types.add("wikipedia")
                elif "arXiv:" in research:
                    source_types.add("academic")
                elif "News:" in research:
                    source_types.add("news")
        
        source_diversity = len(source_types)
        
        # Overall quality score (0-10)
        quality_score = min(10, (
            (total_findings / 5) * 2 +  # Completeness (max 2 points)
            (external_sources_used / total_findings if total_findings > 0 else 0) * 3 +  # External validation (max 3 points)
            avg_confidence * 3 +  # Confidence (max 3 points)
            min(source_diversity / 4, 1) * 2  # Source diversity (max 2 points)
        ))
        
        return {
            "overall_quality_score": round(quality_score, 2),
            "total_findings": total_findings,
            "external_sources_used": external_sources_used,
            "source_diversity": source_diversity,
            "confidence_assessment": avg_confidence,
            "quality_indicators": {
                "comprehensive": total_findings >= 3,
                "well_sourced": external_sources_used >= total_findings * 0.5,
                "high_confidence": avg_confidence >= 0.7,
                "diverse_sources": source_diversity >= 2
            }
        }
    
    def should_continue(self, state: AgentState) -> str:
        """Determine if research should continue - Phase 4 Enhanced"""
        if state["current_step"] >= len(state["research_plan"]):
            return "intelligence"  # Go to intelligence analysis before synthesis
        elif state["iteration_count"] >= self.max_iterations:
            return "intelligence"
        else:
            return "continue"
    
    def build_graph(self) -> StateGraph:
        """Build the research agent graph - Phase 4 Enhanced with Intelligence Layer"""
        workflow = StateGraph(AgentState)
        
        # Add nodes - Phase 4 Complete Workflow
        workflow.add_node("plan", self.create_research_plan)
        workflow.add_node("research", self.execute_research_step)
        workflow.add_node("intelligence", self.intelligence_analysis)  # Phase 4: Intelligence Layer
        workflow.add_node("synthesize", self.synthesize_answer)
        
        # Add edges - Phase 4 Enhanced Flow
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "research")
        workflow.add_conditional_edges(
            "research",
            self.should_continue,
            {
                "continue": "research",
                "intelligence": "intelligence"  # Phase 4: Route to intelligence analysis
            }
        )
        workflow.add_edge("intelligence", "synthesize")  # Phase 4: Intelligence -> Synthesis
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()

def create_agent():
    """Create and return a compiled research agent"""
    agent = ResearchAgent()
    return agent.build_graph()