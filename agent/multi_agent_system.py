# -*- coding: utf-8 -*-
"""
Multi-Agent System for AI Research Agent - Phase 4 Intelligence Layer
Implements researcher, critic, and synthesizer agents for collaborative intelligence
"""

from typing import Dict, List, Any, Optional, TypedDict
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from llm.groq_wrapper import load_groq_llm
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class AgentRole(Enum):
    RESEARCHER = "researcher"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    METHODOLOGIST = "methodologist"

@dataclass
class AgentResponse:
    """Response from an individual agent"""
    agent_role: AgentRole
    content: str
    confidence: float
    reasoning: str
    suggestions: List[str]
    timestamp: datetime

class ResearcherAgent:
    """Primary research agent focused on information gathering"""
    
    def __init__(self):
        self.llm = load_groq_llm()
        self.role = AgentRole.RESEARCHER
        
    def analyze_research_question(self, question: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Analyze research question and provide research strategy"""
        
        prompt = f"""
        You are an expert research agent. Analyze this research question: "{question}"
        
        Context: {context or "No additional context"}
        
        Provide a comprehensive analysis including:
        1. Research complexity assessment (1-10)
        2. Key research domains involved
        3. Potential challenges and limitations
        4. Recommended research methodology
        5. Expected information sources needed
        6. Time/effort estimation
        
        Format your response as:
        COMPLEXITY: [1-10 score with justification]
        DOMAINS: [list key research domains]
        CHALLENGES: [potential obstacles and limitations]
        METHODOLOGY: [recommended approach]
        SOURCES: [types of sources needed]
        ESTIMATION: [time/effort assessment]
        CONFIDENCE: [your confidence in this analysis 0-1]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Extract confidence score
        confidence = 0.7  # Default
        if "CONFIDENCE:" in response.content:
            try:
                conf_text = response.content.split("CONFIDENCE:")[1].strip().split()[0]
                confidence = float(conf_text)
            except:
                pass
        
        return AgentResponse(
            agent_role=self.role,
            content=response.content,
            confidence=confidence,
            reasoning="Research question analysis based on complexity, domains, and methodology",
            suggestions=self._extract_suggestions(response.content),
            timestamp=datetime.now()
        )
    
    def evaluate_findings(self, findings: List[Dict[str, Any]], question: str) -> AgentResponse:
        """Evaluate research findings for completeness and relevance"""
        
        findings_summary = "\n".join([
            f"Finding {i+1}: {finding.get('analysis', str(finding))[:200]}..."
            for i, finding in enumerate(findings[:5])
        ])
        
        prompt = f"""
        You are evaluating research findings for the question: "{question}"
        
        Current findings:
        {findings_summary}
        
        Evaluate these findings on:
        1. Completeness - Do they fully address the research question?
        2. Relevance - How relevant are they to the core question?
        3. Quality - Are the sources reliable and findings well-supported?
        4. Gaps - What important aspects are missing?
        5. Contradictions - Are there any conflicting findings?
        
        Provide assessment:
        COMPLETENESS: [score 1-10 with explanation]
        RELEVANCE: [score 1-10 with explanation]
        QUALITY: [score 1-10 with explanation]
        GAPS: [list missing elements]
        CONTRADICTIONS: [any conflicts found]
        RECOMMENDATIONS: [suggestions for improvement]
        CONFIDENCE: [your confidence 0-1]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        confidence = self._extract_confidence(response.content)
        
        return AgentResponse(
            agent_role=self.role,
            content=response.content,
            confidence=confidence,
            reasoning="Evaluation of research findings completeness and quality",
            suggestions=self._extract_recommendations(response.content),
            timestamp=datetime.now()
        )
    
    def _extract_suggestions(self, content: str) -> List[str]:
        """Extract suggestions from response content"""
        suggestions = []
        if "METHODOLOGY:" in content:
            method_text = content.split("METHODOLOGY:")[1].split("SOURCES:")[0].strip()
            suggestions.append(f"Methodology: {method_text[:100]}...")
        return suggestions
    
    def _extract_confidence(self, content: str) -> float:
        """Extract confidence score from content"""
        if "CONFIDENCE:" in content:
            try:
                conf_text = content.split("CONFIDENCE:")[1].strip().split()[0]
                return float(conf_text)
            except:
                pass
        return 0.7
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from content"""
        if "RECOMMENDATIONS:" in content:
            rec_text = content.split("RECOMMENDATIONS:")[1].strip()
            return [rec.strip() for rec in rec_text.split('\n')[:3] if rec.strip()]
        return []

class CriticAgent:
    """Critical analysis agent for quality assessment and fact-checking"""
    
    def __init__(self):
        self.llm = load_groq_llm()
        self.role = AgentRole.CRITIC
        
    def critique_research_plan(self, plan: List[str], question: str) -> AgentResponse:
        """Critically analyze research plan for potential issues"""
        
        plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])
        
        prompt = f"""
        You are a critical research analyst. Evaluate this research plan for: "{question}"
        
        Research Plan:
        {plan_text}
        
        Critically assess:
        1. Logical flow - Do steps build upon each other logically?
        2. Comprehensiveness - Are all important aspects covered?
        3. Efficiency - Is the approach efficient or redundant?
        4. Bias potential - Could this approach introduce bias?
        5. Feasibility - Are all steps practically achievable?
        6. Alternative approaches - What other methods could be better?
        
        Provide critical analysis:
        LOGICAL_FLOW: [assessment with issues identified]
        COMPREHENSIVENESS: [gaps or missing elements]
        EFFICIENCY: [redundancies or inefficiencies]
        BIAS_POTENTIAL: [potential sources of bias]
        FEASIBILITY: [practical concerns]
        ALTERNATIVES: [suggest better approaches]
        OVERALL_RATING: [1-10 with justification]
        CONFIDENCE: [your confidence 0-1]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        confidence = self._extract_confidence(response.content)
        
        return AgentResponse(
            agent_role=self.role,
            content=response.content,
            confidence=confidence,
            reasoning="Critical analysis of research plan structure and approach",
            suggestions=self._extract_alternatives(response.content),
            timestamp=datetime.now()
        )
    
    def fact_check_findings(self, findings: List[Dict[str, Any]]) -> AgentResponse:
        """Perform fact-checking and credibility assessment"""
        
        findings_text = "\n".join([
            f"Finding {i+1}: {finding.get('analysis', str(finding))[:300]}..."
            for i, finding in enumerate(findings[:3])
        ])
        
        prompt = f"""
        You are a fact-checking expert. Analyze these research findings for accuracy and credibility:
        
        {findings_text}
        
        Evaluate each finding for:
        1. Source credibility - Are sources reliable and authoritative?
        2. Claim verification - Can claims be independently verified?
        3. Logical consistency - Are arguments logically sound?
        4. Evidence quality - Is evidence sufficient and appropriate?
        5. Potential misinformation - Any red flags for false information?
        
        Provide fact-check assessment:
        SOURCE_CREDIBILITY: [analysis of source reliability]
        CLAIM_VERIFICATION: [verifiability of key claims]
        LOGICAL_CONSISTENCY: [logical soundness assessment]
        EVIDENCE_QUALITY: [strength of supporting evidence]
        MISINFORMATION_RISK: [potential false information identified]
        CREDIBILITY_SCORE: [overall score 1-10]
        CONFIDENCE: [your confidence 0-1]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        confidence = self._extract_confidence(response.content)
        
        return AgentResponse(
            agent_role=self.role,
            content=response.content,
            confidence=confidence,
            reasoning="Fact-checking and credibility assessment of research findings",
            suggestions=self._extract_credibility_concerns(response.content),
            timestamp=datetime.now()
        )
    
    def _extract_confidence(self, content: str) -> float:
        """Extract confidence score from content"""
        if "CONFIDENCE:" in content:
            try:
                conf_text = content.split("CONFIDENCE:")[1].strip().split()[0]
                return float(conf_text)
            except:
                pass
        return 0.8
    
    def _extract_alternatives(self, content: str) -> List[str]:
        """Extract alternative suggestions"""
        if "ALTERNATIVES:" in content:
            alt_text = content.split("ALTERNATIVES:")[1].strip()
            return [alt.strip() for alt in alt_text.split('\n')[:3] if alt.strip()]
        return []
    
    def _extract_credibility_concerns(self, content: str) -> List[str]:
        """Extract credibility concerns"""
        concerns = []
        if "MISINFORMATION_RISK:" in content:
            risk_text = content.split("MISINFORMATION_RISK:")[1].strip()
            concerns.append(f"Misinformation risk: {risk_text[:100]}...")
        return concerns

class SynthesizerAgent:
    """Synthesis agent for combining insights and generating final answers"""
    
    def __init__(self):
        self.llm = load_groq_llm()
        self.role = AgentRole.SYNTHESIZER
        
    def synthesize_multi_agent_insights(self, 
                                      researcher_responses: List[AgentResponse],
                                      critic_responses: List[AgentResponse],
                                      question: str) -> AgentResponse:
        """Synthesize insights from multiple agents"""
        
        researcher_insights = "\n".join([
            f"Researcher Analysis {i+1}: {resp.content[:200]}..."
            for i, resp in enumerate(researcher_responses)
        ])
        
        critic_insights = "\n".join([
            f"Critic Analysis {i+1}: {resp.content[:200]}..."
            for i, resp in enumerate(critic_responses)
        ])
        
        prompt = f"""
        You are a synthesis expert combining insights from multiple research agents for: "{question}"
        
        RESEARCHER INSIGHTS:
        {researcher_insights}
        
        CRITIC INSIGHTS:
        {critic_insights}
        
        Synthesize these perspectives to provide:
        1. Consensus findings - What do all agents agree on?
        2. Conflicting views - Where do agents disagree?
        3. Confidence assessment - Overall confidence in findings
        4. Knowledge gaps - What remains unknown?
        5. Integrated conclusion - Balanced final assessment
        
        Provide synthesis:
        CONSENSUS: [points of agreement across agents]
        CONFLICTS: [areas of disagreement with analysis]
        CONFIDENCE_ASSESSMENT: [overall confidence with reasoning]
        KNOWLEDGE_GAPS: [remaining unknowns and limitations]
        INTEGRATED_CONCLUSION: [balanced final assessment]
        SYNTHESIS_CONFIDENCE: [your confidence in this synthesis 0-1]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        confidence = self._extract_confidence(response.content)
        
        return AgentResponse(
            agent_role=self.role,
            content=response.content,
            confidence=confidence,
            reasoning="Multi-agent perspective synthesis and integration",
            suggestions=self._extract_integration_suggestions(response.content),
            timestamp=datetime.now()
        )
    
    def generate_final_answer(self, 
                            all_findings: List[Dict[str, Any]],
                            agent_insights: List[AgentResponse],
                            question: str) -> AgentResponse:
        """Generate comprehensive final answer"""
        
        findings_summary = "\n".join([
            f"Finding {i+1}: {finding.get('analysis', str(finding))[:150]}..."
            for i, finding in enumerate(all_findings[:5])
        ])
        
        insights_summary = "\n".join([
            f"{resp.agent_role.value.title()}: {resp.content[:150]}..."
            for resp in agent_insights[-3:]  # Last 3 insights
        ])
        
        prompt = f"""
        Generate a comprehensive final answer for: "{question}"
        
        RESEARCH FINDINGS:
        {findings_summary}
        
        AGENT INSIGHTS:
        {insights_summary}
        
        Create a final answer that:
        1. Directly addresses the research question
        2. Integrates all reliable findings
        3. Acknowledges limitations and uncertainties
        4. Provides confidence assessment
        5. Suggests areas for future research
        
        Structure your response:
        DIRECT_ANSWER: [clear, direct response to the question]
        SUPPORTING_EVIDENCE: [key evidence and sources]
        LIMITATIONS: [acknowledged limitations and uncertainties]
        CONFIDENCE_LEVEL: [overall confidence 1-10 with reasoning]
        FUTURE_RESEARCH: [suggested areas for further investigation]
        FINAL_CONFIDENCE: [your confidence in this final answer 0-1]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        confidence = self._extract_confidence(response.content)
        
        return AgentResponse(
            agent_role=self.role,
            content=response.content,
            confidence=confidence,
            reasoning="Comprehensive synthesis of all research findings and agent insights",
            suggestions=self._extract_future_research(response.content),
            timestamp=datetime.now()
        )
    
    def _extract_confidence(self, content: str) -> float:
        """Extract confidence score from content"""
        if "FINAL_CONFIDENCE:" in content:
            try:
                conf_text = content.split("FINAL_CONFIDENCE:")[1].strip().split()[0]
                return float(conf_text)
            except:
                pass
        return 0.8
    
    def _extract_integration_suggestions(self, content: str) -> List[str]:
        """Extract integration suggestions"""
        if "KNOWLEDGE_GAPS:" in content:
            gaps_text = content.split("KNOWLEDGE_GAPS:")[1].strip()
            return [gap.strip() for gap in gaps_text.split('\n')[:3] if gap.strip()]
        return []
    
    def _extract_future_research(self, content: str) -> List[str]:
        """Extract future research suggestions"""
        if "FUTURE_RESEARCH:" in content:
            future_text = content.split("FUTURE_RESEARCH:")[1].strip()
            return [item.strip() for item in future_text.split('\n')[:3] if item.strip()]
        return []

class MultiAgentOrchestrator:
    """Orchestrates collaboration between multiple agents"""
    
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.critic = CriticAgent()
        self.synthesizer = SynthesizerAgent()
        self.agent_responses: List[AgentResponse] = []
        
    def collaborative_research_analysis(self, 
                                      question: str,
                                      research_plan: List[str],
                                      findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform collaborative multi-agent analysis"""
        
        print("🤖 Starting multi-agent collaborative analysis...")
        
        # Phase 1: Researcher analysis
        print("🔬 Researcher agent analyzing question...")
        researcher_question_analysis = self.researcher.analyze_research_question(question)
        self.agent_responses.append(researcher_question_analysis)
        
        researcher_findings_analysis = self.researcher.evaluate_findings(findings, question)
        self.agent_responses.append(researcher_findings_analysis)
        
        # Phase 2: Critic analysis
        print("🔍 Critic agent reviewing research...")
        critic_plan_analysis = self.critic.critique_research_plan(research_plan, question)
        self.agent_responses.append(critic_plan_analysis)
        
        critic_fact_check = self.critic.fact_check_findings(findings)
        self.agent_responses.append(critic_fact_check)
        
        # Phase 3: Synthesizer integration
        print("🧠 Synthesizer agent integrating insights...")
        researcher_responses = [resp for resp in self.agent_responses if resp.agent_role == AgentRole.RESEARCHER]
        critic_responses = [resp for resp in self.agent_responses if resp.agent_role == AgentRole.CRITIC]
        
        synthesis = self.synthesizer.synthesize_multi_agent_insights(
            researcher_responses, critic_responses, question
        )
        self.agent_responses.append(synthesis)
        
        # Phase 4: Final answer generation
        print("📝 Generating collaborative final answer...")
        final_answer = self.synthesizer.generate_final_answer(
            findings, self.agent_responses, question
        )
        self.agent_responses.append(final_answer)
        
        return {
            "multi_agent_analysis": {
                "researcher_insights": [resp.content for resp in researcher_responses],
                "critic_insights": [resp.content for resp in critic_responses],
                "synthesis": synthesis.content,
                "final_answer": final_answer.content
            },
            "confidence_scores": {
                "researcher_avg": sum(resp.confidence for resp in researcher_responses) / len(researcher_responses),
                "critic_avg": sum(resp.confidence for resp in critic_responses) / len(critic_responses),
                "synthesis_confidence": synthesis.confidence,
                "final_confidence": final_answer.confidence
            },
            "agent_responses": self.agent_responses,
            "collaboration_summary": self._generate_collaboration_summary()
        }
    
    def _generate_collaboration_summary(self) -> Dict[str, Any]:
        """Generate summary of agent collaboration"""
        
        total_responses = len(self.agent_responses)
        avg_confidence = sum(resp.confidence for resp in self.agent_responses) / total_responses if total_responses > 0 else 0
        
        agent_counts = {}
        for resp in self.agent_responses:
            role = resp.agent_role.value
            agent_counts[role] = agent_counts.get(role, 0) + 1
        
        return {
            "total_agent_responses": total_responses,
            "average_confidence": avg_confidence,
            "agent_participation": agent_counts,
            "collaboration_timestamp": datetime.now().isoformat(),
            "quality_indicators": {
                "multi_perspective": len(agent_counts) >= 2,
                "high_confidence": avg_confidence >= 0.7,
                "comprehensive_analysis": total_responses >= 4
            }
        }

def get_multi_agent_system():
    """Get the multi-agent system orchestrator"""
    return MultiAgentOrchestrator()