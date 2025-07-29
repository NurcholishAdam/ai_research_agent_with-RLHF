# -*- coding: utf-8 -*-
"""
Hypothesis Generation and Testing Engine - Phase 4 Intelligence Layer
Generates, tests, and validates research hypotheses automatically
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain.schema import HumanMessage
from llm.groq_wrapper import load_groq_llm
import json
import re
from datetime import datetime

class HypothesisType(Enum):
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    DESCRIPTIVE = "descriptive"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"

class EvidenceStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"

@dataclass
class Hypothesis:
    """Represents a research hypothesis"""
    id: str
    statement: str
    type: HypothesisType
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    testable: bool
    variables: List[str]
    predictions: List[str]
    generated_at: datetime

@dataclass
class HypothesisTest:
    """Represents a hypothesis test result"""
    hypothesis_id: str
    test_method: str
    evidence_strength: EvidenceStrength
    support_score: float
    contradictions: List[str]
    limitations: List[str]
    conclusion: str
    confidence: float

class HypothesisGenerator:
    """Generates research hypotheses from findings"""
    
    def __init__(self):
        self.llm = load_groq_llm()
        
    def generate_hypotheses(self, 
                          research_question: str,
                          findings: List[Dict[str, Any]],
                          max_hypotheses: int = 5) -> List[Hypothesis]:
        """Generate hypotheses based on research findings"""
        
        findings_summary = self._summarize_findings(findings)
        
        prompt = f"""
        You are a hypothesis generation expert. Based on this research question and findings, generate testable hypotheses.
        
        Research Question: {research_question}
        
        Key Findings:
        {findings_summary}
        
        Generate {max_hypotheses} distinct hypotheses that:
        1. Are directly related to the research question
        2. Are testable and falsifiable
        3. Are supported by at least some evidence
        4. Cover different aspects or relationships
        5. Vary in scope (broad to specific)
        
        For each hypothesis, provide:
        HYPOTHESIS_[N]: [clear, testable statement]
        TYPE_[N]: [causal/correlational/descriptive/predictive/comparative]
        CONFIDENCE_[N]: [0.0-1.0 confidence score]
        EVIDENCE_[N]: [supporting evidence from findings]
        VARIABLES_[N]: [key variables involved]
        PREDICTIONS_[N]: [testable predictions this hypothesis makes]
        TESTABLE_[N]: [yes/no - is this practically testable?]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return self._parse_hypotheses(response.content)
    
    def generate_alternative_hypotheses(self, 
                                      primary_hypothesis: Hypothesis,
                                      findings: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Generate alternative hypotheses to test against primary"""
        
        findings_summary = self._summarize_findings(findings)
        
        prompt = f"""
        Generate alternative hypotheses to test against this primary hypothesis:
        
        Primary Hypothesis: {primary_hypothesis.statement}
        Type: {primary_hypothesis.type.value}
        
        Research Findings:
        {findings_summary}
        
        Generate 3 alternative hypotheses that:
        1. Explain the same phenomena differently
        2. Are mutually exclusive with the primary hypothesis
        3. Are equally testable
        4. Account for the same evidence
        
        Format each as:
        ALT_HYPOTHESIS_[N]: [alternative explanation]
        ALT_TYPE_[N]: [hypothesis type]
        ALT_CONFIDENCE_[N]: [confidence score]
        ALT_EVIDENCE_[N]: [supporting evidence]
        ALT_VARIABLES_[N]: [key variables]
        ALT_PREDICTIONS_[N]: [different predictions this makes]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return self._parse_alternative_hypotheses(response.content)
    
    def _summarize_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Summarize findings for hypothesis generation"""
        summary_parts = []
        
        for i, finding in enumerate(findings[:10]):  # Limit to 10 findings
            if isinstance(finding, dict):
                analysis = finding.get('analysis', '')
                if 'KEY_FINDINGS:' in analysis:
                    key_part = analysis.split('KEY_FINDINGS:')[1].split('NEW_CONCEPTS:')[0]
                    summary_parts.append(f"Finding {i+1}: {key_part.strip()[:200]}...")
                else:
                    summary_parts.append(f"Finding {i+1}: {str(finding)[:200]}...")
        
        return "\n".join(summary_parts)
    
    def _parse_hypotheses(self, content: str) -> List[Hypothesis]:
        """Parse generated hypotheses from LLM response"""
        hypotheses = []
        
        # Extract hypothesis blocks
        hypothesis_pattern = r'HYPOTHESIS_(\d+):\s*(.+?)(?=HYPOTHESIS_\d+:|$)'
        matches = re.findall(hypothesis_pattern, content, re.DOTALL)
        
        for match in matches:
            hypothesis_num, hypothesis_block = match
            
            try:
                # Extract components
                statement = self._extract_field(hypothesis_block, f'HYPOTHESIS_{hypothesis_num}')
                hyp_type = self._extract_field(hypothesis_block, f'TYPE_{hypothesis_num}')
                confidence = self._extract_field(hypothesis_block, f'CONFIDENCE_{hypothesis_num}')
                evidence = self._extract_field(hypothesis_block, f'EVIDENCE_{hypothesis_num}')
                variables = self._extract_field(hypothesis_block, f'VARIABLES_{hypothesis_num}')
                predictions = self._extract_field(hypothesis_block, f'PREDICTIONS_{hypothesis_num}')
                testable = self._extract_field(hypothesis_block, f'TESTABLE_{hypothesis_num}')
                
                # Create hypothesis object
                hypothesis = Hypothesis(
                    id=f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hypothesis_num}",
                    statement=statement or f"Generated hypothesis {hypothesis_num}",
                    type=self._parse_hypothesis_type(hyp_type),
                    confidence=self._parse_confidence(confidence),
                    supporting_evidence=self._parse_list(evidence),
                    contradicting_evidence=[],
                    testable=self._parse_testable(testable),
                    variables=self._parse_list(variables),
                    predictions=self._parse_list(predictions),
                    generated_at=datetime.now()
                )
                
                hypotheses.append(hypothesis)
                
            except Exception as e:
                print(f"Error parsing hypothesis {hypothesis_num}: {e}")
                continue
        
        return hypotheses
    
    def _parse_alternative_hypotheses(self, content: str) -> List[Hypothesis]:
        """Parse alternative hypotheses from LLM response"""
        hypotheses = []
        
        alt_pattern = r'ALT_HYPOTHESIS_(\d+):\s*(.+?)(?=ALT_HYPOTHESIS_\d+:|$)'
        matches = re.findall(alt_pattern, content, re.DOTALL)
        
        for match in matches:
            hypothesis_num, hypothesis_block = match
            
            try:
                statement = self._extract_field(hypothesis_block, f'ALT_HYPOTHESIS_{hypothesis_num}')
                hyp_type = self._extract_field(hypothesis_block, f'ALT_TYPE_{hypothesis_num}')
                confidence = self._extract_field(hypothesis_block, f'ALT_CONFIDENCE_{hypothesis_num}')
                evidence = self._extract_field(hypothesis_block, f'ALT_EVIDENCE_{hypothesis_num}')
                variables = self._extract_field(hypothesis_block, f'ALT_VARIABLES_{hypothesis_num}')
                predictions = self._extract_field(hypothesis_block, f'ALT_PREDICTIONS_{hypothesis_num}')
                
                hypothesis = Hypothesis(
                    id=f"alt_hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hypothesis_num}",
                    statement=statement or f"Alternative hypothesis {hypothesis_num}",
                    type=self._parse_hypothesis_type(hyp_type),
                    confidence=self._parse_confidence(confidence),
                    supporting_evidence=self._parse_list(evidence),
                    contradicting_evidence=[],
                    testable=True,
                    variables=self._parse_list(variables),
                    predictions=self._parse_list(predictions),
                    generated_at=datetime.now()
                )
                
                hypotheses.append(hypothesis)
                
            except Exception as e:
                print(f"Error parsing alternative hypothesis {hypothesis_num}: {e}")
                continue
        
        return hypotheses
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract field value from text"""
        pattern = f'{field_name}:\\s*(.+?)(?=\\n[A-Z_]+_\\d+:|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _parse_hypothesis_type(self, type_str: str) -> HypothesisType:
        """Parse hypothesis type from string"""
        type_str = type_str.lower().strip()
        for hyp_type in HypothesisType:
            if hyp_type.value in type_str:
                return hyp_type
        return HypothesisType.DESCRIPTIVE
    
    def _parse_confidence(self, conf_str: str) -> float:
        """Parse confidence score from string"""
        try:
            # Extract number from string
            numbers = re.findall(r'0?\.\d+|\d+\.?\d*', conf_str)
            if numbers:
                conf = float(numbers[0])
                return min(max(conf, 0.0), 1.0)  # Clamp between 0 and 1
        except:
            pass
        return 0.5
    
    def _parse_list(self, list_str: str) -> List[str]:
        """Parse list from string"""
        if not list_str:
            return []
        
        # Split by common delimiters
        items = re.split(r'[,;]\s*|\n\s*[-•]\s*', list_str)
        return [item.strip() for item in items if item.strip()][:5]  # Limit to 5 items
    
    def _parse_testable(self, testable_str: str) -> bool:
        """Parse testable boolean from string"""
        return 'yes' in testable_str.lower() or 'true' in testable_str.lower()

class HypothesisTester:
    """Tests hypotheses against available evidence"""
    
    def __init__(self):
        self.llm = load_groq_llm()
        
    def test_hypothesis(self, 
                       hypothesis: Hypothesis,
                       findings: List[Dict[str, Any]],
                       additional_evidence: List[str] = None) -> HypothesisTest:
        """Test a hypothesis against available evidence"""
        
        findings_summary = self._summarize_findings_for_testing(findings)
        additional_evidence_text = "\n".join(additional_evidence or [])
        
        prompt = f"""
        Test this hypothesis against available evidence:
        
        HYPOTHESIS: {hypothesis.statement}
        TYPE: {hypothesis.type.value}
        PREDICTED OUTCOMES: {'; '.join(hypothesis.predictions)}
        
        AVAILABLE EVIDENCE:
        {findings_summary}
        
        ADDITIONAL EVIDENCE:
        {additional_evidence_text}
        
        Evaluate the hypothesis by:
        1. Comparing predictions with actual evidence
        2. Identifying supporting evidence
        3. Identifying contradicting evidence
        4. Assessing evidence quality and reliability
        5. Determining overall support level
        
        Provide assessment:
        SUPPORTING_EVIDENCE: [evidence that supports the hypothesis]
        CONTRADICTING_EVIDENCE: [evidence that contradicts the hypothesis]
        EVIDENCE_STRENGTH: [strong/moderate/weak/insufficient]
        SUPPORT_SCORE: [0.0-1.0 how well evidence supports hypothesis]
        LIMITATIONS: [limitations in the evidence or test]
        TEST_CONCLUSION: [accept/reject/inconclusive with reasoning]
        CONFIDENCE: [confidence in this assessment 0.0-1.0]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return self._parse_test_result(hypothesis.id, response.content)
    
    def compare_hypotheses(self, 
                          hypotheses: List[Hypothesis],
                          findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple hypotheses and rank them"""
        
        test_results = []
        for hypothesis in hypotheses:
            test_result = self.test_hypothesis(hypothesis, findings)
            test_results.append((hypothesis, test_result))
        
        # Rank hypotheses by support score
        ranked_hypotheses = sorted(
            test_results, 
            key=lambda x: x[1].support_score, 
            reverse=True
        )
        
        findings_summary = self._summarize_findings_for_testing(findings)
        
        # Generate comparative analysis
        comparison_prompt = f"""
        Compare these tested hypotheses and provide analysis:
        
        HYPOTHESES AND TEST RESULTS:
        {self._format_hypothesis_comparison(ranked_hypotheses)}
        
        EVIDENCE BASE:
        {findings_summary}
        
        Provide comparative analysis:
        BEST_SUPPORTED: [which hypothesis has strongest support]
        RANKING_RATIONALE: [why this ranking makes sense]
        EVIDENCE_GAPS: [what evidence would help discriminate better]
        ALTERNATIVE_EXPLANATIONS: [other possible explanations not covered]
        RESEARCH_RECOMMENDATIONS: [what research would resolve uncertainties]
        """
        
        comparison_response = self.llm.invoke([HumanMessage(content=comparison_prompt)])
        
        return {
            "ranked_hypotheses": [
                {
                    "hypothesis": hyp.statement,
                    "type": hyp.type.value,
                    "support_score": test.support_score,
                    "evidence_strength": test.evidence_strength.value,
                    "conclusion": test.conclusion
                }
                for hyp, test in ranked_hypotheses
            ],
            "comparative_analysis": comparison_response.content,
            "test_results": test_results,
            "summary": {
                "total_hypotheses": len(hypotheses),
                "best_hypothesis": ranked_hypotheses[0][0].statement if ranked_hypotheses else None,
                "best_support_score": ranked_hypotheses[0][1].support_score if ranked_hypotheses else 0,
                "average_support": sum(test.support_score for _, test in test_results) / len(test_results) if test_results else 0
            }
        }
    
    def _summarize_findings_for_testing(self, findings: List[Dict[str, Any]]) -> str:
        """Summarize findings for hypothesis testing"""
        summary_parts = []
        
        for i, finding in enumerate(findings[:8]):  # Limit for testing
            if isinstance(finding, dict):
                analysis = finding.get('analysis', '')
                # Extract key findings and evidence
                if 'KEY_FINDINGS:' in analysis:
                    key_part = analysis.split('KEY_FINDINGS:')[1].split('NEW_CONCEPTS:')[0]
                    summary_parts.append(f"Evidence {i+1}: {key_part.strip()[:250]}...")
                else:
                    summary_parts.append(f"Evidence {i+1}: {str(finding)[:250]}...")
        
        return "\n".join(summary_parts)
    
    def _parse_test_result(self, hypothesis_id: str, content: str) -> HypothesisTest:
        """Parse hypothesis test result from LLM response"""
        
        supporting = self._extract_test_field(content, 'SUPPORTING_EVIDENCE')
        contradicting = self._extract_test_field(content, 'CONTRADICTING_EVIDENCE')
        strength = self._extract_test_field(content, 'EVIDENCE_STRENGTH')
        support_score = self._extract_test_field(content, 'SUPPORT_SCORE')
        limitations = self._extract_test_field(content, 'LIMITATIONS')
        conclusion = self._extract_test_field(content, 'TEST_CONCLUSION')
        confidence = self._extract_test_field(content, 'CONFIDENCE')
        
        return HypothesisTest(
            hypothesis_id=hypothesis_id,
            test_method="evidence_comparison",
            evidence_strength=self._parse_evidence_strength(strength),
            support_score=self._parse_support_score(support_score),
            contradictions=self._parse_test_list(contradicting),
            limitations=self._parse_test_list(limitations),
            conclusion=conclusion or "Inconclusive",
            confidence=self._parse_test_confidence(confidence)
        )
    
    def _extract_test_field(self, text: str, field_name: str) -> str:
        """Extract field from test result"""
        pattern = f'{field_name}:\\s*(.+?)(?=\\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _parse_evidence_strength(self, strength_str: str) -> EvidenceStrength:
        """Parse evidence strength from string"""
        strength_str = strength_str.lower()
        for strength in EvidenceStrength:
            if strength.value in strength_str:
                return strength
        return EvidenceStrength.MODERATE
    
    def _parse_support_score(self, score_str: str) -> float:
        """Parse support score from string"""
        try:
            numbers = re.findall(r'0?\.\d+|\d+\.?\d*', score_str)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
        except:
            pass
        return 0.5
    
    def _parse_test_list(self, list_str: str) -> List[str]:
        """Parse list from test result"""
        if not list_str:
            return []
        items = re.split(r'[,;]\s*|\n\s*[-•]\s*', list_str)
        return [item.strip() for item in items if item.strip()][:3]
    
    def _parse_test_confidence(self, conf_str: str) -> float:
        """Parse confidence from test result"""
        try:
            numbers = re.findall(r'0?\.\d+|\d+\.?\d*', conf_str)
            if numbers:
                conf = float(numbers[0])
                return min(max(conf, 0.0), 1.0)
        except:
            pass
        return 0.7
    
    def _format_hypothesis_comparison(self, ranked_hypotheses: List[Tuple[Hypothesis, HypothesisTest]]) -> str:
        """Format hypotheses for comparison"""
        formatted = []
        for i, (hyp, test) in enumerate(ranked_hypotheses, 1):
            formatted.append(f"""
Hypothesis {i}: {hyp.statement}
Support Score: {test.support_score:.2f}
Evidence Strength: {test.evidence_strength.value}
Conclusion: {test.conclusion}
""")
        return "\n".join(formatted)

def get_hypothesis_engine():
    """Get hypothesis generation and testing engine"""
    return {
        "generator": HypothesisGenerator(),
        "tester": HypothesisTester()
    }
