#!/usr/bin/env python3
# Phase 6: Results Interpretation and Recommendation

import os
import json
import logging
from datetime import datetime

import openai

logger = logging.getLogger("LitRevRA.Phase6")

class ResultsInterpreter:
    """Class to interpret analysis results and provide recommendations."""
    
    def __init__(self, config, analysis_results, research_plan):
        """
        Initialize the ResultsInterpreter.
        
        Args:
            config (dict): Configuration settings
            analysis_results (dict): Results from analysis phase
            research_plan (dict): Research plan from Phase 2
        """
        self.config = config
        self.analysis_results = analysis_results
        self.research_plan = research_plan
        
        # Set up OpenAI API
        openai.api_key = config['api_keys']['openai']
        
        # Load the task description and research question
        self.task_description = config.get('task_description', '')
        self.research_question = config.get('research_question', '')
        
        # Set model parameters
        model_settings = config.get('model_settings', {})
        self.model = model_settings.get('model', 'gpt-4')
        self.temperature = model_settings.get('temperature', 0.7)
        self.max_tokens = model_settings.get('max_tokens', 4000)
        
        # Initialize the interpretation
        self.interpretation = {
            "task_description": self.task_description,
            "research_question": self.research_question,
            "timestamp": datetime.now().isoformat(),
            "summary": "",
            "answers_to_research_question": [],
            "limitations": [],
            "recommendations": [],
            "future_work": []
        }
    
    def interpret_results(self):
        """
        Interpret the analysis results and provide recommendations.
        
        Returns:
            dict: The interpretation and recommendations
        """
        logger.info("Starting results interpretation and recommendation")
        
        # Step 1: Generate a comprehensive summary
        self._generate_summary()
        
        # Step 2: Extract answers to the research question
        self._extract_answers()
        
        # Step 3: Identify limitations
        self._identify_limitations()
        
        # Step 4: Generate recommendations
        self._generate_recommendations()
        
        # Step 5: Suggest future work
        self._suggest_future_work()
        
        return self.interpretation
    
    def _generate_summary(self):
        """Generate a comprehensive summary of the literature review and analysis."""
        logger.info("Generating comprehensive summary")
        
        # Extract key information for the prompt
        key_findings = self.analysis_results.get('key_findings', [])
        trends = self.analysis_results.get('trends', [])
        research_directions = self.analysis_results.get('research_directions', [])
        visualizations = self.analysis_results.get('visualizations', [])
        
        # Create summary texts
        findings_summary = "\n".join([f"{i+1}. {finding.get('statement')}" 
                                      for i, finding in enumerate(key_findings)])
        
        trends_summary = "\n".join([f"{i+1}. {trend.get('title')}: {trend.get('thematic_structure', '')}" 
                                   for i, trend in enumerate(trends)])
        
        directions_summary = "\n".join([f"{i+1}. {direction.get('title')}: {direction.get('description')}" 
                                      for i, direction in enumerate(research_directions)])
        
        # Create visualization list
        vis_list = "\n".join([f"{i+1}. {vis.get('title')}: {vis.get('description')}" 
                             for i, vis in enumerate(visualizations)])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting Postdoc #3, a senior researcher with extensive experience in synthesizing complex research findings. Your task is to interpret detailed analysis and visualizations to identify significant patterns, trends, and relationships within the body of literature.
        
        Based on the following outputs from the Analysis Execution phase:
        
        Key Findings:
        {findings_summary}
        
        Research Trends:
        {trends_summary}
        
        Promising Research Directions:
        {directions_summary}
        
        Visualizations Created:
        {vis_list}
        
        Generate a comprehensive synthesis (approximately 1000-1500 words) that:
        
        1. Identifies significant patterns, trends, and relationships across the analyzed literature
        2. Synthesizes individual findings into a cohesive narrative that directly addresses the research question
        3. Interprets the visualization results to extract deeper insights
        4. Connects observed patterns to broader theoretical frameworks in the field
        5. Highlights areas of consensus and contradiction among researchers
        6. Evaluates the strength and quality of evidence supporting key conclusions
        7. Articulates the implications of these findings for both theory and practice
        
        The summary should demonstrate deep analytical thinking, critical evaluation of the evidence, and a sophisticated understanding of how these findings collectively address the original research question. Structure your response as a scholarly synthesis that builds a bridge between the detailed analysis and actionable recommendations.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping a senior postdoctoral researcher interpret and synthesize complex literature review findings into a cohesive narrative."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the summary
            summary = response.choices[0].message.content.strip()
            
            self.interpretation['summary'] = summary
            
            logger.info("Generated comprehensive summary")
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
    
    def _extract_answers(self):
        """Extract direct answers to the research question from the analysis."""
        logger.info("Extracting answers to the research question")
        
        # Extract key information for the prompt
        key_findings = self.analysis_results.get('key_findings', [])
        
        # Create findings text
        findings_text = ""
        for i, finding in enumerate(key_findings):
            findings_text += f"Finding {i+1}: {finding.get('statement')}\n"
            findings_text += f"Evidence: {finding.get('evidence')}\n"
            findings_text += f"Significance: {finding.get('significance')}\n\n"
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Based on the following key findings from our literature review:
        
        {findings_text}
        
        Extract 3-5 clear, direct answers to the research question. For each answer:
        
        1. Provide a concise statement that directly addresses the research question
        2. Explain the supporting evidence from the literature
        3. Discuss any caveats or qualifications to this answer
        4. Rate the confidence level (High/Medium/Low) based on the strength of the evidence
        
        Format your response with clear sections for each answer.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to extract direct answers to research questions from literature review findings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the answers
            answers_text = response.choices[0].message.content.strip()
            
            # Parse the answers (simple approach)
            answers = []
            current_answer = None
            
            for line in answers_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for answer headers
                if "Answer" in line or "ANSWER" in line:
                    # Save the previous answer
                    if current_answer and current_answer.get('statement'):
                        answers.append(current_answer)
                    
                    # Extract answer statement
                    statement = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new answer
                    current_answer = {
                        "statement": statement,
                        "evidence": "",
                        "caveats": "",
                        "confidence": ""
                    }
                elif current_answer:
                    # Parse answer components
                    if "evidence" in line.lower() or "support" in line.lower():
                        current_answer["evidence"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "caveat" in line.lower() or "qualification" in line.lower():
                        current_answer["caveats"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "confidence" in line.lower():
                        current_answer["confidence"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["confidence", "caveats", "evidence", "statement"]:
                            if current_answer[component]:
                                current_answer[component] += " " + line
                                break
            
            # Add the last answer
            if current_answer and current_answer.get('statement'):
                answers.append(current_answer)
            
            self.interpretation['answers_to_research_question'] = answers
            
            logger.info(f"Extracted {len(answers)} answers to the research question")
            
        except Exception as e:
            logger.error(f"Error extracting answers: {str(e)}")
    
    def _identify_limitations(self):
        """Identify limitations of the literature review and analysis."""
        logger.info("Identifying limitations")
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Based on our literature review and analysis, identify 5-7 limitations or constraints that affect the interpretation of our findings. For each limitation:
        
        1. Clearly describe the limitation
        2. Explain how it affects the interpretation of results
        3. Discuss potential methods to address or mitigate this limitation
        
        Consider limitations related to:
        - Scope of the literature review
        - Methodological issues in the reviewed studies
        - Gaps in current knowledge
        - Biases in the literature
        - Contradictory findings
        - Measurement or analytical challenges
        
        Format your response with clear sections for each limitation.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to identify limitations in a literature review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the limitations
            limitations_text = response.choices[0].message.content.strip()
            
            # Parse the limitations (simple approach)
            limitations = []
            current_limitation = None
            
            for line in limitations_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for limitation headers
                if "Limitation" in line or "LIMITATION" in line:
                    # Save the previous limitation
                    if current_limitation and current_limitation.get('description'):
                        limitations.append(current_limitation)
                    
                    # Extract limitation description
                    description = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new limitation
                    current_limitation = {
                        "description": description,
                        "impact": "",
                        "mitigation": ""
                    }
                elif current_limitation:
                    # Parse limitation components
                    if "impact" in line.lower() or "affect" in line.lower() or "effect" in line.lower():
                        current_limitation["impact"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "mitigation" in line.lower() or "address" in line.lower() or "solution" in line.lower():
                        current_limitation["mitigation"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["mitigation", "impact", "description"]:
                            if current_limitation[component]:
                                current_limitation[component] += " " + line
                                break
            
            # Add the last limitation
            if current_limitation and current_limitation.get('description'):
                limitations.append(current_limitation)
            
            self.interpretation['limitations'] = limitations
            
            logger.info(f"Identified {len(limitations)} limitations")
            
        except Exception as e:
            logger.error(f"Error identifying limitations: {str(e)}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on the literature review and analysis."""
        logger.info("Generating recommendations")
        
        # Extract key information for the prompt
        key_findings = self.analysis_results.get('key_findings', [])
        research_directions = self.analysis_results.get('research_directions', [])
        
        # Create findings and directions text
        findings_text = "\n".join([f"{i+1}. {finding.get('statement')}" 
                                  for i, finding in enumerate(key_findings)])
        
        directions_text = "\n".join([f"{i+1}. {direction.get('title')}: {direction.get('description')}" 
                                    for i, direction in enumerate(research_directions)])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Based on the following information from our literature review and analysis:
        
        Key Findings:
        {findings_text}
        
        Promising Research Directions:
        {directions_text}
        
        Generate 5-7 practical, actionable recommendations for researchers, practitioners, or stakeholders in this field. For each recommendation:
        
        1. Provide a clear, specific recommendation
        2. Explain the rationale behind this recommendation
        3. Discuss how it builds on the literature findings
        4. Outline potential implementation approaches
        5. Explain the expected benefits of following this recommendation
        
        Your recommendations should be evidence-based, practical, and directly relevant to the research question. Format your response with clear sections for each recommendation.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to generate recommendations based on literature review findings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the recommendations
            recommendations_text = response.choices[0].message.content.strip()
            
            # Parse the recommendations (simple approach)
            recommendations = []
            current_recommendation = None
            
            for line in recommendations_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for recommendation headers
                if "Recommendation" in line or "RECOMMENDATION" in line:
                    # Save the previous recommendation
                    if current_recommendation and current_recommendation.get('statement'):
                        recommendations.append(current_recommendation)
                    
                    # Extract recommendation statement
                    statement = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new recommendation
                    current_recommendation = {
                        "statement": statement,
                        "rationale": "",
                        "literature_basis": "",
                        "implementation": "",
                        "benefits": ""
                    }
                elif current_recommendation:
                    # Parse recommendation components
                    if "rationale" in line.lower() or "reason" in line.lower():
                        current_recommendation["rationale"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "literature" in line.lower() or "findings" in line.lower() or "builds on" in line.lower():
                        current_recommendation["literature_basis"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "implementation" in line.lower() or "approach" in line.lower():
                        current_recommendation["implementation"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "benefit" in line.lower() or "advantage" in line.lower() or "impact" in line.lower():
                        current_recommendation["benefits"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["benefits", "implementation", "literature_basis", "rationale", "statement"]:
                            if current_recommendation[component]:
                                current_recommendation[component] += " " + line
                                break
            
            # Add the last recommendation
            if current_recommendation and current_recommendation.get('statement'):
                recommendations.append(current_recommendation)
            
            self.interpretation['recommendations'] = recommendations
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
    
    def _suggest_future_work(self):
        """Suggest future work based on the literature review and analysis."""
        logger.info("Suggesting future work")
        
        # Extract key information for the prompt
        research_directions = self.analysis_results.get('research_directions', [])
        limitations = self.interpretation.get('limitations', [])
        
        # Create directions and limitations text
        directions_text = "\n".join([f"{i+1}. {direction.get('title')}: {direction.get('description')}" 
                                    for i, direction in enumerate(research_directions)])
        
        limitations_text = "\n".join([f"{i+1}. {limitation.get('description')}" 
                                     for i, limitation in enumerate(limitations)])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Based on the following information from our literature review and analysis:
        
        Promising Research Directions:
        {directions_text}
        
        Limitations:
        {limitations_text}
        
        Suggest 4-6 specific areas for future research that would advance knowledge in this field. For each suggestion:
        
        1. Provide a clear title for the future research area
        2. Describe specific research questions that could be explored
        3. Suggest methodological approaches that could be employed
        4. Explain how this future work addresses gaps or limitations in the current literature
        5. Discuss the potential impact and significance of this research
        
        Your suggestions should be forward-looking, innovative, and directly address the limitations and gaps identified in the current literature. Format your response with clear sections for each future research suggestion.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to suggest future research directions based on literature review findings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Slightly higher temperature for creativity
                max_tokens=self.max_tokens,
            )
            
            # Extract the future work suggestions
            future_work_text = response.choices[0].message.content.strip()
            
            # Parse the future work suggestions (simple approach)
            future_work = []
            current_suggestion = None
            
            for line in future_work_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for suggestion headers
                if "Future" in line or "Research Area" in line or "Suggestion" in line:
                    # Save the previous suggestion
                    if current_suggestion and current_suggestion.get('title'):
                        future_work.append(current_suggestion)
                    
                    # Extract suggestion title
                    title = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new suggestion
                    current_suggestion = {
                        "title": title,
                        "research_questions": [],
                        "methodologies": "",
                        "addresses_gaps": "",
                        "impact": ""
                    }
                elif current_suggestion:
                    # Parse suggestion components
                    if "question" in line.lower() or "explore" in line.lower():
                        if "research_questions" in current_suggestion and isinstance(current_suggestion["research_questions"], list):
                            questions = line.split(':', 1)[1].strip().split('?') if ':' in line else [line]
                            current_suggestion["research_questions"].extend([q.strip() + '?' if not q.strip().endswith('?') and q.strip() else q.strip() for q in questions if q.strip()])
                        else:
                            current_suggestion["research_questions"] = []
                    elif "methodolog" in line.lower() or "approach" in line.lower():
                        current_suggestion["methodologies"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "address" in line.lower() or "gap" in line.lower() or "limitation" in line.lower():
                        current_suggestion["addresses_gaps"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "impact" in line.lower() or "significance" in line.lower():
                        current_suggestion["impact"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["impact", "addresses_gaps", "methodologies"]:
                            if current_suggestion[component]:
                                current_suggestion[component] += " " + line
                                break
            
            # Add the last suggestion
            if current_suggestion and current_suggestion.get('title'):
                future_work.append(current_suggestion)
            
            self.interpretation['future_work'] = future_work
            
            logger.info(f"Suggested {len(future_work)} areas for future work")
            
        except Exception as e:
            logger.error(f"Error suggesting future work: {str(e)}")

def interpret_results(config, analysis_results, research_plan):
    """
    Main function to interpret results and provide recommendations.
    
    Args:
        config (dict): Configuration settings
        analysis_results (dict): Results from analysis phase
        research_plan (dict): Research plan from Phase 2
        
    Returns:
        dict: The interpretation and recommendations
    """
    # Create a results interpreter
    interpreter = ResultsInterpreter(config, analysis_results, research_plan)
    
    # Interpret results
    interpretation = interpreter.interpret_results()
    
    return interpretation

if __name__ == "__main__":
    # Example usage if run directly
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from phase0_config import load_configuration
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_path = "../../example_config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = load_configuration(config_path)
    
    # Load analysis results
    analysis_path = "analysis_results.json"
    if len(sys.argv) > 2:
        analysis_path = sys.argv[2]
    
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)
    
    # Load research plan
    plan_path = "research_plan.json"
    if len(sys.argv) > 3:
        plan_path = sys.argv[3]
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        research_plan = json.load(f)
    
    # Interpret results
    interpretation = interpret_results(config, analysis_results, research_plan)
    
    # Save output
    output_path = "interpretation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(interpretation, f, indent=2, ensure_ascii=False)
    
    print(f"Interpretation saved to {output_path}")