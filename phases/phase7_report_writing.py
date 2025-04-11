#!/usr/bin/env python3
# Phase 7: Report Writing

import os
import json
import logging
from datetime import datetime
import openai

logger = logging.getLogger("LitRevRA.Phase7")

class ReportWriter:
    """Class to write a comprehensive research report."""
    
    def __init__(self, config, literature_data, research_plan, ideas, scoring_framework, analysis_results, interpretation):
        """
        Initialize the ReportWriter.
        
        Args:
            config (dict): Configuration settings
            literature_data (dict): Collected literature data
            research_plan (dict): Research plan from Phase 2
            ideas (dict): Identified research ideas
            scoring_framework (dict): Scoring framework from Phase 4
            analysis_results (dict): Results from analysis phase
            interpretation (dict): Interpretation and recommendations
        """
        self.config = config
        self.literature_data = literature_data
        self.research_plan = research_plan
        self.ideas = ideas
        self.scoring_framework = scoring_framework
        self.analysis_results = analysis_results
        self.interpretation = interpretation
        
        # Set up OpenAI API
        self.openai_api_key = config.get('api_keys', {}).get('openai')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Load the task description and research question
        self.task_description = config.get('task_description', '')
        self.research_question = config.get('research_question', '')
        
        # Set model parameters
        model_settings = config.get('model_settings', {})
        self.model = model_settings.get('model', 'gpt-4')
        self.temperature = model_settings.get('temperature', 0.7)
        self.max_tokens = model_settings.get('max_tokens', 4000)
        
        # Initialize the report sections
        self.report_sections = {}
    
    def generate_report(self):
        """
        Generate a comprehensive research report.
        
        Returns:
            str: The complete report
        """
        logger.info("Starting comprehensive report generation")
        
        # Generate each section of the report
        self._generate_abstract()
        self._generate_introduction()
        self._generate_methodology()
        self._generate_literature_review()
        self._generate_findings()
        self._generate_discussion()
        self._generate_future_directions()
        self._generate_conclusion()
        self._generate_references()
        
        # Combine all sections into a full report
        full_report = self._assemble_full_report()
        
        logger.info("Report generation completed")
        return full_report
    
    def _generate_abstract(self):
        """Generate the abstract section of the report."""
        logger.info("Generating abstract")
        
        # Prepare data for the prompt
        summary = self.interpretation.get('summary', '')
        answers = self.interpretation.get('answers_to_research_question', [])
        
        # Create a condensed version of the answers
        answers_text = "\n".join([f"- {answer.get('statement', '')}" for answer in answers])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following summary and key answers from our literature review:
        
        Summary:
        {summary[:500]}...
        
        Key Answers to Research Question:
        {answers_text}
        
        Write a concise, scholarly abstract (250-300 words) for a scientific paper that:
        
        1. Introduces the research question and its importance
        2. Briefly describes the methodology of using LitRevRA for conducting the literature review
        3. Summarizes the key findings
        4. Highlights the main implications and contributions to the field
        5. Emphasizes the novelty, rigor, and relevance of this work
        
        The abstract should be self-contained, scholarly in tone, and adhere to high academic standards. It should clearly convey the paper's contribution to the field and entice readers to explore the full paper.
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the abstract
                abstract = response.choices[0].message.content.strip()
                
                self.report_sections['abstract'] = abstract
                
                logger.info("Abstract generation completed")
                
            except Exception as e:
                logger.error(f"Error generating abstract: {str(e)}")
                self.report_sections['abstract'] = "# Abstract\n\nError generating abstract section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder abstract.")
            self.report_sections['abstract'] = "# Abstract\n\nPlaceholder for abstract section."
    
    def _generate_introduction(self):
        """Generate the introduction section of the report."""
        logger.info("Generating introduction")
        
        # Prepare data for the prompt
        summary = self.interpretation.get('summary', '')
        research_gaps = self.research_plan.get('research_gaps', [])
        
        # Create research gaps text
        gaps_text = "\n".join([f"- {gap.get('description', '')}" for gap in research_gaps])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following summary and identified research gaps:
        
        Summary:
        {summary[:500]}...
        
        Research Gaps:
        {gaps_text}
        
        Write a comprehensive introduction section (800-1000 words) that:
        
        1. Sets the context and background of the research area
        2. Clearly states the research question and its significance
        3. Explains the motivation for using LitRevRA as an innovative approach to literature review
        4. Discusses the current state of the field and identified gaps in knowledge
        5. Outlines the structure of the paper
        6. Articulates the key contributions of this work, emphasizing:
           - Novelty: How this approach differs from traditional literature reviews
           - Rigor: The thoroughness of the analysis
           - Relevance: The significance and potential impact on the field
        
        The introduction should engage readers, establish the importance of the research question, and provide a clear roadmap for the paper. It should be scholarly, well-structured, and set up expectations for what follows.
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the introduction
                introduction = response.choices[0].message.content.strip()
                
                self.report_sections['introduction'] = introduction
                
                logger.info("Introduction generation completed")
                
            except Exception as e:
                logger.error(f"Error generating introduction: {str(e)}")
                self.report_sections['introduction'] = "# Introduction\n\nError generating introduction section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder introduction.")
            self.report_sections['introduction'] = "# Introduction\n\nPlaceholder for introduction section."
    
    def _generate_methodology(self):
        """Generate the methodology section of the report."""
        logger.info("Generating methodology")
        
        # Prepare data for the prompt
        methodology = self.research_plan.get('methodology', {}).get('full_text', '')
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following methodology information:
        
        Methodology:
        {methodology[:800]}...
        
        Write a detailed methodology section (800-1000 words) that:
        
        1. Describes the LitRevRA system and its seven phases:
           - Phase 1: Data Collection
           - Phase 2: Analysis and Planning
           - Phase 3: Idea Identification
           - Phase 4: Relevancy Scoring
           - Phase 5: Analysis Execution
           - Phase 6: Results Interpretation
           - Phase 7: Report Writing
        
        2. Details the specific processes used in each phase, including:
           - Data sources and collection methods
           - Analytical techniques employed
           - Criteria used for scoring and evaluation
           - Tools and technologies utilized
        
        3. Explains the rationale behind methodological choices
        
        4. Addresses methodological rigor and verifiability by:
           - Detailing how data quality and reliability were ensured
           - Explaining how potential biases were addressed
           - Describing steps taken to ensure reproducibility
        
        The methodology section should be transparent, precise, and provide sufficient detail for other researchers to understand and potentially replicate the approach. It should demonstrate scientific rigor and adhere to high academic standards.
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the methodology
                methodology_section = response.choices[0].message.content.strip()
                
                self.report_sections['methodology'] = methodology_section
                
                logger.info("Methodology generation completed")
                
            except Exception as e:
                logger.error(f"Error generating methodology: {str(e)}")
                self.report_sections['methodology'] = "# Methodology\n\nError generating methodology section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder methodology.")
            self.report_sections['methodology'] = "# Methodology\n\nPlaceholder for methodology section."
    
    def _generate_literature_review(self):
        """Generate the literature review section of the report."""
        logger.info("Generating literature review")
        
        # Prepare data for the prompt
        lit_overview = self.research_plan.get('literature_overview', {}).get('detailed_overview', '')
        trends = self.analysis_results.get('trends', [])
        
        # Create trends text
        trends_text = ""
        for i, trend in enumerate(trends[:5]):  # Limit to first 5 trends
            trends_text += f"Trend {i+1}: {trend.get('title', '')}\n"
            trends_text += f"Description: {trend.get('description', '')}\n\n"
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following literature overview and identified trends:
        
        Literature Overview:
        {lit_overview[:800]}...
        
        Key Trends:
        {trends_text}
        
        Write a comprehensive literature review section (1200-1500 words) that:
        
        1. Synthesizes the current state of knowledge in the field
        2. Organizes the literature into clear thematic categories
        3. Discusses major trends and patterns identified in the research
        4. Highlights areas of consensus and debate among researchers
        5. Critically evaluates the strengths and limitations of existing research
        6. Identifies gaps and opportunities for further investigation
        
        The literature review should be well-structured, scholarly, and demonstrate a deep understanding of the field. It should provide a solid foundation for the findings and discussion sections that follow. Use appropriate academic citations and maintain a balanced, objective tone throughout.
        
        The section should demonstrate:
        - Rigor in the comprehensive coverage of relevant literature
        - Clarity in explaining complex concepts and relationships
        - Transparency in how conclusions about the literature were reached
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the literature review
                lit_review = response.choices[0].message.content.strip()
                
                self.report_sections['literature_review'] = lit_review
                
                logger.info("Literature review generation completed")
                
            except Exception as e:
                logger.error(f"Error generating literature review: {str(e)}")
                self.report_sections['literature_review'] = "# Literature Review\n\nError generating literature review section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder literature review.")
            self.report_sections['literature_review'] = "# Literature Review\n\nPlaceholder for literature review section."
    
    def _generate_findings(self):
        """Generate the findings section of the report."""
        logger.info("Generating findings")
        
        # Prepare data for the prompt
        key_findings = self.analysis_results.get('key_findings', [])
        visualizations = self.analysis_results.get('visualizations', [])
        
        # Create findings text
        findings_text = ""
        for i, finding in enumerate(key_findings):
            findings_text += f"Finding {i+1}: {finding.get('statement', '')}\n"
            findings_text += f"Evidence: {finding.get('evidence', '')}\n"
            findings_text += f"Significance: {finding.get('significance', '')}\n\n"
        
        # Create visualizations text
        vis_text = "\n".join([f"- {vis.get('title', '')}: {vis.get('description', '')}" 
                             for vis in visualizations[:5]])  # Limit to first 5 visualizations
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following key findings and visualizations:
        
        Key Findings:
        {findings_text}
        
        Visualizations:
        {vis_text}
        
        Write a detailed findings section (1000-1200 words) that:
        
        1. Presents the key findings in a logical, coherent structure
        2. Describes the patterns, trends, and relationships discovered in the literature
        3. Incorporates references to the visualizations and how they illustrate the findings
        4. Connects the findings directly to the research question
        5. Presents the evidence supporting each finding
        6. Maintains objectivity while highlighting the significance of each finding
        
        The findings section should be comprehensive, well-organized, and focused on presenting the results rather than interpreting them (which will come in the discussion section). Use clear, precise language and maintain a scholarly tone throughout.
        
        The section should demonstrate:
        - Rigor in the presentation of findings
        - Clarity in explaining complex results
        - Transparency in how findings were derived from the literature
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the findings
                findings_section = response.choices[0].message.content.strip()
                
                self.report_sections['findings'] = findings_section
                
                logger.info("Findings generation completed")
                
            except Exception as e:
                logger.error(f"Error generating findings: {str(e)}")
                self.report_sections['findings'] = "# Findings\n\nError generating findings section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder findings.")
            self.report_sections['findings'] = "# Findings\n\nPlaceholder for findings section."
    
    def _generate_discussion(self):
        """Generate the discussion section of the report."""
        logger.info("Generating discussion")
        
        # Prepare data for the prompt
        key_findings = self.analysis_results.get('key_findings', [])
        answers = self.interpretation.get('answers_to_research_question', [])
        limitations = self.interpretation.get('limitations', [])
        
        # Create findings summary
        findings_summary = "\n".join([f"- {finding.get('statement', '')}" 
                                     for finding in key_findings])
        
        # Create answers text
        answers_text = ""
        for i, answer in enumerate(answers):
            answers_text += f"Answer {i+1}: {answer.get('statement', '')}\n"
            answers_text += f"Evidence: {answer.get('evidence', '')}\n"
            answers_text += f"Confidence: {answer.get('confidence', '')}\n\n"
        
        # Create limitations text
        limitations_text = "\n".join([f"- {limitation.get('description', '')}" 
                                    for limitation in limitations])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following key findings, answers to the research question, and limitations:
        
        Key Findings:
        {findings_summary}
        
        Answers to Research Question:
        {answers_text}
        
        Limitations:
        {limitations_text}
        
        Write a comprehensive discussion section (1000-1200 words) that:
        
        1. Interprets the key findings in the context of the research question
        2. Discusses how the findings compare to existing literature and theories
        3. Addresses the limitations of the study and their implications
        4. Evaluates the strengths and weaknesses of using LitRevRA for literature review
        5. Discusses the broader implications of the findings for theory, practice, and research
        6. Reflects on unexpected or surprising results
        
        The discussion should demonstrate critical thinking, nuanced interpretation, and scholarly depth. It should go beyond merely summarizing the findings to explore their meaning, context, and implications.
        
        The section should emphasize:
        - Novelty of the insights gained through this approach
        - Rigor in the interpretation and contextualization of findings
        - Relevance of the findings to the broader field
        - Transparency about limitations and potential biases
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the discussion
                discussion = response.choices[0].message.content.strip()
                
                self.report_sections['discussion'] = discussion
                
                logger.info("Discussion generation completed")
                
            except Exception as e:
                logger.error(f"Error generating discussion: {str(e)}")
                self.report_sections['discussion'] = "# Discussion\n\nError generating discussion section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder discussion.")
            self.report_sections['discussion'] = "# Discussion\n\nPlaceholder for discussion section."
    
    def _generate_future_directions(self):
        """Generate the future directions section of the report."""
        logger.info("Generating future directions")
        
        # Prepare data for the prompt
        future_work = self.interpretation.get('future_work', [])
        recommendations = self.interpretation.get('recommendations', [])
        
        # Create future work text
        future_work_text = ""
        for i, work in enumerate(future_work):
            future_work_text += f"Direction {i+1}: {work.get('title', '')}\n"
            if 'research_questions' in work and isinstance(work['research_questions'], list):
                future_work_text += "Research Questions:\n"
                for q in work['research_questions']:
                    future_work_text += f"- {q}\n"
            future_work_text += f"Methodologies: {work.get('methodologies', '')}\n"
            future_work_text += f"Impact: {work.get('impact', '')}\n\n"
        
        # Create recommendations text
        recommendations_text = ""
        for i, rec in enumerate(recommendations):
            recommendations_text += f"Recommendation {i+1}: {rec.get('statement', '')}\n"
            recommendations_text += f"Rationale: {rec.get('rationale', '')}\n\n"
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following future work directions and recommendations:
        
        Future Research Directions:
        {future_work_text}
        
        Recommendations:
        {recommendations_text}
        
        Write a forward-looking future directions section (800-1000 words) that:
        
        1. Outlines promising avenues for future research in this field
        2. Discusses specific research questions that could be pursued
        3. Suggests methodological approaches for addressing these questions
        4. Explains how these directions build upon the current findings and address identified gaps
        5. Discusses potential applications and implications of the recommended future work
        6. Includes recommendations for practitioners and researchers in the field
        
        The future directions section should be innovative, practical, and well-grounded in the findings of the literature review. It should provide clear guidance for advancing knowledge in the field.
        
        The section should emphasize:
        - Novelty of the proposed research directions
        - Rigor in how these directions connect to the identified gaps and findings
        - Relevance of these directions to advancing knowledge in the field
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the future directions
                future_directions = response.choices[0].message.content.strip()
                
                self.report_sections['future_directions'] = future_directions
                
                logger.info("Future directions generation completed")
                
            except Exception as e:
                logger.error(f"Error generating future directions: {str(e)}")
                self.report_sections['future_directions'] = "# Future Directions\n\nError generating future directions section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder future directions.")
            self.report_sections['future_directions'] = "# Future Directions\n\nPlaceholder for future directions section."
    
    def _generate_conclusion(self):
        """Generate the conclusion section of the report."""
        logger.info("Generating conclusion")
        
        # Prepare data for the prompt
        summary = self.interpretation.get('summary', '')
        answers = self.interpretation.get('answers_to_research_question', [])
        
        # Create answers text
        answers_text = "\n".join([f"- {answer.get('statement', '')}" 
                                 for answer in answers])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are assisting a team of researchers in writing a high-quality scientific paper based on a comprehensive literature review conducted using LitRevRA (Literature Review Research Assistant), an automated tool for synthesizing research literature.
        
        Based on the following summary and answers to the research question:
        
        Summary:
        {summary[:500]}...
        
        Answers to Research Question:
        {answers_text}
        
        Write a concise, impactful conclusion section (400-500 words) that:
        
        1. Summarizes the key findings and their significance
        2. Directly addresses the research question
        3. Highlights the main contributions of this work to the field
        4. Discusses the broader implications and importance of the findings
        5. Emphasizes the value of using LitRevRA for literature reviews
        6. Ends with a compelling statement about the significance of this work
        
        The conclusion should be clear, precise, and leave readers with a strong understanding of the importance of this research. It should tie together the various elements of the paper into a cohesive closing statement.
        
        The section should emphasize:
        - Novelty of the insights and approach
        - Rigor of the methodology and analysis
        - Relevance to the field and future research
        """
        
        if self.openai_api_key:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a scientific writing assistant helping researchers create a high-quality scientific paper based on a comprehensive literature review."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the conclusion
                conclusion = response.choices[0].message.content.strip()
                
                self.report_sections['conclusion'] = conclusion
                
                logger.info("Conclusion generation completed")
                
            except Exception as e:
                logger.error(f"Error generating conclusion: {str(e)}")
                self.report_sections['conclusion'] = "# Conclusion\n\nError generating conclusion section."
        else:
            logger.warning("OpenAI API key not available. Using placeholder conclusion.")
            self.report_sections['conclusion'] = "# Conclusion\n\nPlaceholder for conclusion section."
    
    def _generate_references(self):
        """Generate the references section of the report."""
        logger.info("Generating references")
        
        # For a real implementation, we would extract actual references from the literature data
        # For now, we'll create a placeholder
        references = "# References\n\n"
        references += "This section would contain the formatted references used in the paper, "
        references += "following an appropriate citation style (e.g., APA, IEEE).\n\n"
        references += "In a full implementation, this would be automatically generated from the "
        references += "literature data and citations used throughout the report."
        
        self.report_sections['references'] = references
        
        logger.info("References generation completed")
    
    def _assemble_full_report(self):
        """
        Assemble all sections into a complete report.
        
        Returns:
            str: The complete report
        """
        logger.info("Assembling full report")
        
        # Define the order of sections
        section_order = [
            'abstract',
            'introduction',
            'methodology',
            'literature_review',
            'findings',
            'discussion',
            'future_directions',
            'conclusion',
            'references'
        ]
        
        # Create the report title
        title = f"# {self.task_description}\n\n"
        
        # Assemble the sections in order
        report_body = ""
        for section in section_order:
            if section in self.report_sections:
                # Add section content
                if section == 'abstract':
                    report_body += "# Abstract\n\n"
                    report_body += self.report_sections[section] + "\n\n"
                elif not self.report_sections[section].startswith('#'):
                    # Add section header if not already included
                    section_title = section.replace('_', ' ').title()
                    report_body += f"# {section_title}\n\n"
                    report_body += self.report_sections[section] + "\n\n"
                else:
                    # Section already has a header
                    report_body += self.report_sections[section] + "\n\n"
        
        # Combine title and body
        full_report = title + report_body
        
        logger.info("Full report assembly completed")
        return full_report

def write_report(config, literature_data, research_plan, ideas, scoring_framework, analysis_results, interpretation):
    """
    Main function to write a comprehensive research report.
    
    Args:
        config (dict): Configuration settings
        literature_data (dict): Collected literature data
        research_plan (dict): Research plan from Phase 2
        ideas (dict): Identified research ideas
        scoring_framework (dict): Scoring framework from Phase 4
        analysis_results (dict): Results from analysis phase
        interpretation (dict): Interpretation and recommendations
        
    Returns:
        str: The complete report in the specified format
    """
    # Create a report writer
    writer = ReportWriter(
        config, 
        literature_data, 
        research_plan, 
        ideas, 
        scoring_framework, 
        analysis_results, 
        interpretation
    )
    
    # Generate the report
    report = writer.generate_report()
    
    return report

# Test code to verify it works
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    config = {
        "task_description": "Literature Review on Automated Software Testing Techniques",
        "research_question": "What are the most effective automated testing techniques for web applications?",
        "api_keys": {
            "openai": openai.api_key
        }
    }
    literature_data = {}
    research_plan = {
        "literature_overview": {
            "detailed_overview": "This is a placeholder for the literature overview."
        },
        "research_gaps": [
            {"description": "Gap in mobile testing automation"},
            {"description": "Limited research on performance testing tools"}
        ],
        "methodology": {
            "full_text": "This is a placeholder for the methodology details."
        }
    }
    ideas = {
        "extracted_ideas": [
            {"title": "AI-Based Test Case Generation", "description": "Using machine learning to automatically generate test cases."}
        ],
        "novel_ideas": [
            {"title": "Cross-platform Testing Framework", "description": "Unified framework for testing web, mobile and desktop applications."}
        ]
    }
    scoring_framework = {}
    analysis_results = {
        "key_findings": [
            {"statement": "Selenium remains the most widely used tool for web testing automation", 
             "evidence": "Found in 75% of the reviewed papers", 
             "significance": "Indicates industry consensus on reliable tools"}
        ],
        "trends": [
            {"title": "Increasing use of AI in test automation", 
             "description": "More papers are exploring machine learning for test optimization"}
        ],
        "visualizations": [
            {"title": "Testing Tools Usage", 
             "description": "Comparison of popularity of different testing frameworks"}
        ]
    }
    interpretation = {
        "summary": "This literature review examined automated testing techniques for web applications. The findings indicate a growing trend toward AI-assisted testing and a continued reliance on established frameworks like Selenium.",
        "recommendations": [
            {"statement": "Integrate AI-based test generation with traditional frameworks", 
             "rationale": "Combines strengths of both approaches"}
        ],
        "limitations": [
            {"description": "Limited studies on recent frameworks released after 2022"}
        ],
        "future_work": [
            {"title": "Mobile-Web Hybrid Testing", 
             "research_questions": ["How can testing frameworks be optimized for progressive web apps?"],
             "methodologies": "Comparative analysis of existing tools",
             "impact": "Would address growing market of hybrid applications"}
        ],
        "answers_to_research_question": [
            {"statement": "Selenium combined with AI-based test case generation shows the highest effectiveness", 
             "evidence": "Multiple studies show reduced testing time and higher defect detection rates",
             "confidence": "High"}
        ]
    }