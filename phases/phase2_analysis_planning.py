#!/usr/bin/env python3
# Phase 2: Analysis and Planning

import os
import json
import logging
from datetime import datetime

import openai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

logger = logging.getLogger("LitRevRA.Phase2")

class ResearchPlanner:
    """Class to analyze literature data and develop a research plan."""
    
    def __init__(self, config, literature_data):
        """
        Initialize the ResearchPlanner.
        
        Args:
            config (dict): Configuration settings
            literature_data (dict): Collected literature data
        """
        self.config = config
        self.literature_data = literature_data
        
        # Set up OpenAI API
        openai.api_key = config['api_keys']['openai']
        
        # Load the task description and research question
        self.task_description = config.get('task_description', '')
        self.research_question = config.get('research_question', '')
        
        # Set model parameters
        model_settings = config.get('model_settings', {})
        self.model = model_settings.get('model', 'gpt-4')
        self.temperature = model_settings.get('temperature', 0.7)
        self.max_tokens = model_settings.get('max_tokens', 3000)
        
        # Initialize the research plan
        self.research_plan = {
            "task_description": self.task_description,
            "research_question": self.research_question,
            "timestamp": datetime.now().isoformat(),
            "literature_overview": {},
            "research_gaps": [],
            "methodology": {},
            "experimental_design": {},
            "timeline": [],
            "resources_needed": []
        }
    
    def analyze_and_plan(self):
        """
        Analyze the literature and develop a research plan.
        
        Returns:
            dict: The research plan
        """
        logger.info("Starting literature analysis and research planning")
        
        # Step 1: Perform basic statistical analysis of the literature
        self._analyze_literature()
        
        # Step 2: Identify research gaps
        self._identify_research_gaps()
        
        # Step 3: Develop methodology
        self._develop_methodology()
        
        # Step 4: Design experiments
        self._design_experiments()
        
        # Step 5: Create timeline and resource plan
        self._create_timeline_and_resources()
        
        return self.research_plan
    
    def _analyze_literature(self):
        """Analyze the collected literature to extract key insights."""
        logger.info("Analyzing literature")
        
        papers = self.literature_data.get('papers', [])
        summaries = self.literature_data.get('summaries', [])
        
        if not papers:
            logger.warning("No papers to analyze")
            return
        
        # Basic statistics
        num_papers = len(papers)
        publication_years = []
        sources = []
        all_authors = []
        
        for paper in papers:
            # Extract publication year
            if 'published' in paper and paper['published']:
                try:
                    if '-' in paper['published']:
                        year = int(paper['published'].split('-')[0])
                    else:
                        year = int(paper['published'])
                    publication_years.append(year)
                except (ValueError, TypeError):
                    pass
            
            # Extract source
            if 'source' in paper:
                sources.append(paper['source'])
            
            # Extract authors
            if 'authors' in paper:
                all_authors.extend(paper['authors'])
        
        # Calculate statistics
        year_counts = Counter(publication_years)
        source_counts = Counter(sources)
        author_counts = Counter(all_authors)
        
        # Find most common authors
        top_authors = [{"name": name, "count": count} 
                      for name, count in author_counts.most_common(10)]
        
        # Create the literature overview
        self.research_plan['literature_overview'] = {
            "num_papers": num_papers,
            "publication_years": dict(sorted(year_counts.items())),
            "sources": dict(source_counts),
            "top_authors": top_authors
        }
        
        # Extract key topics and methods using TF-IDF
        if papers:
            self._extract_key_topics_and_methods(papers)
        
        # Use OpenAI to generate a comprehensive literature overview
        self._generate_literature_overview_with_openai()
        
        logger.info("Literature analysis completed")
    
    def _extract_key_topics_and_methods(self, papers):
        """
        Extract key topics and methods from papers using TF-IDF.
        
        Args:
            papers (list): List of paper dictionaries
        """
        try:
            # Combine titles and abstracts
            documents = []
            for paper in papers:
                doc = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                documents.append(doc)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each term across all documents
            tfidf_sums = np.sum(tfidf_matrix.toarray(), axis=0)
            
            # Get top terms
            top_indices = tfidf_sums.argsort()[-30:][::-1]
            top_terms = [{"term": feature_names[i], "score": float(tfidf_sums[i])} 
                        for i in top_indices]
            
            self.research_plan['literature_overview']['key_terms'] = top_terms
            
        except Exception as e:
            logger.error(f"Error extracting key topics and methods: {str(e)}")
    
    def _generate_literature_overview_with_openai(self):
        """Generate a comprehensive literature overview using OpenAI."""
        logger.info("Generating literature overview with OpenAI")
        
        # Prepare data for the prompt
        papers = self.literature_data.get('papers', [])
        summaries = self.literature_data.get('summaries', [])
        
        if not papers or not summaries:
            logger.warning("Insufficient data for generating literature overview")
            return
        
        # Create a mapping of paper IDs to summaries
        summary_map = {s.get('paper_id'): s for s in summaries if 'paper_id' in s}
        
        # Prepare paper information for the prompt
        paper_info = []
        for paper in papers[:15]:  # Limit to 15 papers to avoid token limits
            paper_id = paper.get('id')
            paper_summary = summary_map.get(paper_id, {}).get('summary', '')
            
            info = f"Title: {paper.get('title')}\n"
            info += f"Authors: {', '.join(paper.get('authors', []))}\n"
            info += f"Abstract: {paper.get('abstract')}\n"
            if paper_summary:
                info += f"Summary: {paper_summary}\n"
            
            paper_info.append(info)
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        You are conducting a critical synthesis of the literature as part of the Analysis & Planning phase of research. Having gathered a substantial corpus of relevant papers, your task is to extract key insights and lay the groundwork for focused analysis. This involves not just summarizing individual papers, but critically synthesizing the literature to identify overarching themes, prominent research questions, and existing theoretical frameworks.
        
        Based on the following academic papers, provide a comprehensive synthesis of the literature that:
        
        1. Identifies and explains overarching themes and patterns across the literature
        2. Maps the intellectual landscape of the field by showing connections and divergences between studies
        3. Pinpoints areas of scholarly consensus and controversy
        4. Evaluates the methodological approaches employed across the field
        5. Analyzes the evolution of key theoretical frameworks and how they've shaped the research domain
        6. Identifies potential areas of overlap or contradiction between different research streams
        
        Papers:
        {"".join([f"--- Paper {i+1} ---\n{info}\n\n" for i, info in enumerate(paper_info)])}
        
        Your synthesis should be thorough and critical, going beyond summarization to provide genuine insights about the state of the field. Structure your response with clear sections addressing each of the points above, and highlight potential avenues for deeper investigation in later research phases.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping a PhD candidate to critically synthesize academic literature and develop a focused research plan."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Add the generated overview to the research plan
            overview_text = response.choices[0].message.content.strip()
            self.research_plan['literature_overview']['detailed_overview'] = overview_text
            
            logger.info("Generated literature overview with OpenAI")
            
        except Exception as e:
            logger.error(f"Error generating literature overview: {str(e)}")
    
    def _identify_research_gaps(self):
        """Identify research gaps in the literature using OpenAI."""
        logger.info("Identifying research gaps")
        
        # Create a prompt for identifying research gaps
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Literature Overview:
        {self.research_plan['literature_overview'].get('detailed_overview', '')}
        
        Based on the literature overview above, identify 3-5 specific research gaps that present opportunities for new research.
        
        For each research gap:
        1. Clearly describe the gap
        2. Explain why addressing this gap is important
        3. Suggest a specific research direction that could address this gap
        
        Format your response as a numbered list of research gaps, with each gap containing the three elements above.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to identify research gaps and opportunities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the research gaps
            gaps_text = response.choices[0].message.content.strip()
            
            # Parse the gaps (simple approach)
            gaps = []
            current_gap = None
            
            for line in gaps_text.split('\n'):
                line = line.strip()
                
                # Start of a new gap
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    if current_gap:
                        gaps.append(current_gap)
                    
                    gap_number = int(line[0])
                    gap_description = line[2:].strip()
                    current_gap = {
                        "number": gap_number,
                        "description": gap_description,
                        "importance": "",
                        "research_direction": ""
                    }
                elif current_gap:
                    # Add to current gap description
                    if not current_gap["importance"] and ("important" in line.lower() or "significance" in line.lower()):
                        current_gap["importance"] = line
                    elif not current_gap["research_direction"] and ("direction" in line.lower() or "approach" in line.lower() or "address" in line.lower()):
                        current_gap["research_direction"] = line
                    else:
                        current_gap["description"] += " " + line
            
            # Add the last gap
            if current_gap:
                gaps.append(current_gap)
            
            self.research_plan['research_gaps'] = gaps
            
            logger.info(f"Identified {len(gaps)} research gaps")
            
        except Exception as e:
            logger.error(f"Error identifying research gaps: {str(e)}")
    
    def _develop_methodology(self):
        """Develop a methodology for the research using OpenAI."""
        logger.info("Developing research methodology")
        
        # Prepare data for the prompt
        research_gaps = self.research_plan.get('research_gaps', [])
        
        if not research_gaps:
            logger.warning("No research gaps identified for methodology development")
            return
        
        # Create a gaps summary for the prompt
        gaps_summary = "\n".join([f"{gap.get('number')}. {gap.get('description')}" 
                                 for gap in research_gaps])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Research Gaps:
        {gaps_summary}
        
        Based on the identified research gaps and the literature overview, develop a comprehensive methodology for conducting this research. 
        
        Include the following components:
        
        1. Research Approach: The overall approach to address the research question
        2. Data Collection Methods: How data will be collected or generated
        3. Analysis Techniques: Statistical or computational methods for analysis
        4. Evaluation Framework: How results will be evaluated
        5. Machine Learning Models: Specific models to be implemented
        6. Validation Strategy: How to ensure the validity of results
        
        For each component, provide a detailed description and justification based on the research gaps and literature.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to develop a research methodology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the methodology
            methodology_text = response.choices[0].message.content.strip()
            
            # Parse the methodology (simple approach)
            methodology = {
                "full_text": methodology_text,
                "components": {}
            }
            
            # Extract components
            current_component = None
            component_text = ""
            
            for line in methodology_text.split('\n'):
                line = line.strip()
                
                # Check for component headers
                component_mapping = {
                    "Research Approach": "research_approach",
                    "Data Collection": "data_collection",
                    "Analysis Techniques": "analysis_techniques",
                    "Evaluation Framework": "evaluation_framework",
                    "Machine Learning Models": "ml_models",
                    "Validation Strategy": "validation_strategy"
                }
                
                found_component = False
                for header, key in component_mapping.items():
                    if any(line.startswith(f"{i}. {header}") or line == header or line.startswith(f"{header}:") 
                          for i in range(1, 7)):
                        if current_component and component_text:
                            methodology["components"][current_component] = component_text.strip()
                        
                        current_component = key
                        component_text = ""
                        found_component = True
                        break
                
                if not found_component and current_component:
                    component_text += line + "\n"
            
            # Add the last component
            if current_component and component_text:
                methodology["components"][current_component] = component_text.strip()
            
            self.research_plan['methodology'] = methodology
            
            logger.info("Developed research methodology")
            
        except Exception as e:
            logger.error(f"Error developing methodology: {str(e)}")
    
    def _design_experiments(self):
        """Design experiments for the research using OpenAI."""
        logger.info("Designing experiments")
        
        # Prepare data for the prompt
        methodology = self.research_plan.get('methodology', {})
        
        if not methodology:
            logger.warning("No methodology available for experiment design")
            return
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Methodology:
        {methodology.get('full_text', '')}
        
        Based on the methodology described above, design detailed experiments to address the research question. 
        
        Include the following in your experimental design:
        
        1. Experiment Objectives: Clear goals for each experiment
        2. Datasets: Specific datasets to be used, with justification
        3. Models and Implementations: Technical details of models to implement
        4. Variables: Independent and dependent variables to measure
        5. Evaluation Metrics: How performance will be measured
        6. Expected Outcomes: What results might be anticipated
        7. Potential Challenges: Issues that might arise and how to address them
        
        Format your response with clear sections for each experiment, including all the elements above.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to design experiments for a research project."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the experimental design
            design_text = response.choices[0].message.content.strip()
            
            # Store the full experimental design
            self.research_plan['experimental_design'] = {
                "full_text": design_text,
                "experiments": []
            }
            
            # Parse experiments (simplified approach)
            experiments = []
            current_experiment = None
            
            for line in design_text.split('\n'):
                line = line.strip()
                
                # Check for experiment headers (assuming they're titled as Experiment 1, etc.)
                if line.startswith(("Experiment", "EXPERIMENT", "Study", "STUDY")):
                    if current_experiment:
                        experiments.append(current_experiment)
                    
                    current_experiment = {
                        "title": line,
                        "description": "",
                        "components": {}
                    }
                elif current_experiment:
                    # Check for component headers
                    component_mapping = {
                        "Objective": "objectives",
                        "Dataset": "datasets",
                        "Model": "models",
                        "Variable": "variables",
                        "Evaluation Metric": "metrics",
                        "Expected Outcome": "outcomes",
                        "Potential Challenge": "challenges"
                    }
                    
                    found_component = False
                    for header, key in component_mapping.items():
                        if header in line or header.lower() in line:
                            current_component = key
                            current_experiment["components"][current_component] = line
                            found_component = True
                            break
                    
                    if not found_component:
                        if line and not line.isspace():
                            # Add to the current experiment description or the last component
                            if any(key in current_experiment["components"] for key in component_mapping.values()):
                                last_key = list(current_experiment["components"].keys())[-1]
                                current_experiment["components"][last_key] += " " + line
                            else:
                                current_experiment["description"] += " " + line
            
            # Add the last experiment
            if current_experiment:
                experiments.append(current_experiment)
            
            self.research_plan['experimental_design']['experiments'] = experiments
            
            logger.info(f"Designed {len(experiments)} experiments")
            
        except Exception as e:
            logger.error(f"Error designing experiments: {str(e)}")
    
    def _create_timeline_and_resources(self):
        """Create a timeline and identify required resources using OpenAI."""
        logger.info("Creating timeline and identifying resources")
        
        # Prepare data for the prompt
        methodology = self.research_plan.get('methodology', {})
        experimental_design = self.research_plan.get('experimental_design', {})
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Methodology Summary:
        {methodology.get('full_text', '')[:500]}...
        
        Experimental Design Summary:
        {experimental_design.get('full_text', '')[:500]}...
        
        Based on the methodology and experimental design, please create:
        
        1. A detailed timeline for conducting this research, including:
           - Major phases of the research
           - Key milestones
           - Estimated duration for each task
           - Dependencies between tasks
        
        2. A comprehensive list of resources needed, including:
           - Computing resources (hardware, software)
           - Data resources
           - Human expertise required
           - Any other necessary tools or resources
        
        Format your response with clear sections for the timeline and resources.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to plan the timeline and resources for a research project."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the timeline and resources
            planning_text = response.choices[0].message.content.strip()
            
            # Parse the response (simplified approach)
            timeline = []
            resources = []
            
            current_section = None
            
            for line in planning_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Detect sections
                if "timeline" in line.lower() or "schedule" in line.lower():
                    current_section = "timeline"
                elif "resource" in line.lower():
                    current_section = "resources"
                elif current_section == "timeline" and any(marker in line.lower() for marker in ["phase", "task", "milestone", "week", "month"]):
                    # This appears to be a timeline item
                    timeline.append(line)
                elif current_section == "resources" and any(marker in line.lower() for marker in ["hardware", "software", "data", "expertise", "tool"]):
                    # This appears to be a resource item
                    resources.append(line)
            
            self.research_plan['timeline'] = timeline
            self.research_plan['resources_needed'] = resources
            
            logger.info("Created timeline and identified resources")
            
        except Exception as e:
            logger.error(f"Error creating timeline and resources: {str(e)}")

def analyze_and_plan(config, literature_data):
    """
    Main function to analyze literature and develop a research plan.
    
    Args:
        config (dict): Configuration settings
        literature_data (dict): Collected literature data
        
    Returns:
        dict: The research plan
    """
    # Create a research planner
    planner = ResearchPlanner(config, literature_data)
    
    # Analyze and plan
    research_plan = planner.analyze_and_plan()
    
    return research_plan

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
    
    # Load literature data
    literature_path = "literature_data.json"
    if len(sys.argv) > 2:
        literature_path = sys.argv[2]
    
    with open(literature_path, 'r', encoding='utf-8') as f:
        literature_data = json.load(f)
    
    # Analyze and plan
    research_plan = analyze_and_plan(config, literature_data)
    
    # Save output
    output_path = "research_plan.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(research_plan, f, indent=2, ensure_ascii=False)
    
    print(f"Research plan saved to {output_path}")