#!/usr/bin/env python3
# Phase 4: Relevancy Scoring and Summarizing Framework Development

import os
import json
import logging
from datetime import datetime

import openai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("LitRevRA.Phase4")

class RelevancyScorer:
    """Class to develop a framework for scoring and summarizing research relevancy."""
    
    def __init__(self, config, ideas, literature_data):
        """
        Initialize the RelevancyScorer.
        
        Args:
            config (dict): Configuration settings
            ideas (dict): Identified research ideas
            literature_data (dict): Collected literature data
        """
        self.config = config
        self.ideas = ideas
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
        
        # Initialize the scoring framework
        self.scoring_framework = {
            "task_description": self.task_description,
            "research_question": self.research_question,
            "timestamp": datetime.now().isoformat(),
            "evaluation_criteria": [],
            "paper_scores": [],
            "idea_scores": [],
            "relevancy_matrix": {},
            "summary_tables": {},
            "visualization_data": {}
        }
    
    def develop_scoring_framework(self):
        """
        Develop a framework for scoring and summarizing research relevancy.
        
        Returns:
            dict: The scoring framework
        """
        logger.info("Starting development of relevancy scoring framework")
        
        # Step 1: Define evaluation criteria
        self._define_evaluation_criteria()
        
        # Step 2: Score papers based on the criteria
        self._score_papers()
        
        # Step 3: Score research ideas based on the criteria
        self._score_ideas()
        
        # Step 4: Create a relevancy matrix
        self._create_relevancy_matrix()
        
        # Step 5: Generate summary tables
        self._generate_summary_tables()
        
        # Step 6: Prepare visualization data
        self._prepare_visualization_data()
        
        return self.scoring_framework
    
    def _define_evaluation_criteria(self):
        """Define evaluation criteria for scoring papers and ideas."""
        logger.info("Defining evaluation criteria")
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        I need to develop a framework to evaluate the relevance and quality of academic papers and research ideas for this task.
        
        Please create a set of 5-7 specific evaluation criteria that can be used to assess:
        1. The relevance of a paper to our research question
        2. The quality and soundness of the research methodology
        3. The impact and significance of the findings
        4. The innovation and originality of the approach
        
        For each criterion:
        1. Provide a clear name
        2. Give a detailed description of what it measures
        3. Explain how it should be scored (on a scale of 1-10)
        4. List specific indicators or features to look for
        5. Provide examples of what would constitute high, medium, and low scores
        
        Format your response with clear sections for each criterion.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to develop evaluation criteria for academic research."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the criteria
            criteria_text = response.choices[0].message.content.strip()
            
            # Parse the criteria (simple approach)
            criteria = []
            current_criterion = None
            
            for line in criteria_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for criterion headers
                if "Criterion" in line or "CRITERION" in line:
                    # Save the previous criterion
                    if current_criterion and current_criterion.get('name'):
                        criteria.append(current_criterion)
                    
                    # Extract criterion name
                    name = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new criterion
                    current_criterion = {
                        "name": name,
                        "description": "",
                        "scoring_guide": "",
                        "indicators": [],
                        "examples": {
                            "high": "",
                            "medium": "",
                            "low": ""
                        }
                    }
                elif current_criterion:
                    # Parse criterion components
                    if "description" in line.lower():
                        current_criterion["description"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "scoring" in line.lower() or "scale" in line.lower():
                        current_criterion["scoring_guide"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "indicator" in line.lower() or "feature" in line.lower():
                        indicators = line.split(':', 1)[1].strip().split(',') if ':' in line else [line]
                        current_criterion["indicators"].extend([i.strip() for i in indicators])
                    elif "high" in line.lower() and "score" in line.lower():
                        current_criterion["examples"]["high"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "medium" in line.lower() and "score" in line.lower():
                        current_criterion["examples"]["medium"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "low" in line.lower() and "score" in line.lower():
                        current_criterion["examples"]["low"] = line.split(':', 1)[1].strip() if ':' in line else line
            
            # Add the last criterion
            if current_criterion and current_criterion.get('name'):
                criteria.append(current_criterion)
            
            self.scoring_framework['evaluation_criteria'] = criteria
            
            logger.info(f"Defined {len(criteria)} evaluation criteria")
            
        except Exception as e:
            logger.error(f"Error defining evaluation criteria: {str(e)}")
    
    def _score_papers(self):
        """Score papers based on the defined criteria."""
        logger.info("Scoring papers based on criteria")
        
        # Get papers and criteria
        papers = self.literature_data.get('papers', [])
        criteria = self.scoring_framework.get('evaluation_criteria', [])
        
        if not papers or not criteria:
            logger.warning("No papers or criteria available for scoring")
            return
        
        # Process papers in batches
        batch_size = 5
        paper_batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
        
        all_paper_scores = []
        
        for batch_idx, batch in enumerate(paper_batches):
            logger.info(f"Scoring paper batch {batch_idx + 1}/{len(paper_batches)}")
            
            # Create a criteria summary for the prompt
            criteria_summary = "\n".join([f"{i+1}. {criterion.get('name')}: {criterion.get('description')}" 
                                        for i, criterion in enumerate(criteria)])
            
            # Create a batch prompt
            batch_texts = []
            for paper in batch:
                text = f"Title: {paper.get('title')}\n"
                text += f"Authors: {', '.join(paper.get('authors', []))}\n"
                text += f"Abstract: {paper.get('abstract')}\n"
                
                batch_texts.append(text)
            
            prompt = f"""
            Task: {self.task_description}
            Research Question: {self.research_question}
            
            Evaluation Criteria:
            {criteria_summary}
            
            Please score the following papers based on the evaluation criteria. For each paper, provide:
            1. An overall relevance score (1-10)
            2. Individual scores for each criterion (1-10)
            3. A brief justification for each score
            
            Papers to evaluate:
            {"".join([f"--- Paper {i+1} ---\n{text}\n\n" for i, text in enumerate(batch_texts)])}
            
            Format your response with clear sections for each paper, listing all scores and justifications.
            """
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant helping to evaluate academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the scores
                scores_text = response.choices[0].message.content.strip()
                
                # Parse the scores (simple approach)
                paper_sections = scores_text.split("Paper")
                
                for i, section in enumerate(paper_sections[1:]):  # Skip the first empty section
                    if i >= len(batch):
                        break
                        
                    paper = batch[i]
                    
                    # Initialize scores
                    paper_score = {
                        "paper_id": paper.get('id'),
                        "title": paper.get('title'),
                        "overall_score": 0,
                        "criteria_scores": {},
                        "justification": ""
                    }
                    
                    # Extract scores
                    lines = section.split('\n')
                    for line in lines:
                        line = line.strip()
                        
                        if "overall" in line.lower() and any(str(score) in line for score in range(1, 11)):
                            # Extract overall score
                            for score in range(1, 11):
                                if str(score) in line:
                                    paper_score["overall_score"] = score
                                    break
                        elif any(criterion.get('name') in line for criterion in criteria):
                            # Extract criterion score
                            for criterion in criteria:
                                if criterion.get('name') in line:
                                    for score in range(1, 11):
                                        if str(score) in line:
                                            paper_score["criteria_scores"][criterion.get('name')] = {
                                                "score": score,
                                                "justification": line
                                            }
                                            break
                                    break
                    
                    # Extract justification (simplified)
                    paper_score["justification"] = "\n".join(lines)
                    
                    all_paper_scores.append(paper_score)
                
            except Exception as e:
                logger.error(f"Error scoring paper batch {batch_idx + 1}: {str(e)}")
        
        self.scoring_framework['paper_scores'] = all_paper_scores
        
        logger.info(f"Scored {len(all_paper_scores)} papers")
    
    def _score_ideas(self):
        """Score research ideas based on the defined criteria."""
        logger.info("Scoring research ideas based on criteria")
        
        # Get ideas and criteria
        extracted_ideas = self.ideas.get('extracted_ideas', [])
        novel_ideas = self.ideas.get('novel_ideas', [])
        all_ideas = extracted_ideas + novel_ideas
        
        criteria = self.scoring_framework.get('evaluation_criteria', [])
        
        if not all_ideas or not criteria:
            logger.warning("No ideas or criteria available for scoring")
            return
        
        # Process ideas in batches
        batch_size = 5
        idea_batches = [all_ideas[i:i + batch_size] for i in range(0, len(all_ideas), batch_size)]
        
        all_idea_scores = []
        
        for batch_idx, batch in enumerate(idea_batches):
            logger.info(f"Scoring idea batch {batch_idx + 1}/{len(idea_batches)}")
            
            # Create a criteria summary for the prompt
            criteria_summary = "\n".join([f"{i+1}. {criterion.get('name')}: {criterion.get('description')}" 
                                        for i, criterion in enumerate(criteria)])
            
            # Create a batch prompt
            batch_texts = []
            for idea in batch:
                text = f"Title: {idea.get('title')}\n"
                text += f"Description: {idea.get('description')}\n"
                text += f"Approach: {idea.get('approach')}\n"
                text += f"Gap Addressed: {idea.get('gap_addressed')}\n"
                text += f"Impact: {idea.get('impact')}\n"
                
                batch_texts.append(text)
            
            prompt = f"""
            Task: {self.task_description}
            Research Question: {self.research_question}
            
            Evaluation Criteria:
            {criteria_summary}
            
            Please score the following research ideas based on the evaluation criteria. For each idea, provide:
            1. An overall potential score (1-10)
            2. Individual scores for each criterion (1-10)
            3. A brief justification for each score
            
            Ideas to evaluate:
            {"".join([f"--- Idea {i+1} ---\n{text}\n\n" for i, text in enumerate(batch_texts)])}
            
            Format your response with clear sections for each idea, listing all scores and justifications.
            """
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant helping to evaluate research ideas."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the scores
                scores_text = response.choices[0].message.content.strip()
                
                # Parse the scores (simple approach)
                idea_sections = scores_text.split("Idea")
                
                for i, section in enumerate(idea_sections[1:]):  # Skip the first empty section
                    if i >= len(batch):
                        break
                        
                    idea = batch[i]
                    
                    # Initialize scores
                    idea_score = {
                        "idea_title": idea.get('title'),
                        "overall_score": 0,
                        "criteria_scores": {},
                        "justification": ""
                    }
                    
                    # Extract scores
                    lines = section.split('\n')
                    for line in lines:
                        line = line.strip()
                        
                        if "overall" in line.lower() and any(str(score) in line for score in range(1, 11)):
                            # Extract overall score
                            for score in range(1, 11):
                                if str(score) in line:
                                    idea_score["overall_score"] = score
                                    break
                        elif any(criterion.get('name') in line for criterion in criteria):
                            # Extract criterion score
                            for criterion in criteria:
                                if criterion.get('name') in line:
                                    for score in range(1, 11):
                                        if str(score) in line:
                                            idea_score["criteria_scores"][criterion.get('name')] = {
                                                "score": score,
                                                "justification": line
                                            }
                                            break
                                    break
                    
                    # Extract justification (simplified)
                    idea_score["justification"] = "\n".join(lines)
                    
                    all_idea_scores.append(idea_score)
                
            except Exception as e:
                logger.error(f"Error scoring idea batch {batch_idx + 1}: {str(e)}")
        
        self.scoring_framework['idea_scores'] = all_idea_scores
        
        logger.info(f"Scored {len(all_idea_scores)} ideas")
    
    def _create_relevancy_matrix(self):
        """Create a relevancy matrix between papers and ideas."""
        logger.info("Creating relevancy matrix")
        
        # Get papers, ideas, and scores
        papers = self.literature_data.get('papers', [])
        extracted_ideas = self.ideas.get('extracted_ideas', [])
        novel_ideas = self.ideas.get('novel_ideas', [])
        all_ideas = extracted_ideas + novel_ideas
        
        paper_scores = self.scoring_framework.get('paper_scores', [])
        idea_scores = self.scoring_framework.get('idea_scores', [])
        
        if not papers or not all_ideas or not paper_scores or not idea_scores:
            logger.warning("Insufficient data for creating relevancy matrix")
            return
        
        # Use TF-IDF to compute relevancy scores
        try:
            # Prepare documents for papers and ideas
            paper_texts = [f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in papers]
            idea_texts = [f"{idea.get('title', '')} {idea.get('description', '')}" for idea in all_ideas]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            
            # Fit and transform papers
            paper_vectors = vectorizer.fit_transform(paper_texts)
            
            # Transform ideas (using the same vocabulary)
            idea_vectors = vectorizer.transform(idea_texts)
            
            # Compute cosine similarity
            similarity = cosine_similarity(paper_vectors, idea_vectors)
            
            # Create the relevancy matrix
            matrix = {
                "papers": [{"id": paper.get('id'), "title": paper.get('title')} for paper in papers],
                "ideas": [{"title": idea.get('title')} for idea in all_ideas],
                "matrix": similarity.tolist()
            }
            
            self.scoring_framework['relevancy_matrix'] = matrix
            
            logger.info(f"Created relevancy matrix of size {similarity.shape}")
            
        except Exception as e:
            logger.error(f"Error creating relevancy matrix: {str(e)}")
    
    def _generate_summary_tables(self):
        """Generate summary tables from the scoring framework."""
        logger.info("Generating summary tables")
        
        # Get scores
        paper_scores = self.scoring_framework.get('paper_scores', [])
        idea_scores = self.scoring_framework.get('idea_scores', [])
        
        if not paper_scores and not idea_scores:
            logger.warning("No scores available for generating summary tables")
            return
        
        # Generate paper summary table
        if paper_scores:
            paper_table = {
                "columns": ["Paper Title", "Overall Score"],
                "data": []
            }
            
            # Add criteria names to columns
            if paper_scores and paper_scores[0].get('criteria_scores'):
                criteria_names = list(paper_scores[0]['criteria_scores'].keys())
                paper_table["columns"].extend(criteria_names)
            
            # Add data rows
            for score in paper_scores:
                row = [score.get('title'), score.get('overall_score')]
                
                # Add criteria scores
                for criterion in paper_table["columns"][2:]:
                    row.append(score.get('criteria_scores', {}).get(criterion, {}).get('score', 0))
                
                paper_table["data"].append(row)
            
            self.scoring_framework['summary_tables']['papers'] = paper_table
        
        # Generate idea summary table
        if idea_scores:
            idea_table = {
                "columns": ["Idea Title", "Overall Score"],
                "data": []
            }
            
            # Add criteria names to columns
            if idea_scores and idea_scores[0].get('criteria_scores'):
                criteria_names = list(idea_scores[0]['criteria_scores'].keys())
                idea_table["columns"].extend(criteria_names)
            
            # Add data rows
            for score in idea_scores:
                row = [score.get('idea_title'), score.get('overall_score')]
                
                # Add criteria scores
                for criterion in idea_table["columns"][2:]:
                    row.append(score.get('criteria_scores', {}).get(criterion, {}).get('score', 0))
                
                idea_table["data"].append(row)
            
            self.scoring_framework['summary_tables']['ideas'] = idea_table
        
        logger.info("Generated summary tables")
    
    def _prepare_visualization_data(self):
        """Prepare data for visualizations."""
        logger.info("Preparing visualization data")
        
        # Get scores and matrix
        paper_scores = self.scoring_framework.get('paper_scores', [])
        idea_scores = self.scoring_framework.get('idea_scores', [])
        relevancy_matrix = self.scoring_framework.get('relevancy_matrix', {})
        
        if not paper_scores and not idea_scores:
            logger.warning("No scores available for visualization data")
            return
        
        visualization_data = {}
        
        # Paper scores visualization data
        if paper_scores:
            # Top papers by overall score
            top_papers = sorted(paper_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:10]
            
            paper_viz = {
                "top_papers": {
                    "titles": [paper.get('title')[:30] + '...' if len(paper.get('title', '')) > 30 else paper.get('title', '') for paper in top_papers],
                    "scores": [paper.get('overall_score', 0) for paper in top_papers]
                }
            }
            
            # Criteria comparison
            if top_papers and top_papers[0].get('criteria_scores'):
                criteria = list(top_papers[0]['criteria_scores'].keys())
                
                criteria_data = {criterion: [] for criterion in criteria}
                for paper in top_papers[:5]:  # Limit to top 5 for readability
                    for criterion in criteria:
                        criteria_data[criterion].append(paper.get('criteria_scores', {}).get(criterion, {}).get('score', 0))
                
                paper_viz["criteria_comparison"] = {
                    "papers": [paper.get('title')[:20] + '...' if len(paper.get('title', '')) > 20 else paper.get('title', '') for paper in top_papers[:5]],
                    "criteria": criteria,
                    "data": criteria_data
                }
            
            visualization_data['papers'] = paper_viz
        
        # Idea scores visualization data
        if idea_scores:
            # Top ideas by overall score
            top_ideas = sorted(idea_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:10]
            
            idea_viz = {
                "top_ideas": {
                    "titles": [idea.get('idea_title')[:30] + '...' if len(idea.get('idea_title', '')) > 30 else idea.get('idea_title', '') for idea in top_ideas],
                    "scores": [idea.get('overall_score', 0) for idea in top_ideas]
                }
            }
            
            # Criteria comparison
            if top_ideas and top_ideas[0].get('criteria_scores'):
                criteria = list(top_ideas[0]['criteria_scores'].keys())
                
                criteria_data = {criterion: [] for criterion in criteria}
                for idea in top_ideas[:5]:  # Limit to top 5 for readability
                    for criterion in criteria:
                        criteria_data[criterion].append(idea.get('criteria_scores', {}).get(criterion, {}).get('score', 0))
                
                idea_viz["criteria_comparison"] = {
                    "ideas": [idea.get('idea_title')[:20] + '...' if len(idea.get('idea_title', '')) > 20 else idea.get('idea_title', '') for idea in top_ideas[:5]],
                    "criteria": criteria,
                    "data": criteria_data
                }
            
            visualization_data['ideas'] = idea_viz
        
        # Relevancy matrix visualization data
        if relevancy_matrix and 'matrix' in relevancy_matrix:
            matrix = np.array(relevancy_matrix['matrix'])
            
            if matrix.size > 0:
                # Get top paper-idea pairs
                flat_indices = matrix.flatten().argsort()[-10:][::-1]
                paper_indices, idea_indices = np.unravel_index(flat_indices, matrix.shape)
                
                top_pairs = []
                for i, j in zip(paper_indices, idea_indices):
                    if i < len(relevancy_matrix.get('papers', [])) and j < len(relevancy_matrix.get('ideas', [])):
                        paper_title = relevancy_matrix['papers'][i]['title']
                        idea_title = relevancy_matrix['ideas'][j]['title']
                        relevancy = matrix[i, j]
                        
                        top_pairs.append({
                            "paper": paper_title[:30] + '...' if len(paper_title) > 30 else paper_title,
                            "idea": idea_title[:30] + '...' if len(idea_title) > 30 else idea_title,
                            "relevancy": relevancy
                        })
                
                visualization_data['relevancy'] = {
                    "top_pairs": top_pairs,
                    "heatmap": {
                        "papers": [paper['title'][:20] + '...' if len(paper['title']) > 20 else paper['title'] for paper in relevancy_matrix['papers'][:10]],
                        "ideas": [idea['title'][:20] + '...' if len(idea['title']) > 20 else idea['title'] for idea in relevancy_matrix['ideas'][:10]],
                        "data": matrix[:10, :10].tolist() if matrix.shape[0] > 10 and matrix.shape[1] > 10 else matrix.tolist()
                    }
                }
        
        self.scoring_framework['visualization_data'] = visualization_data
        
        logger.info("Prepared visualization data")

def develop_scoring_framework(config, ideas, literature_data):
    """
    Main function to develop a relevancy scoring framework.
    
    Args:
        config (dict): Configuration settings
        ideas (dict): Identified research ideas
        literature_data (dict): Collected literature data
        
    Returns:
        dict: The scoring framework
    """
    # Create a relevancy scorer
    scorer = RelevancyScorer(config, ideas, literature_data)
    
    # Develop the scoring framework
    scoring_framework = scorer.develop_scoring_framework()
    
    return scoring_framework

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
    
    # Load ideas
    ideas_path = "identified_ideas.json"
    if len(sys.argv) > 3:
        ideas_path = sys.argv[3]
    
    with open(ideas_path, 'r', encoding='utf-8') as f:
        ideas = json.load(f)
    
    # Develop scoring framework
    scoring_framework = develop_scoring_framework(config, ideas, literature_data)
    
    # Save output
    output_path = "scoring_framework.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scoring_framework, f, indent=2, ensure_ascii=False)
    
    print(f"Scoring framework saved to {output_path}")
