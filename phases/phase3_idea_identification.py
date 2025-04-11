#!/usr/bin/env python3
# Phase 3: Idea Identification and Classification

import os
import json
import logging
from datetime import datetime

import openai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

logger = logging.getLogger("LitRevRA.Phase3")

class IdeaIdentifier:
    """Class to identify and classify research ideas from literature."""
    
    def __init__(self, config, literature_data, research_plan):
        """
        Initialize the IdeaIdentifier.
        
        Args:
            config (dict): Configuration settings
            literature_data (dict): Collected literature data
            research_plan (dict): Research plan from Phase 2
        """
        self.config = config
        self.literature_data = literature_data
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
        self.max_tokens = model_settings.get('max_tokens', 3000)
        
        # Initialize the idea collection
        self.ideas = {
            "task_description": self.task_description,
            "research_question": self.research_question,
            "timestamp": datetime.now().isoformat(),
            "extracted_ideas": [],
            "novel_ideas": [],
            "idea_clusters": {},
            "methodology_combinations": []
        }
    
    def identify_ideas(self):
        """
        Identify and classify research ideas from the literature.
        
        Returns:
            dict: The identified ideas
        """
        logger.info("Starting idea identification and classification")
        
        # Step 1: Extract ideas from literature
        self._extract_ideas_from_literature()
        
        # Step 2: Generate novel ideas
        self._generate_novel_ideas()
        
        # Step 3: Cluster ideas by approach
        self._cluster_ideas()
        
        # Step 4: Generate methodology combinations
        self._generate_methodology_combinations()
        
        return self.ideas
    
    def _extract_ideas_from_literature(self):
        """Extract ideas directly from the literature summaries."""
        logger.info("Extracting ideas from literature")
        
        # Get papers and summaries
        papers = self.literature_data.get('papers', [])
        summaries = self.literature_data.get('summaries', [])
        research_gaps = self.research_plan.get('research_gaps', [])
        literature_overview = self.research_plan.get('literature_overview', {}).get('detailed_overview', '')
        
        if not papers or not summaries:
            logger.warning("Insufficient data for extracting ideas")
            return
        
        # Create a mapping of paper IDs to summaries
        summary_map = {s.get('paper_id'): s for s in summaries if 'paper_id' in s}
        
        # Extract research gaps text
        gaps_text = "\n".join([f"{gap.get('number')}. {gap.get('description')}" 
                              for gap in research_gaps])
        
        # Prepare paper information for the prompt
        paper_texts = []
        for paper in papers[:15]:  # Limit to 15 papers to avoid token limits
            paper_id = paper.get('id')
            paper_summary = summary_map.get(paper_id, {}).get('summary', '')
            
            info = f"Title: {paper.get('title')}\n"
            info += f"Authors: {', '.join(paper.get('authors', []))}\n"
            info += f"Abstract: {paper.get('abstract')}\n"
            if paper_summary:
                info += f"Summary: {paper_summary}\n"
            
            paper_texts.append(info)
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Literature Overview:
        {literature_overview[:500]}...
        
        Research Gaps Identified:
        {gaps_text}
        
        You are assisting a Master's student in the Idea Identification phase of a research project. Building upon the synthesized literature and research plan outlines, your task is to systematically categorize themes from the literature, identify potential research ideas, and pinpoint existing gaps and underexplored areas within the current body of knowledge.
        
        Based on the following academic papers and the identified research gaps, please:
        
        1. Identify key themes and categorize them, noting overlaps, contradictions, and areas of significant consensus or divergence
        2. Extract specific research ideas that emerge from these categorized themes
        3. Highlight underexplored areas that present opportunities for novel contributions
        
        Papers:
        {"".join([f"--- Paper {i+1} ---\n{text}\n\n" for i, text in enumerate(paper_texts)])}
        
        For each research idea you identify, please provide:
        1. A clear title for the idea
        2. A detailed description showing how it emerges from the categorized themes
        3. The methodological approach it involves
        4. Which research gap or underexplored area it addresses
        5. The estimated impact of pursuing this idea and how it would contribute novel insights to the field
        
        Identify at least 5 distinct research ideas that have the potential to contribute novel insights to the field. Format each as a separate numbered idea with the 5 components above.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping a Master's student identify and categorize research ideas from the academic literature, focusing on finding overlaps, contradictions, and underexplored areas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the ideas
            ideas_text = response.choices[0].message.content.strip()
            
            # Parse the ideas (simple approach)
            extracted_ideas = []
            current_idea = None
            
            for line in ideas_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Detect the start of a new idea
                if line.startswith(('Idea', 'IDEA', 'Research Idea', 'Idea ')):
                    # Save the previous idea
                    if current_idea and current_idea.get('title'):
                        extracted_ideas.append(current_idea)
                    
                    # Start a new idea
                    current_idea = {
                        "title": line.split(':', 1)[1].strip() if ':' in line else line,
                        "description": "",
                        "approach": "",
                        "gap_addressed": "",
                        "impact": "",
                        "source": "literature"
                    }
                elif current_idea:
                    # Parse idea components
                    if "description" in line.lower() or "details" in line.lower():
                        current_idea["description"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "approach" in line.lower() or "method" in line.lower():
                        current_idea["approach"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "gap" in line.lower() or "addresses" in line.lower():
                        current_idea["gap_addressed"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "impact" in line.lower() or "significance" in line.lower() or "contribution" in line.lower():
                        current_idea["impact"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["impact", "gap_addressed", "approach", "description"]:
                            if current_idea[component]:
                                current_idea[component] += " " + line
                                break
            
            # Add the last idea
            if current_idea and current_idea.get('title'):
                extracted_ideas.append(current_idea)
            
            self.ideas['extracted_ideas'] = extracted_ideas
            
            logger.info(f"Extracted {len(extracted_ideas)} ideas from literature")
            
        except Exception as e:
            logger.error(f"Error extracting ideas from literature: {str(e)}")
    
    def _generate_novel_ideas(self):
        """Generate novel research ideas using OpenAI."""
        logger.info("Generating novel research ideas")
        
        # Get existing ideas
        extracted_ideas = self.ideas.get('extracted_ideas', [])
        existing_titles = [idea.get('title', '') for idea in extracted_ideas]
        
        # Get research gaps and methodology
        research_gaps = self.research_plan.get('research_gaps', [])
        methodology = self.research_plan.get('methodology', {}).get('full_text', '')
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Research Gaps:
        {"".join([f"{gap.get('number')}. {gap.get('description')}\n" for gap in research_gaps])}
        
        Methodology Summary:
        {methodology[:500]}...
        
        Existing Research Ideas:
        {"".join([f"- {title}\n" for title in existing_titles])}
        
        Based on the research gaps and methodology, generate 5 novel research ideas that are NOT already covered by the existing ideas.
        
        These ideas should be:
        1. Innovative and novel
        2. Feasible given current technology
        3. Directly address one or more of the research gaps
        4. Have potential for significant impact
        
        For each idea, provide:
        1. A clear title
        2. A detailed description
        3. The methodological approach it would involve
        4. Which research gap it addresses
        5. The potential impact if successful
        
        Format each idea as a separate numbered idea with the 5 components above.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to generate novel research ideas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Slightly higher temperature for creativity
                max_tokens=self.max_tokens,
            )
            
            # Extract the ideas
            ideas_text = response.choices[0].message.content.strip()
            
            # Parse the ideas (simple approach)
            novel_ideas = []
            current_idea = None
            
            for line in ideas_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Detect the start of a new idea
                if line.startswith(('Idea', 'IDEA', 'Novel Idea', 'Idea ')):
                    # Save the previous idea
                    if current_idea and current_idea.get('title'):
                        novel_ideas.append(current_idea)
                    
                    # Start a new idea
                    current_idea = {
                        "title": line.split(':', 1)[1].strip() if ':' in line else line,
                        "description": "",
                        "approach": "",
                        "gap_addressed": "",
                        "impact": "",
                        "source": "generated"
                    }
                elif current_idea:
                    # Parse idea components
                    if "description" in line.lower() or "details" in line.lower():
                        current_idea["description"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "approach" in line.lower() or "method" in line.lower():
                        current_idea["approach"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "gap" in line.lower() or "addresses" in line.lower():
                        current_idea["gap_addressed"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "impact" in line.lower() or "significance" in line.lower():
                        current_idea["impact"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["impact", "gap_addressed", "approach", "description"]:
                            if current_idea[component]:
                                current_idea[component] += " " + line
                                break
            
            # Add the last idea
            if current_idea and current_idea.get('title'):
                novel_ideas.append(current_idea)
            
            self.ideas['novel_ideas'] = novel_ideas
            
            logger.info(f"Generated {len(novel_ideas)} novel ideas")
            
        except Exception as e:
            logger.error(f"Error generating novel ideas: {str(e)}")
    
    def _cluster_ideas(self):
        """Cluster the research ideas by methodological approach."""
        logger.info("Clustering research ideas")
        
        # Combine all ideas
        all_ideas = self.ideas.get('extracted_ideas', []) + self.ideas.get('novel_ideas', [])
        
        if not all_ideas:
            logger.warning("No ideas to cluster")
            return
        
        # Use OpenAI to cluster the ideas
        self._cluster_ideas_with_openai(all_ideas)
    
    def _cluster_ideas_with_openai(self, all_ideas):
        """
        Cluster ideas using OpenAI.
        
        Args:
            all_ideas (list): List of idea dictionaries
        """
        # Prepare idea information for the prompt
        idea_texts = []
        for i, idea in enumerate(all_ideas):
            text = f"Idea {i+1}: {idea.get('title')}\n"
            text += f"Description: {idea.get('description')}\n"
            text += f"Approach: {idea.get('approach')}\n"
            
            idea_texts.append(text)
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        I have the following research ideas:
        
        {"".join([f"--- {i+1} ---\n{text}\n\n" for i, text in enumerate(idea_texts)])}
        
        Please cluster these ideas based on their methodological approach. 
        
        For each cluster:
        1. Provide a descriptive name for the cluster
        2. List which ideas belong to this cluster (by their number)
        3. Explain the common methodological thread that unites these ideas
        
        Format your response with clear sections for each cluster.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to cluster research ideas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the clusters
            clusters_text = response.choices[0].message.content.strip()
            
            # Parse the clusters (simple approach)
            clusters = {}
            current_cluster = None
            cluster_description = ""
            
            for line in clusters_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Detect the start of a new cluster
                if "Cluster" in line or "GROUP" in line:
                    # Save the previous cluster
                    if current_cluster and cluster_description:
                        clusters[current_cluster] = {
                            "description": cluster_description,
                            "ideas": []
                        }
                    
                    # Start a new cluster
                    current_cluster = line
                    cluster_description = ""
                elif current_cluster:
                    # Check if this line contains idea numbers
                    if any(f"Idea {i+1}" in line for i in range(len(all_ideas))) or any(f"{i+1}," in line for i in range(len(all_ideas))):
                        # Extract idea indices
                        idea_indices = []
                        for i in range(len(all_ideas)):
                            if f"Idea {i+1}" in line or f"{i+1}," in line or f"{i+1}." in line:
                                idea_indices.append(i)
                        
                        if "ideas" not in clusters.get(current_cluster, {}):
                            if current_cluster not in clusters:
                                clusters[current_cluster] = {"description": "", "ideas": []}
                            
                            clusters[current_cluster]["ideas"] = idea_indices
                        else:
                            clusters[current_cluster]["ideas"].extend(idea_indices)
                    else:
                        # Add to the cluster description
                        cluster_description += " " + line
            
            # Add the last cluster
            if current_cluster and cluster_description:
                if current_cluster not in clusters:
                    clusters[current_cluster] = {"description": cluster_description, "ideas": []}
                else:
                    clusters[current_cluster]["description"] = cluster_description
            
            # Convert idea indices to actual ideas
            for cluster_name, cluster_data in clusters.items():
                idea_indices = cluster_data.get("ideas", [])
                cluster_ideas = []
                
                for idx in idea_indices:
                    if 0 <= idx < len(all_ideas):
                        cluster_ideas.append(all_ideas[idx])
                
                clusters[cluster_name]["ideas"] = cluster_ideas
            
            self.ideas['idea_clusters'] = clusters
            
            logger.info(f"Created {len(clusters)} idea clusters")
            
        except Exception as e:
            logger.error(f"Error clustering ideas: {str(e)}")
    
    def _generate_methodology_combinations(self):
        """Generate potential combinations of methodologies."""
        logger.info("Generating methodology combinations")
        
        # Get clusters and all ideas
        clusters = self.ideas.get('idea_clusters', {})
        all_ideas = self.ideas.get('extracted_ideas', []) + self.ideas.get('novel_ideas', [])
        
        if not clusters or not all_ideas:
            logger.warning("Insufficient data for generating methodology combinations")
            return
        
        # Create the prompt
        cluster_texts = []
        for name, data in clusters.items():
            text = f"{name}\n"
            text += f"Description: {data.get('description')}\n"
            text += "Ideas in this cluster:\n"
            
            for idea in data.get('ideas', []):
                text += f"- {idea.get('title')}\n"
            
            cluster_texts.append(text)
        
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        I have clustered research ideas into the following methodological groups:
        
        {"".join([f"--- Cluster {i+1} ---\n{text}\n\n" for i, text in enumerate(cluster_texts)])}
        
        Based on these methodological clusters, suggest 3-5 innovative combinations of approaches from different clusters that could yield novel insights.
        
        For each combination:
        1. Provide a title for the combined approach
        2. Describe how methods from different clusters would be integrated
        3. Explain the potential benefits of this integrated approach
        4. Identify any challenges or limitations
        5. Suggest a specific research question that this combination could address
        
        Format your response with clear sections for each combination.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to generate innovative methodology combinations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for creativity
                max_tokens=self.max_tokens,
            )
            
            # Extract the combinations
            combinations_text = response.choices[0].message.content.strip()
            
            # Parse the combinations (simple approach)
            combinations = []
            current_combination = None
            
            for line in combinations_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Detect the start of a new combination
                if line.startswith(('Combination', 'COMBINATION', 'Approach')):
                    # Save the previous combination
                    if current_combination and current_combination.get('title'):
                        combinations.append(current_combination)
                    
                    # Start a new combination
                    current_combination = {
                        "title": line.split(':', 1)[1].strip() if ':' in line else line,
                        "description": "",
                        "benefits": "",
                        "challenges": "",
                        "research_question": ""
                    }
                elif current_combination:
                    # Parse combination components
                    if "integration" in line.lower() or "description" in line.lower():
                        current_combination["description"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "benefit" in line.lower() or "advantage" in line.lower():
                        current_combination["benefits"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "challenge" in line.lower() or "limitation" in line.lower():
                        current_combination["challenges"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "question" in line.lower() or "research goal" in line.lower():
                        current_combination["research_question"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["research_question", "challenges", "benefits", "description"]:
                            if current_combination[component]:
                                current_combination[component] += " " + line
                                break
            
            # Add the last combination
            if current_combination and current_combination.get('title'):
                combinations.append(current_combination)
            
            self.ideas['methodology_combinations'] = combinations
            
            logger.info(f"Generated {len(combinations)} methodology combinations")
            
        except Exception as e:
            logger.error(f"Error generating methodology combinations: {str(e)}")

def identify_ideas(config, literature_data, research_plan):
    """
    Main function to identify and classify research ideas.
    
    Args:
        config (dict): Configuration settings
        literature_data (dict): Collected literature data
        research_plan (dict): Research plan from Phase 2
        
    Returns:
        dict: The identified ideas
    """
    # Create an idea identifier
    identifier = IdeaIdentifier(config, literature_data, research_plan)
    
    # Identify and classify ideas
    ideas = identifier.identify_ideas()
    
    return ideas

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
    
    # Load research plan
    plan_path = "research_plan.json"
    if len(sys.argv) > 3:
        plan_path = sys.argv[3]
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        research_plan = json.load(f)
    
    # Identify ideas
    ideas = identify_ideas(config, literature_data, research_plan)
    
    # Save output
    output_path = "identified_ideas.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ideas, f, indent=2, ensure_ascii=False)
    
    print(f"Identified ideas saved to {output_path}")