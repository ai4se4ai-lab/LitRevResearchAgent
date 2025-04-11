#!/usr/bin/env python3
# Phase 5: Analysis Execution and Visualization

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger("LitRevRA.Phase5")

class AnalysisExecutor:
    """Class to execute analysis and create visualizations."""
    
    def __init__(self, config, literature_data, research_plan, ideas, scoring_framework, output_dir):
        """
        Initialize the AnalysisExecutor.
        
        Args:
            config (dict): Configuration settings
            literature_data (dict): Collected literature data
            research_plan (dict): Research plan from Phase 2
            ideas (dict): Identified research ideas
            scoring_framework (dict): Scoring framework from Phase 4
            output_dir (str): Directory to save visualizations
        """
        self.config = config
        self.literature_data = literature_data
        self.research_plan = research_plan
        self.ideas = ideas
        self.scoring_framework = scoring_framework
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        # Initialize the analysis results
        self.analysis_results = {
            "task_description": self.task_description,
            "research_question": self.research_question,
            "timestamp": datetime.now().isoformat(),
            "visualizations": [],
            "trends": [],
            "key_findings": [],
            "research_directions": []
        }
    
    def execute_analysis(self):
        """
        Execute analysis and create visualizations.
        
        Returns:
            dict: The analysis results
        """
        logger.info("Starting analysis execution and visualization")
        
        # Step 1: Create basic visualizations
        self._create_basic_visualizations()
        
        # Step 2: Create advanced visualizations
        self._create_advanced_visualizations()
        
        # Step 3: Identify research trends
        self._identify_research_trends()
        
        # Step 4: Extract key findings
        self._extract_key_findings()
        
        # Step 5: Generate research directions
        self._generate_research_directions()
        
        return self.analysis_results
    
    def _create_basic_visualizations(self):
        """Create basic visualizations of the data."""
        logger.info("Creating basic visualizations")
        
        # Get data
        papers = self.literature_data.get('papers', [])
        paper_scores = self.scoring_framework.get('paper_scores', [])
        idea_scores = self.scoring_framework.get('idea_scores', [])
        
        if not papers:
            logger.warning("No papers for basic visualizations")
            return
        
        visualizations = []
        
        # 1. Publication year distribution
        try:
            years = []
            for paper in papers:
                if 'published' in paper and paper['published']:
                    # Extract year from ISO format or just use the string
                    try:
                        if '-' in paper['published']:
                            year = int(paper['published'].split('-')[0])
                        else:
                            year = int(paper['published'])
                        years.append(year)
                    except (ValueError, TypeError):
                        pass
            
            if years:
                plt.figure(figsize=(10, 6))
                plt.hist(years, bins=range(min(years), max(years) + 2), alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Distribution of Publication Years')
                plt.xlabel('Year')
                plt.ylabel('Number of Papers')
                plt.grid(axis='y', alpha=0.75)
                plt.tight_layout()
                
                # Save the figure
                year_dist_path = os.path.join(self.output_dir, 'publication_year_distribution.png')
                plt.savefig(year_dist_path)
                plt.close()
                
                visualizations.append({
                    "title": "Publication Year Distribution",
                    "description": "Distribution of papers by publication year",
                    "type": "histogram",
                    "path": year_dist_path
                })
                
                logger.info(f"Created publication year distribution visualization: {year_dist_path}")
        except Exception as e:
            logger.error(f"Error creating publication year distribution: {str(e)}")
        
        # 2. Top authors
        try:
            all_authors = []
            for paper in papers:
                if 'authors' in paper:
                    all_authors.extend(paper['authors'])
            
            if all_authors:
                author_counts = pd.Series(all_authors).value_counts().head(15)
                
                plt.figure(figsize=(12, 8))
                author_counts.plot(kind='barh', color='lightgreen')
                plt.title('Top Authors by Number of Papers')
                plt.xlabel('Number of Papers')
                plt.ylabel('Author')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save the figure
                top_authors_path = os.path.join(self.output_dir, 'top_authors.png')
                plt.savefig(top_authors_path)
                plt.close()
                
                visualizations.append({
                    "title": "Top Authors",
                    "description": "Top authors by number of papers in the literature collection",
                    "type": "bar_chart",
                    "path": top_authors_path
                })
                
                logger.info(f"Created top authors visualization: {top_authors_path}")
        except Exception as e:
            logger.error(f"Error creating top authors visualization: {str(e)}")
        
        # 3. Paper source distribution
        try:
            sources = []
            for paper in papers:
                if 'source' in paper:
                    sources.append(paper['source'])
            
            if sources:
                source_counts = pd.Series(sources).value_counts()
                
                plt.figure(figsize=(10, 6))
                source_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, shadow=True, explode=[0.05] * len(source_counts))
                plt.title('Distribution of Paper Sources')
                plt.axis('equal')
                plt.tight_layout()
                
                # Save the figure
                sources_path = os.path.join(self.output_dir, 'paper_sources.png')
                plt.savefig(sources_path)
                plt.close()
                
                visualizations.append({
                    "title": "Paper Sources",
                    "description": "Distribution of papers by source",
                    "type": "pie_chart",
                    "path": sources_path
                })
                
                logger.info(f"Created paper sources visualization: {sources_path}")
        except Exception as e:
            logger.error(f"Error creating paper sources visualization: {str(e)}")
        
        # 4. Top papers by score
        if paper_scores:
            try:
                # Sort papers by overall score
                top_papers = sorted(paper_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:10]
                
                titles = [p.get('title', '')[:40] + '...' if len(p.get('title', '')) > 40 else p.get('title', '') for p in top_papers]
                scores = [p.get('overall_score', 0) for p in top_papers]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(titles)), scores, color='coral')
                plt.yticks(range(len(titles)), titles)
                plt.title('Top 10 Papers by Overall Score')
                plt.xlabel('Score')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save the figure
                top_papers_path = os.path.join(self.output_dir, 'top_papers_by_score.png')
                plt.savefig(top_papers_path)
                plt.close()
                
                visualizations.append({
                    "title": "Top Papers by Score",
                    "description": "Top 10 papers ranked by overall relevance score",
                    "type": "bar_chart",
                    "path": top_papers_path
                })
                
                logger.info(f"Created top papers visualization: {top_papers_path}")
            except Exception as e:
                logger.error(f"Error creating top papers visualization: {str(e)}")
        
        # 5. Top ideas by score
        if idea_scores:
            try:
                # Sort ideas by overall score
                top_ideas = sorted(idea_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:10]
                
                titles = [i.get('idea_title', '')[:40] + '...' if len(i.get('idea_title', '')) > 40 else i.get('idea_title', '') for i in top_ideas]
                scores = [i.get('overall_score', 0) for i in top_ideas]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(titles)), scores, color='lightblue')
                plt.yticks(range(len(titles)), titles)
                plt.title('Top 10 Research Ideas by Overall Score')
                plt.xlabel('Score')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save the figure
                top_ideas_path = os.path.join(self.output_dir, 'top_ideas_by_score.png')
                plt.savefig(top_ideas_path)
                plt.close()
                
                visualizations.append({
                    "title": "Top Research Ideas by Score",
                    "description": "Top 10 research ideas ranked by overall potential score",
                    "type": "bar_chart",
                    "path": top_ideas_path
                })
                
                logger.info(f"Created top ideas visualization: {top_ideas_path}")
            except Exception as e:
                logger.error(f"Error creating top ideas visualization: {str(e)}")
        
        # 6. Word cloud from paper titles and abstracts
        try:
            # Combine titles and abstracts
            text = ' '.join([f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in papers])
            
            if text:
                # Create word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                     max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate(text)
                
                plt.figure(figsize=(16, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                
                # Save the figure
                wordcloud_path = os.path.join(self.output_dir, 'word_cloud.png')
                plt.savefig(wordcloud_path)
                plt.close()
                
                visualizations.append({
                    "title": "Literature Word Cloud",
                    "description": "Word cloud generated from paper titles and abstracts",
                    "type": "word_cloud",
                    "path": wordcloud_path
                })
                
                logger.info(f"Created word cloud visualization: {wordcloud_path}")
        except Exception as e:
            logger.error(f"Error creating word cloud visualization: {str(e)}")
        
        # Add visualizations to the results
        self.analysis_results['visualizations'].extend(visualizations)
    
    def _create_advanced_visualizations(self):
        """Create advanced visualizations of the data."""
        logger.info("Creating advanced visualizations")
        
        # Get data
        papers = self.literature_data.get('papers', [])
        extracted_ideas = self.ideas.get('extracted_ideas', [])
        novel_ideas = self.ideas.get('novel_ideas', [])
        all_ideas = extracted_ideas + novel_ideas
        paper_scores = self.scoring_framework.get('paper_scores', [])
        relevancy_matrix = self.scoring_framework.get('relevancy_matrix', {})
        
        if not papers or not all_ideas:
            logger.warning("Insufficient data for advanced visualizations")
            return
        
        visualizations = []
        
        # 1. Paper-Idea Heatmap
        if relevancy_matrix and 'matrix' in relevancy_matrix:
            try:
                matrix = np.array(relevancy_matrix['matrix'])
                
                if matrix.size > 0:
                    # Limit to top 15 papers and ideas for readability
                    n_papers = min(15, matrix.shape[0])
                    n_ideas = min(15, matrix.shape[1])
                    
                    # Get paper and idea titles
                    paper_titles = [p.get('title', '')[:30] + '...' if len(p.get('title', '')) > 30 else p.get('title', '') 
                                   for p in relevancy_matrix.get('papers', [])[:n_papers]]
                    idea_titles = [i.get('title', '')[:30] + '...' if len(i.get('title', '')) > 30 else i.get('title', '') 
                                  for i in relevancy_matrix.get('ideas', [])[:n_ideas]]
                    
                    # Create heatmap
                    plt.figure(figsize=(14, 10))
                    sns.heatmap(matrix[:n_papers, :n_ideas], annot=True, cmap='YlGnBu', fmt='.2f',
                               xticklabels=idea_titles, yticklabels=paper_titles)
                    plt.title('Relevancy Between Papers and Ideas')
                    plt.xlabel('Research Ideas')
                    plt.ylabel('Papers')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    
                    # Save the figure
                    heatmap_path = os.path.join(self.output_dir, 'paper_idea_heatmap.png')
                    plt.savefig(heatmap_path)
                    plt.close()
                    
                    visualizations.append({
                        "title": "Paper-Idea Relevancy Heatmap",
                        "description": "Heatmap showing relevancy scores between papers and research ideas",
                        "type": "heatmap",
                        "path": heatmap_path
                    })
                    
                    logger.info(f"Created paper-idea heatmap visualization: {heatmap_path}")
            except Exception as e:
                logger.error(f"Error creating paper-idea heatmap: {str(e)}")
        
        # 2. Topic modeling visualization (simple PCA on TF-IDF)
        try:
            # Combine titles and abstracts
            paper_texts = [f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in papers]
            
            if paper_texts:
                # Create TF-IDF vectors
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(paper_texts)
                
                # Apply PCA for dimensionality reduction
                pca = PCA(n_components=2)
                paper_points = pca.fit_transform(tfidf_matrix.toarray())
                
                # Create scatter plot
                plt.figure(figsize=(12, 8))
                plt.scatter(paper_points[:, 0], paper_points[:, 1], alpha=0.7)
                
                # Add paper titles as annotations (limited to avoid clutter)
                for i in range(min(15, len(papers))):
                    title = papers[i].get('title', '')[:20] + '...' if len(papers[i].get('title', '')) > 20 else papers[i].get('title', '')
                    plt.annotate(title, (paper_points[i, 0], paper_points[i, 1]), fontsize=8)
                
                plt.title('Topic Distribution of Papers (PCA on TF-IDF)')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save the figure
                topic_path = os.path.join(self.output_dir, 'topic_distribution.png')
                plt.savefig(topic_path)
                plt.close()
                
                visualizations.append({
                    "title": "Topic Distribution",
                    "description": "2D visualization of paper distribution based on topic similarity",
                    "type": "scatter_plot",
                    "path": topic_path
                })
                
                logger.info(f"Created topic distribution visualization: {topic_path}")
        except Exception as e:
            logger.error(f"Error creating topic distribution visualization: {str(e)}")
        
        # 3. Criteria score comparison for top papers
        if paper_scores and len(paper_scores) > 0 and 'criteria_scores' in paper_scores[0]:
            try:
                # Get top 5 papers
                top_papers = sorted(paper_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:5]
                
                # Get criteria names
                criteria = list(top_papers[0]['criteria_scores'].keys())
                
                # Create dataframe for plotting
                data = []
                for paper in top_papers:
                    paper_data = {
                        'Paper': paper.get('title', '')[:20] + '...' if len(paper.get('title', '')) > 20 else paper.get('title', '')
                    }
                    
                    # Add criteria scores
                    for criterion in criteria:
                        paper_data[criterion] = paper.get('criteria_scores', {}).get(criterion, {}).get('score', 0)
                    
                    data.append(paper_data)
                
                df = pd.DataFrame(data)
                
                # Melt the dataframe for seaborn
                df_melted = pd.melt(df, id_vars=['Paper'], var_name='Criterion', value_name='Score')
                
                # Create grouped bar chart
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Paper', y='Score', hue='Criterion', data=df_melted)
                plt.title('Criteria Scores for Top Papers')
                plt.xlabel('')
                plt.ylabel('Score')
                plt.legend(title='Criterion')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save the figure
                criteria_path = os.path.join(self.output_dir, 'paper_criteria_scores.png')
                plt.savefig(criteria_path)
                plt.close()
                
                visualizations.append({
                    "title": "Paper Criteria Scores",
                    "description": "Comparison of criteria scores for top papers",
                    "type": "grouped_bar_chart",
                    "path": criteria_path
                })
                
                logger.info(f"Created paper criteria scores visualization: {criteria_path}")
            except Exception as e:
                logger.error(f"Error creating paper criteria scores visualization: {str(e)}")
        
        # 4. Interactive visualization with Plotly (paper/idea network)
        try:
            # Create a network of papers and ideas
            G = nx.Graph()
            
            # Add papers as nodes
            for i, paper in enumerate(papers[:20]):  # Limit to 20 papers for clarity
                G.add_node(f"P{i}", type="paper", title=paper.get('title', ''))
            
            # Add ideas as nodes
            for i, idea in enumerate(all_ideas[:10]):  # Limit to 10 ideas for clarity
                G.add_node(f"I{i}", type="idea", title=idea.get('title', ''))
            
            # Add edges based on relevancy
            if relevancy_matrix and 'matrix' in relevancy_matrix:
                matrix = np.array(relevancy_matrix['matrix'])
                
                if matrix.size > 0:
                    # Limit to same number of papers/ideas as above
                    n_papers = min(20, matrix.shape[0])
                    n_ideas = min(10, matrix.shape[1])
                    
                    for p in range(n_papers):
                        for i in range(n_ideas):
                            # Only add edges with relevancy above threshold
                            if matrix[p, i] > 0.3:  # Threshold for visibility
                                G.add_edge(f"P{p}", f"I{i}", weight=matrix[p, i])
            
            # Create positions using spring layout
            pos = nx.spring_layout(G)
            
            # Extract node positions
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(G.nodes[node]['title'])
                node_color.append('blue' if G.nodes[node]['type'] == 'paper' else 'red')
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_width = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_width.append(edge[2].get('weight', 0.5) * 2)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=15,
                    color=node_color,
                    line=dict(width=2, color='#FFF')
                ),
                text=node_text,
                hoverinfo='text'
            ))
            
            # Set layout
            fig.update_layout(
                title='Paper-Idea Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Save the figure
            network_path = os.path.join(self.output_dir, 'paper_idea_network.html')
            pio.write_html(fig, file=network_path)
            
            visualizations.append({
                "title": "Paper-Idea Network",
                "description": "Interactive network visualization of papers and ideas",
                "type": "network",
                "path": network_path
            })
            
            logger.info(f"Created paper-idea network visualization: {network_path}")
        except Exception as e:
            logger.error(f"Error creating paper-idea network visualization: {str(e)}")
        
        # Add visualizations to the results
        self.analysis_results['visualizations'].extend(visualizations)
    
    def _identify_research_trends(self):
        """Identify research trends from the literature."""
        logger.info("Identifying research trends")
        
        # Get data
        papers = self.literature_data.get('papers', [])
        paper_scores = self.scoring_framework.get('paper_scores', [])
        research_gaps = self.research_plan.get('research_gaps', [])
        literature_overview = self.research_plan.get('literature_overview', {}).get('detailed_overview', '')
        
        if not papers:
            logger.warning("No papers for identifying research trends")
            return
        
        # Get top papers based on relevancy score
        top_papers = sorted(paper_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:15] if paper_scores else papers[:15]
        
        # Prepare paper summaries for the prompt
        paper_summaries = []
        for paper in top_papers:
            summary = f"Title: {paper.get('title')}\n"
            summary += f"Authors: {', '.join(paper.get('authors', []))}\n"
            summary += f"Abstract: {paper.get('abstract')}\n"
            
            paper_summaries.append(summary)
        
        # Extract research gaps text
        gaps_text = ""
        if research_gaps:
            gaps_text = "\n".join([f"{gap.get('number')}. {gap.get('description')}" 
                                  for gap in research_gaps])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Literature Overview:
        {literature_overview[:500]}...
        
        You are assisting PhD #2 with a granular examination of highly relevant papers identified through a rigorous relevancy scoring framework. Your task is to identify and analyze research trends by leveraging advanced analytical techniques including topic modeling, sentiment analysis, and network analysis.
        
        Based on the following summaries of top papers and identified research gaps, please:
        
        1. Identify 5-7 major research trends in this field
        2. For each trend, analyze its thematic structure, prevailing perspectives, and connections to other concepts
        3. Map the relationships between key concepts and authors within each trend
        4. Identify potential gaps or underexplored areas within each trend
        
        Papers:
        {"".join([f"--- Paper {i+1} ---\n{summary}\n\n" for i, summary in enumerate(paper_summaries)])}
        
        Research Gaps:
        {gaps_text}
        
        For each identified trend, provide:
        
        1. A descriptive title for the trend
        2. A detailed analysis of its thematic structure and key concepts
        3. Prevailing perspectives and arguments within this trend (sentiment analysis)
        4. Key papers and authors contributing to this trend
        5. Network of relationships between concepts, methods, and authors
        6. Trajectory of the trend (emerging, established, declining)
        7. Potential gaps or areas for further exploration within this trend
        
        Format your response with clear sections for each trend, emphasizing insights that could inform subsequent interpretation and report writing stages.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping a PhD candidate with granular analysis of research literature using topic modeling, sentiment analysis, and network analysis techniques."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the trends
            trends_text = response.choices[0].message.content.strip()
            
            # Parse the trends (simple approach)
            trends = []
            current_trend = None
            
            for line in trends_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for trend headers
                if "Trend" in line or "TREND" in line:
                    # Save the previous trend
                    if current_trend and current_trend.get('title'):
                        trends.append(current_trend)
                    
                    # Extract trend title
                    title = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new trend
                    current_trend = {
                        "title": title,
                        "thematic_structure": "",
                        "perspectives": "",
                        "key_papers": [],
                        "network": "",
                        "trajectory": "",
                        "gaps": ""
                    }
                elif current_trend:
                    # Parse trend components
                    if any(kw in line.lower() for kw in ["thematic", "structure", "key concepts"]):
                        current_trend["thematic_structure"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif any(kw in line.lower() for kw in ["perspective", "sentiment", "argument"]):
                        current_trend["perspectives"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif any(kw in line.lower() for kw in ["key papers", "authors", "contribut"]):
                        papers_text = line.split(':', 1)[1].strip() if ':' in line else line
                        current_trend["key_papers"] = [p.strip() for p in papers_text.split(',')]
                    elif any(kw in line.lower() for kw in ["network", "relationship", "connection"]):
                        current_trend["network"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif any(kw in line.lower() for kw in ["trajectory", "status", "emerging", "established"]):
                        current_trend["trajectory"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif any(kw in line.lower() for kw in ["gap", "future", "unexplored", "further"]):
                        current_trend["gaps"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["gaps", "trajectory", "network", "key_papers", "perspectives", "thematic_structure"]:
                            if current_trend[component]:
                                if isinstance(current_trend[component], list):
                                    current_trend[component].append(line)
                                else:
                                    current_trend[component] += " " + line
                                break
            
            # Add the last trend
            if current_trend and current_trend.get('title'):
                trends.append(current_trend)
            
            self.analysis_results['trends'] = trends
            
            logger.info(f"Identified {len(trends)} research trends")
            
        except Exception as e:
            logger.error(f"Error identifying research trends: {str(e)}")
    
    def _extract_key_findings(self):
        """Extract key findings from the literature and analysis."""
        logger.info("Extracting key findings")
        
        # Get data
        papers = self.literature_data.get('papers', [])
        paper_scores = self.scoring_framework.get('paper_scores', [])
        idea_scores = self.scoring_framework.get('idea_scores', [])
        trends = self.analysis_results.get('trends', [])
        visualizations = self.analysis_results.get('visualizations', [])
        
        if not papers:
            logger.warning("No papers for extracting key findings")
            return
        
        # Create a prompt for key findings extraction
        # Prepare summaries of top papers
        top_papers = sorted(paper_scores, key=lambda x: x.get('overall_score', 0), reverse=True)[:10] if paper_scores else papers[:10]
        
        paper_summaries = []
        for paper in top_papers:
            summary = f"Title: {paper.get('title')}\n"
            summary += f"Authors: {', '.join(paper.get('authors', []))}\n"
            summary += f"Abstract: {paper.get('abstract')}\n"
            
            paper_summaries.append(summary)
        
        # Extract trends summary
        trends_summary = ""
        if trends:
            trends_summary = "\n".join([f"{i+1}. {trend.get('title')}: {trend.get('description')}" 
                                      for i, trend in enumerate(trends)])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Based on the following information, extract 7-10 key findings from the literature and our analysis:
        
        Top Papers:
        {"".join([f"--- Paper {i+1} ---\n{summary}\n\n" for i, summary in enumerate(paper_summaries)])}
        
        Research Trends:
        {trends_summary}
        
        For each key finding, provide:
        1. A clear statement of the finding
        2. Supporting evidence from the literature
        3. The significance of this finding to our research question
        4. Any implications for future research
        
        Format your response with clear sections for each key finding.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to extract key findings from a literature review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract the findings
            findings_text = response.choices[0].message.content.strip()
            
            # Parse the findings (simple approach)
            findings = []
            current_finding = None
            
            for line in findings_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for finding headers
                if "Finding" in line or "KEY FINDING" in line:
                    # Save the previous finding
                    if current_finding and current_finding.get('statement'):
                        findings.append(current_finding)
                    
                    # Extract finding statement
                    statement = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new finding
                    current_finding = {
                        "statement": statement,
                        "evidence": "",
                        "significance": "",
                        "implications": ""
                    }
                elif current_finding:
                    # Parse finding components
                    if "evidence" in line.lower() or "support" in line.lower():
                        current_finding["evidence"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "significance" in line.lower() or "importance" in line.lower():
                        current_finding["significance"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "implication" in line.lower() or "future" in line.lower():
                        current_finding["implications"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["implications", "significance", "evidence", "statement"]:
                            if current_finding[component]:
                                current_finding[component] += " " + line
                                break
            
            # Add the last finding
            if current_finding and current_finding.get('statement'):
                findings.append(current_finding)
            
            self.analysis_results['key_findings'] = findings
            
            logger.info(f"Extracted {len(findings)} key findings")
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {str(e)}")
    
    def _generate_research_directions(self):
        """Generate potential research directions based on the analysis."""
        logger.info("Generating research directions")
        
        # Get data
        key_findings = self.analysis_results.get('key_findings', [])
        trends = self.analysis_results.get('trends', [])
        research_gaps = self.research_plan.get('research_gaps', [])
        
        if not key_findings and not trends and not research_gaps:
            logger.warning("Insufficient data for generating research directions")
            return
        
        # Extract findings summary
        findings_summary = ""
        if key_findings:
            findings_summary = "\n".join([f"{i+1}. {finding.get('statement')}" 
                                      for i, finding in enumerate(key_findings)])
        
        # Extract trends summary
        trends_summary = ""
        if trends:
            trends_summary = "\n".join([f"{i+1}. {trend.get('title')}: {trend.get('description')}" 
                                      for i, trend in enumerate(trends)])
        
        # Extract research gaps text
        gaps_text = ""
        if research_gaps:
            gaps_text = "\n".join([f"{gap.get('number')}. {gap.get('description')}" 
                                  for gap in research_gaps])
        
        # Create the prompt
        prompt = f"""
        Task: {self.task_description}
        Research Question: {self.research_question}
        
        Based on the following information from our literature review and analysis:
        
        Key Findings:
        {findings_summary}
        
        Research Trends:
        {trends_summary}
        
        Research Gaps:
        {gaps_text}
        
        Generate 5-7 promising research directions that could be pursued. For each direction:
        
        1. Provide a clear title for the research direction
        2. Describe the research direction in detail
        3. Explain how it addresses identified gaps and builds on existing findings
        4. Outline potential methodologies or approaches
        5. Discuss expected outcomes and impact
        
        Format your response with clear sections for each research direction.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to generate promising research directions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Slightly higher temperature for creativity
                max_tokens=self.max_tokens,
            )
            
            # Extract the directions
            directions_text = response.choices[0].message.content.strip()
            
            # Parse the directions (simple approach)
            directions = []
            current_direction = None
            
            for line in directions_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                # Check for direction headers
                if "Direction" in line or "RESEARCH DIRECTION" in line:
                    # Save the previous direction
                    if current_direction and current_direction.get('title'):
                        directions.append(current_direction)
                    
                    # Extract direction title
                    title = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Start a new direction
                    current_direction = {
                        "title": title,
                        "description": "",
                        "addresses_gaps": "",
                        "methodologies": "",
                        "expected_outcomes": ""
                    }
                elif current_direction:
                    # Parse direction components
                    if "description" in line.lower() or "detail" in line.lower():
                        current_direction["description"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "gap" in line.lower() or "address" in line.lower() or "build" in line.lower():
                        current_direction["addresses_gaps"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "methodolog" in line.lower() or "approach" in line.lower():
                        current_direction["methodologies"] = line.split(':', 1)[1].strip() if ':' in line else line
                    elif "outcome" in line.lower() or "impact" in line.lower() or "expect" in line.lower():
                        current_direction["expected_outcomes"] = line.split(':', 1)[1].strip() if ':' in line else line
                    else:
                        # Add to the most recently assigned component
                        for component in ["expected_outcomes", "methodologies", "addresses_gaps", "description"]:
                            if current_direction[component]:
                                current_direction[component] += " " + line
                                break
            
            # Add the last direction
            if current_direction and current_direction.get('title'):
                directions.append(current_direction)
            
            self.analysis_results['research_directions'] = directions
            
            logger.info(f"Generated {len(directions)} research directions")
            
        except Exception as e:
            logger.error(f"Error generating research directions: {str(e)}")

def execute_analysis(config, literature_data, research_plan, ideas, scoring_framework):
    """
    Main function to execute analysis and create visualizations.
    
    Args:
        config (dict): Configuration settings
        literature_data (dict): Collected literature data
        research_plan (dict): Research plan from Phase 2
        ideas (dict): Identified research ideas
        scoring_framework (dict): Scoring framework from Phase 4
        
    Returns:
        dict: The analysis results
    """
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'output', 'phase5', 'visualizations')
    
    # Create an analysis executor
    executor = AnalysisExecutor(config, literature_data, research_plan, ideas, scoring_framework, output_dir)
    
    # Execute analysis
    analysis_results = executor.execute_analysis()
    
    return analysis_results

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
    
    # Load ideas
    ideas_path = "identified_ideas.json"
    if len(sys.argv) > 4:
        ideas_path = sys.argv[4]
    
    with open(ideas_path, 'r', encoding='utf-8') as f:
        ideas = json.load(f)
    
    # Load scoring framework
    framework_path = "scoring_framework.json"
    if len(sys.argv) > 5:
        framework_path = sys.argv[5]
    
    with open(framework_path, 'r', encoding='utf-8') as f:
        scoring_framework = json.load(f)
    
    # Execute analysis
    analysis_results = execute_analysis(config, literature_data, research_plan, ideas, scoring_framework)
    
    # Save output
    output_path = "analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis results saved to {output_path}")