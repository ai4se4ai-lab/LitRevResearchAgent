#!/usr/bin/env python3
# Phase 1: Data Collection

import logging
import time
import random
import json
import os
from datetime import datetime
from urllib.parse import quote_plus

import openai
import requests
from bs4 import BeautifulSoup
import arxiv
from scholarly import scholarly

logger = logging.getLogger("LitRevRA.Phase1")

class LiteratureCollector:
    """Class to collect literature data from various sources."""
    
    def __init__(self, config):
        """
        Initialize the LiteratureCollector.
        
        Args:
            config (dict): Configuration settings
        """
        self.config = config
        
        # Set up OpenAI API
        openai.api_key = config['api_keys']['openai']
        
        # Initialize data storage
        self.literature_data = {
            "papers": [],
            "summaries": [],
            "search_metadata": {
                "search_terms": config.get('search_settings', {}).get('search_terms', []),
                "sources": config.get('search_settings', {}).get('search_sources', []),
                "timestamp": datetime.now().isoformat(),
            },
            "external_resources": config.get('external_resources', [])
        }
        
        # Load the task description
        self.task_description = config.get('task_description', '')
        self.research_question = config.get('research_question', '')
        
        # Set model parameters
        model_settings = config.get('model_settings', {})
        self.model = model_settings.get('model', 'gpt-4')
        self.temperature = model_settings.get('temperature', 0.7)
        self.max_tokens = model_settings.get('max_tokens', 2000)
        
        # Search settings
        search_settings = config.get('search_settings', {})
        self.max_papers = search_settings.get('max_papers', 30)
        self.sources = search_settings.get('search_sources', ['arxiv', 'google_scholar'])
        self.publication_years = search_settings.get('publication_years', [2018, 2023])
        
        # Initialize search terms if not provided
        if not self.literature_data['search_metadata']['search_terms']:
            self._generate_search_terms()
            
    def _generate_search_terms(self):
        """Generate search terms based on the task description using OpenAI."""
        logger.info("Generating search terms using OpenAI")
        
        prompt = f"""
        Task: {self.task_description}
        
        Based on the above task, generate 5-8 specific search terms or phrases that would be effective 
        for finding relevant academic papers. These search terms should:
        
        1. Be specific enough to yield relevant results
        2. Cover different aspects of the research question
        3. Use technical terminology appropriate for academic search
        
        Format your response as a list of search terms, one per line.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to generate effective academic search terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            # Extract the search terms from the response
            content = response.choices[0].message.content.strip()
            search_terms = [term.strip() for term in content.split('\n') if term.strip()]
            
            # Update the search terms
            self.literature_data['search_metadata']['search_terms'] = search_terms
            logger.info(f"Generated {len(search_terms)} search terms")
            
        except Exception as e:
            logger.error(f"Error generating search terms: {str(e)}")
            # Fallback to basic terms based on the task
            basic_terms = [self.task_description]
            if self.research_question:
                basic_terms.append(self.research_question)
            self.literature_data['search_metadata']['search_terms'] = basic_terms
    
    def collect_literature(self):
        """
        Collect literature from all configured sources.
        
        Returns:
            dict: The collected literature data
        """
        logger.info("Starting literature collection")
        
        # Collection strategy
        papers_collected = 0
        
        # Distribute paper collection across sources
        papers_per_source = max(5, self.max_papers // len(self.sources))
        
        # Collect papers from each source
        for source in self.sources:
            if papers_collected >= self.max_papers:
                break
                
            logger.info(f"Collecting papers from {source}")
            
            # Use the appropriate collection method based on the source
            if source == 'arxiv':
                self._collect_from_arxiv(papers_per_source)
            elif source == 'google_scholar':
                self._collect_from_google_scholar(papers_per_source)
            elif source == 'ieee':
                self._collect_from_ieee(papers_per_source)
            else:
                logger.warning(f"Unknown source: {source}")
                
            papers_collected = len(self.literature_data['papers'])
            logger.info(f"Total papers collected so far: {papers_collected}")
            
        # Get more detailed information for each paper using OpenAI
        self._generate_paper_summaries()
            
        return self.literature_data
    
    def _collect_from_arxiv(self, max_papers):
        """
        Collect papers from ArXiv.
        
        Args:
            max_papers (int): Maximum number of papers to collect
        """
        try:
            for term in self.literature_data['search_metadata']['search_terms']:
                # Skip if we have enough papers
                if len(self.literature_data['papers']) >= self.max_papers:
                    break
                
                logger.info(f"Searching arXiv for: {term}")
                
                # Prepare search query
                search = arxiv.Search(
                    query=term,
                    max_results=max_papers,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                for result in search.results():
                    # Extract paper information
                    paper = {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "url": result.entry_id,
                        "pdf_url": result.pdf_url,
                        "published": result.published.isoformat(),
                        "source": "arxiv",
                        "id": result.entry_id.split('/')[-1],
                        "categories": result.categories
                    }
                    
                    # Add paper to the collection
                    self.literature_data['papers'].append(paper)
                    
                    # Stop if we have enough papers
                    if len(self.literature_data['papers']) >= self.max_papers:
                        break
                        
                # Avoid rate limiting
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error collecting papers from arXiv: {str(e)}")
    
    def _collect_from_google_scholar(self, max_papers):
        """
        Collect papers from Google Scholar.
        
        Args:
            max_papers (int): Maximum number of papers to collect
        """
        try:
            for term in self.literature_data['search_metadata']['search_terms']:
                # Skip if we have enough papers
                if len(self.literature_data['papers']) >= self.max_papers:
                    break
                    
                logger.info(f"Searching Google Scholar for: {term}")
                
                # Create a scholarly search query
                search_query = scholarly.search_pubs(term)
                
                # Get the specified number of results
                count = 0
                for result in search_query:
                    if count >= max_papers:
                        break
                        
                    # Check if the year is within the specified range
                    year = result.get('bib', {}).get('pub_year')
                    if year:
                        year = int(year)
                        if (self.publication_years[0] <= year <= self.publication_years[1]):
                            # Extract paper information
                            paper = {
                                "title": result.get('bib', {}).get('title', ''),
                                "authors": result.get('bib', {}).get('author', []),
                                "abstract": result.get('bib', {}).get('abstract', ''),
                                "url": result.get('pub_url', ''),
                                "published": str(year),
                                "source": "google_scholar",
                                "id": result.get('author_id', '') + '_' + str(count),
                                "citations": result.get('num_citations', 0)
                            }
                            
                            # Add paper to the collection
                            self.literature_data['papers'].append(paper)
                            count += 1
                    
                    # Avoid rate limiting
                    time.sleep(2 + random.random())
                    
        except Exception as e:
            logger.error(f"Error collecting papers from Google Scholar: {str(e)}")
    
    def _collect_from_ieee(self, max_papers):
        """
        Collect papers from IEEE Xplore.
        
        Args:
            max_papers (int): Maximum number of papers to collect
        """
        try:
            # Check if IEEE API key is available
            ieee_api_key = self.config['api_keys'].get('ieee')
            
            for term in self.literature_data['search_metadata']['search_terms']:
                # Skip if we have enough papers
                if len(self.literature_data['papers']) >= self.max_papers:
                    break
                
                logger.info(f"Searching IEEE Xplore for: {term}")
                
                # If API key is available, use the IEEE API
                if ieee_api_key:
                    self._collect_from_ieee_api(term, max_papers, ieee_api_key)
                else:
                    # Otherwise, attempt web scraping (note: this may be against terms of service)
                    logger.warning("IEEE API key not available, attempting alternative method")
                    self._collect_from_ieee_alternative(term, max_papers)
                    
                # Avoid rate limiting
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error collecting papers from IEEE: {str(e)}")
    
    def _collect_from_ieee_api(self, term, max_papers, api_key):
        """
        Collect papers from IEEE Xplore using the API.
        
        Args:
            term (str): Search term
            max_papers (int): Maximum number of papers to collect
            api_key (str): IEEE API key
        """
        base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        
        params = {
            "apikey": api_key,
            "format": "json",
            "max_records": max_papers,
            "start_record": 1,
            "sort_order": "desc",
            "sort_field": "relevance",
            "querytext": term,
            "start_year": self.publication_years[0],
            "end_year": self.publication_years[1]
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            for article in data.get('articles', []):
                paper = {
                    "title": article.get('title', ''),
                    "authors": [a.get('full_name', '') for a in article.get('authors', {}).get('authors', [])],
                    "abstract": article.get('abstract', ''),
                    "url": f"https://ieeexplore.ieee.org/document/{article.get('article_number', '')}",
                    "published": article.get('publication_year', ''),
                    "source": "ieee",
                    "id": article.get('article_number', ''),
                    "doi": article.get('doi', '')
                }
                
                # Add paper to the collection
                self.literature_data['papers'].append(paper)
                
                # Stop if we have enough papers
                if len(self.literature_data['papers']) >= self.max_papers:
                    break
        else:
            logger.error(f"IEEE API request failed with status code {response.status_code}")
    
    def _collect_from_ieee_alternative(self, term, max_papers):
        """
        Alternative method to collect papers from IEEE Xplore.
        This is a fallback when the API key is not available.
        
        Args:
            term (str): Search term
            max_papers (int): Maximum number of papers to collect
        """
        # This is a simplified implementation and might not work reliably
        # due to potential changes in the IEEE Xplore website structure
        
        encoded_term = quote_plus(term)
        url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={encoded_term}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract paper information
                paper_elements = soup.select('.List-results-items')
                
                for element in paper_elements[:max_papers]:
                    title_elem = element.select_one('.title')
                    authors_elem = element.select_one('.author')
                    abstract_elem = element.select_one('.abstract')
                    
                    if title_elem:
                        title = title_elem.text.strip()
                        
                        # Extract other information
                        authors = []
                        if authors_elem:
                            authors_text = authors_elem.text.strip()
                            authors = [a.strip() for a in authors_text.split(';')]
                        
                        abstract = ""
                        if abstract_elem:
                            abstract = abstract_elem.text.strip()
                        
                        # Extract URL
                        paper_url = ""
                        if title_elem.find('a'):
                            paper_url = "https://ieeexplore.ieee.org" + title_elem.find('a').get('href', '')
                        
                        paper = {
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "url": paper_url,
                            "published": "",  # Publication year not easily available
                            "source": "ieee",
                            "id": paper_url.split('/')[-1] if paper_url else f"ieee_{len(self.literature_data['papers'])}"
                        }
                        
                        # Add paper to the collection
                        self.literature_data['papers'].append(paper)
                        
                        # Stop if we have enough papers
                        if len(self.literature_data['papers']) >= self.max_papers:
                            break
            else:
                logger.warning(f"Failed to retrieve IEEE search results: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error in IEEE alternative collection: {str(e)}")
            
    def _generate_paper_summaries(self):
        """Generate detailed summaries for each collected paper using OpenAI."""
        logger.info("Generating paper summaries using OpenAI")
        
        # Skip if no papers were collected
        if not self.literature_data['papers']:
            logger.warning("No papers to summarize")
            return
        
        # Group papers by batches for efficiency
        batch_size = 5
        paper_batches = [self.literature_data['papers'][i:i + batch_size] 
                        for i in range(0, len(self.literature_data['papers']), batch_size)]
        
        for batch_idx, batch in enumerate(paper_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(paper_batches)}")
            
            # Create a prompt for the batch
            prompt = f"""
            Research task: {self.task_description}
            
            For each of the following academic papers, provide a concise summary that includes:
            1. Main contributions and findings
            2. Methodology used
            3. Key strengths and limitations
            4. Relevance to our research question: "{self.research_question}"
            
            Rate each paper's relevance to our research task on a scale of 1-10.
            
            Papers to analyze:
            """
            
            for i, paper in enumerate(batch):
                prompt += f"\n--- Paper {i+1} ---\n"
                prompt += f"Title: {paper['title']}\n"
                prompt += f"Authors: {', '.join(paper['authors'])}\n"
                prompt += f"Abstract: {paper['abstract']}\n"
            
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant helping to analyze academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Extract the summaries
                summaries_text = response.choices[0].message.content.strip()
                
                # Process and store the summaries
                self._process_batch_summaries(summaries_text, batch)
                
                # Avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error generating summaries for batch {batch_idx + 1}: {str(e)}")
    
    def _process_batch_summaries(self, summaries_text, papers_batch):
        """
        Process and store the batch summaries.
        
        Args:
            summaries_text (str): The generated summaries text
            papers_batch (list): The batch of papers
        """
        # Split the text by paper markers
        try:
            paper_sections = summaries_text.split("--- Paper ")
            
            # Remove any empty sections
            paper_sections = [s for s in paper_sections if s.strip()]
            
            for i, section in enumerate(paper_sections):
                # Skip if there are more sections than papers
                if i >= len(papers_batch):
                    break
                    
                # Extract paper ID
                paper_id = papers_batch[i]['id']
                
                # Extract the relevance score
                relevance_score = 5  # Default score
                relevance_line = None
                
                for line in section.split('\n'):
                    if 'relevance' in line.lower() and any(str(score) in line for score in range(1, 11)):
                        relevance_line = line
                        # Extract the score (1-10)
                        for score in range(1, 11):
                            if str(score) in line:
                                relevance_score = score
                                break
                        break
                
                summary = {
                    "paper_id": paper_id,
                    "title": papers_batch[i]['title'],
                    "summary": section.strip(),
                    "relevance_score": relevance_score,
                    "relevance_explanation": relevance_line if relevance_line else "No explicit relevance explanation provided."
                }
                
                self.literature_data['summaries'].append(summary)
                
        except Exception as e:
            logger.error(f"Error processing batch summaries: {str(e)}")

def collect_literature(config):
    """
    Main function to collect literature data.
    
    Args:
        config (dict): Configuration settings
        
    Returns:
        dict: The collected literature data
    """
    # Create a literature collector
    collector = LiteratureCollector(config)
    
    # Collect literature from various sources
    literature_data = collector.collect_literature()
    
    # Generate a reference list
    generate_reference_list(literature_data)
    
    return literature_data

def generate_reference_list(literature_data):
    """
    Generate a formatted reference list from the collected papers.
    
    Args:
        literature_data (dict): The collected literature data
    """
    references = []
    
    for paper in literature_data['papers']:
        # Format authors
        if paper['authors']:
            if len(paper['authors']) > 3:
                authors = f"{paper['authors'][0]} et al."
            else:
                authors = ', '.join(paper['authors'])
        else:
            authors = "Unknown"
        
        # Extract year
        year = "Unknown"
        if 'published' in paper and paper['published']:
            # Try to extract year from ISO format or just use the string
            try:
                if '-' in paper['published']:
                    year = paper['published'].split('-')[0]
                else:
                    year = paper['published']
            except:
                year = paper['published']
        
        # Create the reference string
        reference = f"{authors} ({year}). {paper['title']}."
        
        # Add source-specific information
        if paper['source'] == 'arxiv':
            reference += f" arXiv:{paper['id']}."
        elif 'doi' in paper and paper['doi']:
            reference += f" DOI: {paper['doi']}."
        
        if 'url' in paper and paper['url']:
            reference += f" Retrieved from {paper['url']}"
        
        references.append(reference)
    
    # Add the references to the literature data
    literature_data['reference_list'] = references

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
    
    # Collect literature
    literature_data = collect_literature(config)
    
    # Print statistics
    print(f"Collected {len(literature_data['papers'])} papers")
    print(f"Generated {len(literature_data['summaries'])} summaries")
    
    # Save output
    output_path = "literature_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(literature_data, f, indent=2, ensure_ascii=False)
    
    print(f"Literature data saved to {output_path}")
