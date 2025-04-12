# LitRevResearchAgent: Literature Review Research Agent

## Overview

LitRevRA (Literature Review Research Agent) is an AI-powered tool designed to automate and streamline the literature review process for researchers. By leveraging Large Language Models, it helps collect, analyze, and synthesize research literature to generate comprehensive, publication-quality reports that meet high academic standards.

## Features

- **Automated Literature Collection**: Searches and retrieves relevant papers from sources like arXiv, Google Scholar, and IEEE Xplore
- **Research Planning**: Analyzes literature to develop experimental plans and identify research gaps
- **Idea Generation**: Identifies potential research ideas based on literature analysis
- **Paper Relevancy Scoring**: Evaluates and ranks papers based on relevance to your research question
- **Data Visualization**: Creates visual representations of research trends and relationships
- **Scientific Report Generation**: Produces high-quality, well-structured scientific papers meeting academic publication standards

## System Architecture

LitRevRA follows a structured seven-phase workflow:

1. **Data Collection**: Gathers relevant papers from academic databases
2. **Analysis & Planning**: Synthesizes literature and generates research plans
3. **Idea Identification**: Categorizes and prioritizes potential research ideas
4. **Relevancy Scoring**: Develops a framework to evaluate paper relevance
5. **Analysis Execution**: Conducts deeper analysis and creates visualizations
6. **Results Interpretation**: Synthesizes findings and formulates recommendations
7. **Report Writing**: Generates a comprehensive scientific paper that meets academic standards

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Install Agent Development Kit (ADK) and LiteLLM::
   ```bash
   pip install google-adk
   pip install litellm
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a configuration file (`config.json`):
   ```json
   {
     "api_keys": {
       "openai": "your-openai-api-key"
     },
     "task_description": "Your research task description",
     "research_question": "Your specific research question",
     "model_settings": {
       "model": "gpt-4",
       "temperature": 0.7,
       "max_tokens": 4000
     },
     "search_settings": {
       "search_terms": ["term1", "term2"],
       "search_sources": ["arxiv", "ieee", "google_scholar"],
       "max_papers": 50
     },
     "output_preferences": {
       "report_format": "markdown",
       "include_visuals": true,
       "include_tables": true,
       "max_report_length": 8000
     }
   }
   ```

## Usage

### Basic Usage

Run the entire pipeline:

```bash
python litrevra-main.py --config your_config.json --output-dir ./output
```

### Advanced Usage

LitRevRA offers several command-line options for customizing execution:

```
--config       Path to configuration file (default: config.json)
--output-dir   Directory to store outputs (default: output)
--phase        Starting phase number (0-7) (default: 0)
--end-phase    Ending phase number (0-7) (default: 7)
--skip-phases  Comma-separated list of phases to skip
```

### Examples

Run only phases 3 through 5:
```bash
python litrevra-main.py --phase 3 --end-phase 5
```

Run the full pipeline but skip phases 2 and 4:
```bash
python litrevra-main.py --skip-phases 2,4
```

Run only the report writing phase:
```bash
python litrevra-main.py --phase 7 --end-phase 7
```

## Workflow Details

### Phase 0: Configuration Loading
Loads and validates the user configuration.

### Phase 1: Data Collection
Searches for and retrieves papers from specified sources using keywords from your research question.

Example prompt:
```
"PhD Literature Review Phase Prompt: Your goal is to perform a literature review for the presented task and add papers to the literature review. You have access to arXiv and can perform two search operations: (1) finding many different paper summaries from a search query and (2) getting a single full paper text for an arXiv paper (3) other resources mentioned in the input file that show you external resources."
```

### Phase 2: Analysis and Planning
Analyzes collected literature and generates a research plan.

Example prompt:
```
"Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic."
```

### Phase 3: Idea Identification
Identifies and categorizes potential research ideas based on the literature, looking for overlaps, contradictions, and areas of significant consensus or divergence.

### Phase 4: Relevancy Scoring
Develops a scoring framework to evaluate paper relevance to your research question and ranks literature based on multiple criteria.

### Phase 5: Analysis Execution
Conducts deeper analysis of the literature using topic modeling, sentiment analysis, and network analysis, creating informative visualizations that highlight key findings and trends.

### Phase 6: Results Interpretation
Synthesizes the analyzed data into a cohesive narrative that addresses the initial research questions, identifying significant patterns, trends, and relationships within the body of literature.

Example prompt:
```
"Your goal is to interpret results from the analysis phase. You should synthesize individual findings into a cohesive narrative that directly addresses the research question, highlighting areas of consensus and contradiction among researchers, and formulating clear recommendations for future research."
```

### Phase 7: Report Writing
Generates a high-quality scientific paper that meets stringent academic standards. The report writing phase is thoroughly enhanced to produce publication-ready content.

The scientific paper includes:
- Abstract: A concise, scholarly overview of the research
- Introduction: Sets context, states research questions, and outlines contributions
- Methodology: Detailed explanation of the LitRevRA system and its seven phases
- Literature Review: Synthesized overview of the field organized into thematic categories
- Findings: Presentation of key patterns, trends, and relationships discovered
- Discussion: Interpretation of findings and their implications
- Future Directions: Outlined avenues for future research
- Conclusion: Summary of key findings and their significance
- References: Properly formatted citations

The report generation emphasizes:
- **Novelty**: Clear articulation of how the LitRevRA approach differs from traditional literature reviews
- **Rigor**: Demonstration of thorough analysis and complete evaluation
- **Relevance**: Significance and potential impact on the field
- **Verifiability**: Transparent methodologies and processes
- **Presentation**: Logical flow, precise language, and effective communication

## Output Structure

The application creates a structured output directory:
```
output/
├── phase0/
│   └── config.json
├── phase1/
│   └── literature_data.json
├── phase2/
│   └── research_plan.json
├── phase3/
│   └── identified_ideas.json
├── phase4/
│   └── scoring_framework.json
├── phase5/
│   ├── analysis_results.json
│   └── visualizations/
│       ├── topic_distribution.png
│       ├── paper_idea_heatmap.png
│       └── ... (other visualization files)
├── phase6/
│   └── interpretation.json
└── phase7/
    ├── literature_review_report.md
    └── literature_review_report.pdf
```

## Benefits and Limitations

### Benefits
- Saves substantial time in literature review process
- Provides systematic coverage of relevant literature
- Identifies patterns and relationships across papers
- Generates publication-quality scientific papers
- Follows rigorous academic standards for scientific reporting
- Creates informative visualizations that enhance understanding

### Limitations
- Quality depends on available papers in selected databases
- Subject to API rate limits and token constraints
- May miss nuanced details a human researcher would catch
- Requires careful review and editing for publication submission

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool leverages OpenAI's GPT models for natural language processing
- Inspired by academic workflows for systematic literature reviews