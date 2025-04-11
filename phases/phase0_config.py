#!/usr/bin/env python3
# Phase 0: Configuration Loading

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("LitRevRA.Phase0")

def load_configuration(config_path):
    """
    Load configuration from a JSON file and environment variables.
    
    Args:
        config_path (str): Path to the config JSON file
        
    Returns:
        dict: Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise
    
    # Add API keys from environment variables if not in config
    if 'api_keys' not in config:
        config['api_keys'] = {}
        
    # OpenAI API key
    if 'openai' not in config['api_keys'] or not config['api_keys']['openai']:
        config['api_keys']['openai'] = os.getenv('OPENAI_API_KEY')
        if not config['api_keys']['openai']:
            logger.warning("OpenAI API key not found in config or environment variables")
    
    # Optional: Other API keys like Google Scholar, IEEE Xplore, etc.
    for api in ['google_scholar', 'ieee', 'arxiv', 'semantic_scholar']:
        env_var = f"{api.upper()}_API_KEY"
        if api not in config['api_keys'] or not config['api_keys'][api]:
            config['api_keys'][api] = os.getenv(env_var)
    
    # Validate required configuration fields
    validate_configuration(config)
    
    return config

def validate_configuration(config):
    """
    Validate that the configuration has all the required fields.
    
    Args:
        config (dict): Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing
    """
    # Check for required fields
    required_fields = ['task_description', 'api_keys']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        error_msg = f"Missing required configuration fields: {', '.join(missing_fields)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if task description is provided
    if not config.get('task_description'):
        error_msg = "Task description is empty in the configuration"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if at least OpenAI API key is provided
    if not config['api_keys'].get('openai'):
        error_msg = "OpenAI API key is required but not provided"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate optional settings
    if 'model_settings' in config:
        if 'model' not in config['model_settings']:
            config['model_settings']['model'] = 'gpt-4'
            logger.info("Using default model: gpt-4")
        
        if 'temperature' not in config['model_settings']:
            config['model_settings']['temperature'] = 0.7
            logger.info("Using default temperature: 0.7")
    else:
        # Set default model settings
        config['model_settings'] = {
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 2000
        }
        logger.info("Using default model settings")
    
    # Validate search settings
    if 'search_settings' not in config:
        config['search_settings'] = {
            'max_papers': 30,
            'search_sources': ['arxiv', 'google_scholar', 'ieee'],
            'publication_years': [2018, 2023],  # Last 5 years by default
            'relevance_threshold': 0.7
        }
        logger.info("Using default search settings")
    
    # Add the timestamp
    from datetime import datetime
    config['timestamp'] = datetime.now().isoformat()
    
    logger.info("Configuration validated successfully")
    return config

def create_example_config(output_path="example_config.json"):
    """
    Create an example configuration file.
    
    Args:
        output_path (str): Path to save the example config
    """
    example_config = {
        "task_description": "Conduct a literature review on the use of transformer models for time series forecasting",
        "research_question": "How effective are transformer models compared to traditional methods for time series forecasting?",
        "api_keys": {
            "openai": "your-openai-api-key-here",
            "google_scholar": "",
            "ieee": "",
            "arxiv": "",
            "semantic_scholar": ""
        },
        "model_settings": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "search_settings": {
            "max_papers": 30,
            "search_sources": ["arxiv", "google_scholar", "ieee"],
            "search_terms": [
                "transformer time series forecasting",
                "attention mechanisms time series prediction",
                "deep learning time series analysis"
            ],
            "publication_years": [2018, 2023],
            "relevance_threshold": 0.7
        },
        "external_resources": [
            {"name": "IEEE Xplore", "url": "https://ieeexplore.ieee.org/"},
            {"name": "Google Scholar", "url": "https://scholar.google.com/"},
            {"name": "arXiv", "url": "https://arxiv.org/"}
        ],
        "output_preferences": {
            "report_format": "markdown",
            "include_visuals": true,
            "include_tables": true,
            "max_report_length": 5000
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Example configuration created at {output_path}")

if __name__ == "__main__":
    # If this script is run directly, create an example configuration
    logging.basicConfig(level=logging.INFO)
    create_example_config()
