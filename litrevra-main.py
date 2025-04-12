#!/usr/bin/env python3
# LitRevRA-app: Literature Review Research Agent Application
# Main entry point for the application

import os
import sys
import json
import argparse
import logging
from pathlib import Path

from phases.phase0_config import load_configuration
from phases.phase1_data_collection import collect_literature
from phases.phase2_analysis_planning import analyze_and_plan
from phases.phase3_idea_identification import identify_ideas
from phases.phase4_relevancy_scoring import develop_scoring_framework
from phases.phase5_analysis_execution import execute_analysis
from phases.phase6_results_interpretation import interpret_results
from phases.phase7_report_writing import write_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("litrevra.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LitRevRA")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Literature Review Research Agent Application')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to store outputs')
    parser.add_argument('--phase', type=int, default=0, help='Starting phase (0-7)')
    parser.add_argument('--end-phase', type=int, default=7, help='Ending phase (0-7)')
    parser.add_argument('--skip-phases', type=str, help='Comma-separated list of phases to skip')
    
    return parser.parse_args()

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(8):  # Create directories for each phase
        os.makedirs(os.path.join(output_dir, f"phase{i}"), exist_ok=True)
    
    logger.info(f"Created output directory structure in {output_dir}")

def main():
    """Main function to run the LitRevRA application."""
    args = parse_arguments()
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Determine which phases to skip
    skip_phases = []
    if args.skip_phases:
        skip_phases = [int(phase) for phase in args.skip_phases.split(',')]
    
    # Phase 0: Load configuration
    if args.phase <= 0 <= args.end_phase and 0 not in skip_phases:
        logger.info("Starting Phase 0: Configuration Loading")
        config = load_configuration(args.config)
        save_result(config, args.output_dir, 0, "config.json")
    else:
        # Load saved config if starting from a later phase
        config = load_saved_result(args.output_dir, 0, "config.json")
    
    # Phase 1: Data Collection
    if args.phase <= 1 <= args.end_phase and 1 not in skip_phases:
        logger.info("Starting Phase 1: Data Collection")
        literature_data = collect_literature(config)
        save_result(literature_data, args.output_dir, 1, "literature_data.json")
    else:
        literature_data = load_saved_result(args.output_dir, 1, "literature_data.json")
    
    return 0
    # Phase 2: Analysis and Planning
    if args.phase <= 2 <= args.end_phase and 2 not in skip_phases:
        logger.info("Starting Phase 2: Analysis and Planning")
        research_plan = analyze_and_plan(config, literature_data)
        save_result(research_plan, args.output_dir, 2, "research_plan.json")
    else:
        research_plan = load_saved_result(args.output_dir, 2, "research_plan.json")
    
    # Phase 3: Idea Identification
    if args.phase <= 3 <= args.end_phase and 3 not in skip_phases:
        logger.info("Starting Phase 3: Idea Identification and Classification")
        ideas = identify_ideas(config, literature_data, research_plan)
        save_result(ideas, args.output_dir, 3, "identified_ideas.json")
    else:
        ideas = load_saved_result(args.output_dir, 3, "identified_ideas.json")
    
    # Phase 4: Relevancy Scoring Framework
    if args.phase <= 4 <= args.end_phase and 4 not in skip_phases:
        logger.info("Starting Phase 4: Relevancy Scoring Framework Development")
        scoring_framework = develop_scoring_framework(config, ideas, literature_data)
        save_result(scoring_framework, args.output_dir, 4, "scoring_framework.json")
    else:
        scoring_framework = load_saved_result(args.output_dir, 4, "scoring_framework.json")
    
    # Phase 5: Analysis Execution
    if args.phase <= 5 <= args.end_phase and 5 not in skip_phases:
        logger.info("Starting Phase 5: Analysis Execution and Visualization")
        analysis_results = execute_analysis(config, literature_data, research_plan, ideas, scoring_framework)
        save_result(analysis_results, args.output_dir, 5, "analysis_results.json")
    else:
        analysis_results = load_saved_result(args.output_dir, 5, "analysis_results.json")
    
    # Phase 6: Results Interpretation
    if args.phase <= 6 <= args.end_phase and 6 not in skip_phases:
        logger.info("Starting Phase 6: Results Interpretation and Recommendation")
        interpretation = interpret_results(config, analysis_results, research_plan)
        save_result(interpretation, args.output_dir, 6, "interpretation.json")
    else:
        interpretation = load_saved_result(args.output_dir, 6, "interpretation.json")
    
    # # Phase 7: Report Writing
    # if args.phase <= 7 <= args.end_phase and 7 not in skip_phases:
    #     logger.info("Starting Phase 7: Report Writing")
    #     report = write_report(
    #         config, 
    #         literature_data, 
    #         research_plan, 
    #         ideas, 
    #         scoring_framework, 
    #         analysis_results, 
    #         interpretation
    #     )
    #     save_result(report, args.output_dir, 7, "final_report.md")
        
    #     # Save the final report as a PDF if possible
    #     try:
    #         from reportlab.pdfgen import canvas
    #         generate_pdf_report(report, os.path.join(args.output_dir, "phase7", "final_report.pdf"))
    #         logger.info("Generated PDF report")
    #     except ImportError:
    #         logger.warning("Could not generate PDF report (reportlab not installed)")
    
    # logger.info("LitRevRA application completed successfully")
    # return 0
    if args.phase <= 7 <= args.end_phase and 7 not in skip_phases:
        logger.info("Starting Phase 7: Report Writing")
        report = write_report(
            config, 
            literature_data, 
            research_plan, 
            ideas, 
            scoring_framework, 
            analysis_results, 
            interpretation
        )
        save_result(report, args.output_dir, 7, "final_report.md")
        
        # Generate PDF report
        pdf_path = os.path.join(args.output_dir, "phase7", "final_report.pdf")
        if generate_pdf_report(report, pdf_path):
            logger.info(f"Generated PDF report at {pdf_path}")

def save_result(data, output_dir, phase, filename):
    """Save phase result to a file."""
    output_path = os.path.join(output_dir, f"phase{phase}", filename)
    
    if isinstance(data, str):  # For text content like markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data)
    else:  # For JSON-serializable data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved phase {phase} result to {output_path}")

def load_saved_result(output_dir, phase, filename):
    """Load previously saved phase result."""
    filepath = os.path.join(output_dir, f"phase{phase}", filename)
    
    if not os.path.exists(filepath):
        logger.error(f"Cannot find saved result for phase {phase} at {filepath}")
        return None
    
    if filename.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:  # For text content
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

# In main.py, replace the generate_pdf_report function with this:

def generate_pdf_report(report_md, output_path):
    """Generate a PDF version of the report."""
    # This is a placeholder. In a real implementation, you would use a
    # library like reportlab or a markdown-to-pdf converter.
    # For now, we'll just create a minimal PDF with the title
    
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(output_path)
        c.drawString(100, 750, "Literature Review Research Report")
        c.drawString(100, 730, "Generated by LitRevRA")
        c.save()
        return True
    except ImportError:
        logger.warning("Could not generate PDF report (reportlab not installed)")
        return False

if __name__ == "__main__":
    sys.exit(main())
