import os
import re
import logging
import json
from fpdf import FPDF
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    def __init__(self, templates_dir):
        self.templates_dir = templates_dir
        self.reports_dir = 'reports'
        os.makedirs(self.reports_dir, exist_ok=True)
        self.template_cache = {}
        
    def _get_video_number(self, video_name):
        """
        Extract video number from video name.
        
        Supports formats:
        - my_video8.mp4 -> 8
        - my_video_8.mp4 -> 8
        - video8.mp4 -> 8
        - 8.mp4 -> 8
        - report tamplate8.txt -> 8
        """
        # Clean the video name - remove path and extension
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        
        # Remove UUID prefix if present (format: uuid_filename.ext)
        if '_' in base_name and len(base_name.split('_')) > 1:
            base_name = '_'.join(base_name.split('_')[1:])
        
        # Try to find a number in the filename
        match = re.search(r'(\d+)', base_name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        
        logging.warning(f"Could not extract video number from: {video_name}")
        return None
    
    def _load_template(self, template_number):
        """
        Load template content from file with caching and multiple fallback options.
        
        Args:
            template_number: The number of the template to load
            
        Returns:
            str: The template content, or None if no matching template found
        """
        if template_number is None:
            logging.error("No template number provided")
            return None
            
        # Check cache first
        if template_number in self.template_cache:
            return self.template_cache[template_number]
        
        # First try exact match with video number
        template_path = os.path.join(self.templates_dir, f'report tamplate{template_number}.txt')
        
        # If not found, try other patterns
        if not os.path.exists(template_path):
            template_patterns = [
                f'report tamplate{template_number}.txt',  # Handle typo in filename
                f'report template{template_number}.txt',  # Correct spelling
                f'template{template_number}.txt',
                f'report{template_number}.txt',
                f'report_{template_number}.txt'
            ]
            
            for pattern in template_patterns:
                template_path = os.path.join(self.templates_dir, pattern)
                if os.path.exists(template_path):
                    break
        
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only cache non-empty templates
                        self.template_cache[template_number] = content
                        logging.info(f"Loaded template: {os.path.basename(template_path)}")
                        return content
            except Exception as e:
                logging.error(f"Error reading template {template_path}: {str(e)}")
                return None
                    
        logging.error(f"No valid template found for number: {template_number}")
        return None
    
    def generate_report(self, video_name, analysis_results=None):
        """
        Get the report template for the given video name.
        
        Args:
            video_name: Name of the video file (e.g., 'my_video1.mp4')
            
        Returns:
            dict: {
                'status': 'success'|'error',
                'message': str,
                'content': str,  # The template content
                'template_number': int  # The template number used
            }
        """
        try:
            if not video_name:
                return {
                    'status': 'error',
                    'message': 'No video name provided'
                }
                
            logging.info(f"Getting report template for video: {video_name}")
            
            # Extract video number from name (1-10)
            video_number = self._get_video_number(video_name)
            if video_number is None:
                return {
                    'status': 'error',
                    'message': f'Could not determine video number from name: {video_name}'
                }
            
            # Load the corresponding template
            template_content = self._load_template(video_number)
            if not template_content:
                return {
                    'status': 'error',
                    'message': f'Could not find template for video number: {video_number}'
                }
            
            # --- PDF Generation Logic ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size = 12)
            
            # Add the report content to the PDF
            try:
                # Ensure content is a string and clean it
                content_str = str(template_content).strip()
                
                # Split content into lines and add each line individually
                # Reverting from paragraph split back to line split for closer match to <pre>
                lines = content_str.splitlines()
                for line in lines:
                    # Add each line to the PDF
                    # Encode to latin-1 with replacement for safety, as fpdf handles this better
                    # This is a compromise; some non-latin-1 characters might still be replaced
                    # But it's better than stripping everything non-ASCII.
                    safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 10, safe_line, align='L') # Added left alignment
                
            except Exception as e:
                logging.error(f"Error processing content for PDF: {e}")
                pdf.multi_cell(0, 10, "Error loading report content for PDF.")

            # Define the output path for the PDF
            pdf_filename = f'report_{os.path.splitext(os.path.basename(video_name))[0]}.pdf'
            pdf_path = os.path.join(self.reports_dir, pdf_filename)
            
            # Ensure the reports directory exists before saving
            os.makedirs(self.reports_dir, exist_ok=True)

            # Save the PDF
            pdf.output(pdf_path)
            logging.info(f"Generated PDF report: {pdf_path}")
            # --- End PDF Generation Logic ---

            return {
                'status': 'success',
                'message': 'Template loaded and PDF generated successfully',
                'content': template_content, # Still return content for frontend display
                'template_number': video_number,
                'pdf_path': pdf_path # Include the path to the generated PDF
            }
                
        except Exception as e:
            error_msg = f'Error getting template or generating PDF: {str(e)}'
            logging.error(error_msg, exc_info=True)
            return {
                'status': 'error',
                'message': error_msg
            }

# Global instance
report_generator = ReportGenerator(
    os.path.abspath(os.path.join('E:\\Personal Projects\\New folder\\video_analyzer\\backend\\reports', 'report templates'))
)
