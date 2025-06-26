from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import os
import logging
from datetime import datetime
import json
from video_analyzer import perform_video_analysis
import uuid
from threading import Lock, Thread
from report_utils import report_generator
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app with static folder configuration
app = Flask(__name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global analysis status tracking
analysis_status = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'error': None,
    'results': None,
    'logs': []  # Store all log messages
}
status_lock = Lock()

def update_analysis_status(status, progress, message, level='info', error=None, results=None):
    """
    Update the global analysis status with thread safety.
    
    Args:
        status: Current status ('idle', 'processing', 'complete', 'error')
        progress: Progress percentage (0-100)
        message: Status message to display
        level: Log level ('info', 'warning', 'error', 'success')
        error: Error details if any
        results: Analysis results when complete
    """
    with status_lock:
        analysis_status['status'] = status
        analysis_status['progress'] = min(100, max(0, int(progress)))  # Ensure progress is 0-100
        analysis_status['message'] = message
        analysis_status['error'] = error
        
        # Append to logs if there's a message
        if message:
            analysis_status.setdefault('logs', [])
            # Keep only the last 100 log entries to prevent memory issues
            if len(analysis_status['logs']) >= 100:
                analysis_status['logs'].pop(0)
                
            analysis_status['logs'].append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': message,
                'type': level
            })
        # Only store results when analysis is complete
        if status == 'complete' or status == 'error':
             analysis_status['results'] = results

# Function to run analysis in a separate thread
def run_analysis_thread(video_path, config):
    """
    Run video analysis in a separate thread and update status.
    
    Args:
        video_path: Path to the video file to analyze
        config: Configuration dictionary for the analysis
    """
    try:
        # Get the original video filename for better status messages
        video_filename = os.path.basename(video_path)
        
        # Reset status before starting analysis in thread
        update_analysis_status(
            status='processing',
            progress=0,
            message=f'Starting analysis of {video_filename}...',
            level='info'
        )

        def thread_progress_callback(progress, message, level='info'):
            """
            Callback function to update analysis progress.
            
            Args:
                progress: Current progress percentage (0-100)
                message: Status message
                level: Log level ('info', 'warning', 'error', 'success')
            """
            update_analysis_status(
                status='processing',
                progress=progress,
                message=message,
                level=level
            )

        # Pass the thread-safe callback to perform_video_analysis
        try:
            analysis_result = perform_video_analysis(
                video_path=video_path,
                config=config,
                progress_callback=thread_progress_callback,
                log_callback=lambda msg, lvl='info': thread_progress_callback(
                    analysis_status.get('progress', 0), 
                    msg, 
                    lvl
                )
            )
        except Exception as e:
            error_msg = f'Analysis failed: {str(e)}'
            logging.exception(error_msg)
            update_analysis_status(
                status='error',
                progress=100,
                message=error_msg,
                level='error',
                error=error_msg
            )
            return

        # Update final status based on result
        if 'error' in analysis_result:
            error_msg = f"Analysis failed: {analysis_result['error']}"
            update_analysis_status(
                status='error',
                progress=100,
                message=error_msg,
                level='error',
                error=error_msg
            )
            return
        try:
            # Get the original video filename from the video path
            video_filename = os.path.basename(video_path)
            
            # Get the report template for this video
            logging.info(f"Getting report template for video: {video_filename}")
            report_result = report_generator.generate_report(video_filename, analysis_result)
            
            if report_result['status'] == 'success':
                # Add report info to the analysis results
                analysis_result['report'] = {
                    'content': report_result['content'],
                    'template_number': report_result['template_number'],
                    'generated_at': datetime.now().isoformat(),
                    'pdf_path': report_result.get('pdf_path')
                }
                
                success_msg = 'Analysis complete! Report template loaded successfully.'
                update_analysis_status(
                    status='complete',
                    progress=100,
                    message=success_msg,
                    level='success',
                    results=analysis_result
                )
                logging.info(success_msg)
            else:
                warning_msg = f'Analysis complete but could not load report template: {report_result.get("message", "Unknown error")}'
                update_analysis_status(
                    status='complete',
                    progress=100,
                    message=warning_msg,
                    level='warning',
                    results=analysis_result
                )
                logging.warning(warning_msg)
        except Exception as e:
            error_msg = f'Error during report generation: {str(e)}'
            logging.exception(error_msg)
            update_analysis_status(
                status='complete',
                progress=100,
                message='Analysis complete but report generation encountered an error',
                level='error',
                results=analysis_result
            )

    except Exception as e:
        error_msg = f'Analysis thread failed: {str(e)}'
        logging.error(error_msg)
        update_analysis_status('error', 100, error_msg, level='error', error=str(e))

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/x-icon')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400

        filename = data['filename']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(video_path):
            update_analysis_status('error', 0, 'Error: Video file not found', level='error')
            return jsonify({'error': 'Video file not found'}), 404

        # Load configuration
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except Exception as e:
            error_msg = f'Error loading config: {str(e)}'
            logging.error(error_msg)
            update_analysis_status('error', 100, error_msg, level='error')
            return jsonify({'error': error_msg}), 500

        # Start analysis in a new thread
        thread = Thread(target=run_analysis_thread, args=(video_path, config))
        thread.start()

        # Immediately return a success response to the frontend
        return jsonify({'success': True, 'message': 'Analysis started in background'}), 202 # Accepted

    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logging.error(error_msg)
        update_analysis_status('error', 100, error_msg, level='error', error=str(e))
        return jsonify({'error': error_msg}), 500

@app.route('/analysis_status')
def get_analysis_status():
    """
    Get the current analysis status.
    Returns:
        JSON with current analysis status, progress, logs, and results if available.
    """
    try:
        with status_lock:
            # Create a deep copy to avoid thread safety issues
            status = {
                'status': analysis_status.get('status', 'idle'),
                'progress': analysis_status.get('progress', 0),
                'message': analysis_status.get('message', ''),
                'error': analysis_status.get('error'),
                'logs': analysis_status.get('logs', [])[-20:],  # Return only recent logs
                'timestamp': datetime.now().isoformat()
            }
            
            # Add results if available
            if analysis_status.get('results'):
                status['results'] = analysis_status['results']
                
                # If there's a report, add the download URL
                if 'report' in analysis_status['results']:
                    report = analysis_status['results']['report']
                    if 'pdf_path' in report:
                        report = report.copy()  # Create a copy to avoid modifying the original
                        report['download_url'] = f'/download_report/{os.path.basename(report["pdf_path"])}'
                        status['results']['report'] = report
            
            return jsonify(status)
    except Exception as e:
        logging.error(f"Error getting analysis status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<path:filename>')
def download_report(filename):
    """Serve the generated PDF report for download"""
    try:
        # Security check: Ensure the filename is safe and points to a PDF
        safe_filename = os.path.basename(filename)
        if not safe_filename.lower().endswith('.pdf'):
            logging.warning(f"Attempted to download non-PDF file: {safe_filename}")
            return jsonify({'error': 'Only PDF files are allowed'}), 400
            
        # Construct the full path to the report
        reports_dir = os.path.abspath('reports')
        report_path = os.path.join(reports_dir, safe_filename)
        
        # Security check: Ensure the path is inside the reports directory
        if not os.path.abspath(report_path).startswith(reports_dir):
            logging.warning(f"Security violation: Attempted directory traversal with filename: {safe_filename}")
            return jsonify({'error': 'Invalid file path'}), 403
        
        # Check if the file exists
        if not os.path.exists(report_path):
            logging.error(f"Report not found: {report_path}")
            return jsonify({'error': 'Report not found'}), 404
            
        # Log the download attempt
        logging.info(f"Serving report for download: {report_path}")
        
        # Send the file with a friendly download name
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f'report_{safe_filename}',
            mimetype='application/pdf'
        )
    except Exception as e:
        error_msg = f"Error serving report {filename}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(debug=True)