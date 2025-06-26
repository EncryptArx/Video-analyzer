import os
import cv2
import numpy as np
from llama_engine import LlamaEngine
import logging
import json
from datetime import datetime
import time
import uuid
from ultralytics import YOLO
import torch
from transformers import pipeline
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
UPLOADS_DIR = 'uploads'
FRAMES_DIR = 'frames'
MODEL_PATH_RELATIVE = os.path.join('models', 'llama-7b.Q4_K_M.gguf')
MODEL_PATH_ABSOLUTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH_RELATIVE)

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

class VideoAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.load_models()
        self.llama_engine = None
        try:
            logging.info(f"Initializing LlamaEngine with model: {MODEL_PATH_ABSOLUTE}")
            llama_config = {'model_path': MODEL_PATH_ABSOLUTE}
            self.llama_engine = LlamaEngine(llama_config)
            logging.info("LlamaEngine initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize LlamaEngine: {e}")
            self.llama_engine = None
            
    def load_models(self):
        """Load all required models."""
        try:
            # Load YOLO models
            self.models['detection'] = YOLO('yolov8n.pt')
            self.models['pose'] = YOLO('yolov8n-pose.pt')
            
            # Load emotion detection model - using a proper image classification model
            self.models['emotion'] = pipeline(
                "image-classification",
                model="microsoft/resnet-50",  # Using ResNet-50 for image classification
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("All models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    def _to_json_serializable(self, obj):
        """Recursively converts object to a JSON serializable format."""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # Convert numpy arrays to lists
        try:
            if hasattr(obj, '_fields'):
                return {field.name: self._to_json_serializable(getattr(obj, field.name)) for field in obj._fields}
            elif hasattr(obj, '__iter__'):
                return [self._to_json_serializable(item) for item in obj]
        except Exception as e:
            logging.warning(f"Could not serialize object of type {type(obj)}: {e}")
            return str(obj)
        
        logging.warning(f"Encountered non-serializable object of type {type(obj)}: {obj}")
        return str(obj)

    def extract_frames(self, video_path: str, analysis_id: str) -> list:
        """Extract frames from video and save them to disk, returning list of frame paths."""
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return []
            
        logging.info(f"Starting frame extraction for {video_path}")
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return []

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Total frames in video: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame every 30 frames (assuming ~30fps, save 1 frame per second)
            if frame_count % 30 == 0:
                frame_filename = f"{analysis_id}_frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(FRAMES_DIR, frame_filename)
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)

            frame_count += 1
            
            # Log progress periodically
            if frame_count % 300 == 0:
                logging.info(f"Extracted frames: {frame_count}/{total_frames}")

        cap.release()
        logging.info(f"Finished frame extraction. Extracted {len(frames)} frames.")
        return frames
            
    def clean_up_frames(self, frame_paths: list):
        """Removes temporary frame files."""
        logging.info("Cleaning up temporary frames.")
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except OSError as e:
                logging.warning(f"Could not remove frame {frame_path}: {e}")
        logging.info("Frame cleanup complete.")

    def analyze_video(self, video_path: str, progress_callback=None, log_callback=None):
        """
        Analyze a video file.
        
        Args:
            video_path (str): Path to the video file
            progress_callback (callable): Function to call with progress updates
            log_callback (callable): Function to call with log messages
        
        Returns:
            dict: Analysis results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        def log(message, level='info'):
            if log_callback:
                log_callback(message, level)
            else:
                getattr(self.logger, level)(message)
        
        def update_progress(progress, message):
            if progress_callback:
                progress_callback(progress, message)
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            log(f"Starting video analysis: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
            
            # Initialize analysis results
            results = {
                'video_info': {
                    'filename': os.path.basename(video_path),
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration': duration
                },
                'detection_results': [],
                'pose_results': [],
                'emotion_results': []
            }
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                update_progress(progress, f"Processing frame {frame_count}/{total_frames}")
                
                # Run detections
                detection_results = self.models['detection'](frame, verbose=False)[0]
                pose_results = self.models['pose'](frame, verbose=False)[0]
                
                # Process detections
                frame_detections = []
                for det in detection_results.boxes.data:
                    x1, y1, x2, y2, conf, cls = det
                    if conf > 0.5:  # Confidence threshold
                        frame_detections.append({
                            'class': detection_results.names[int(cls)],
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                
                # Process pose detections
                frame_poses = []
                for pose in pose_results.keypoints.data:
                    if pose is not None:
                        frame_poses.append({
                            'keypoints': pose.cpu().numpy().tolist()
                        })
                
                # Process emotions for detected faces
                frame_emotions = []
                for det in frame_detections:
                    if det['class'] == 'person':
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            try:
                                emotion_results = self.models['emotion'](face)
                                frame_emotions.append({
                                    'bbox': det['bbox'],
                                    'emotions': emotion_results
                                })
                            except Exception as e:
                                log(f"Error processing emotion for face: {str(e)}", 'warning')
                
                # Store results
                results['detection_results'].append(frame_detections)
                results['pose_results'].append(frame_poses)
                results['emotion_results'].append(frame_emotions)
                
                frame_count += 1
            
            cap.release()
            
            update_progress(100, "Analysis complete!")
            log("Video analysis completed successfully")
            
            return results
            
        except Exception as e:
            log(f"Error during video analysis: {str(e)}", 'error')
            raise
    
    def _analyze_frame(self, frame_path: str) -> dict:
        """Analyze a single frame using the Llama model.
        
        Args:
            frame_path: Path to the frame image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load and preprocess the image
            image = cv2.imread(frame_path)
            if image is None:
                raise ValueError(f"Could not load image: {frame_path}")
            
            # Convert to RGB for better analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare the prompt for the model
            prompt = f"Analyze this image and describe what you see in detail. Focus on:\n"
            prompt += "1. Objects and people present\n"
            prompt += "2. Actions or activities\n"
            prompt += "3. Scene context and setting\n"
            prompt += "4. Notable details or unusual elements\n"
            
            # Get analysis from Llama model
            response = self.llama_engine.analyze_image(image_rgb, prompt)
            return {
                'frame_path': frame_path,
                'analysis': response,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing frame {frame_path}: {str(e)}")
            return {
                'frame_path': frame_path,
                'error': str(e),
                'timestamp': time.time()
            }

def perform_video_analysis(video_path: str, config: Dict[str, Any], progress_callback: callable = None, log_callback: callable = None) -> Dict:
    """
    Performs a full video analysis sequence.
    
    Args:
        video_path: Path to the video file
        config: Configuration dictionary
        progress_callback: Callback function for progress updates (progress: int, message: str, level: str)
        log_callback: Callback function for logging (message: str, level: str)
        
    Returns:
        Dict containing analysis results and metadata
    """
    # Get the original video filename for reporting
    video_filename = os.path.basename(video_path)
    logging.info(f"Starting full analysis for video: {video_filename}")
    
    def update_progress_and_log(progress, message, level='info'):
        """Helper function to update progress and log messages"""
        if progress_callback:
            progress_callback(progress, message, level)
        if log_callback:
            log_callback(message, level)
        logging.log(getattr(logging, level.upper(), logging.INFO), message)
    
    # Ensure initial state update is sent
    update_progress_and_log(0, "Starting video analysis")
        
    try:
        # Initialize LlamaEngine with proper error handling
        try:
            logging.info("Initializing LlamaEngine...")
            llama_engine = LlamaEngine(config)
            if not llama_engine:
                raise ValueError("Failed to initialize LlamaEngine")
            logging.info("LlamaEngine initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize LlamaEngine: {str(e)}"
            logging.error(error_msg)
            update_progress_and_log(100, error_msg, "error")
            return {"error": error_msg}

        # Create frames directory if it doesn't exist
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames from the video
        update_progress_and_log(5, f"Extracting frames from {video_filename}...")
        logging.info(f"Extracting frames from {video_filename}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video file: {video_path}"
            update_progress_and_log(100, error_msg, "error")
            logging.error(error_msg)
            return {"error": error_msg}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            error_msg = f"No frames found in video file: {video_path}"
            update_progress_and_log(100, error_msg, "warning")
            logging.warning(error_msg)
            return {"error": "No frames found in video file."}

        # Determine a reasonable max_frames if not provided in config
        max_frames = config.get("max_analysis_frames", 30)  # Reduced to 30 frames for better performance
        interval = max(1, total_frames // max_frames)

        frames = []
        frame_count = 0
        count = 0
        
        # Set video capture properties for faster reading
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Could not read frame {count} from video {video_path}")
                break
                
            if count % interval == 0:
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                try:
                    success = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        logging.warning(f"Could not save frame {frame_count} to {frame_filename}")
                        count += 1
                        continue
                except Exception as img_save_e:
                    logging.warning(f"Error saving frame {frame_count} to {frame_filename}: {img_save_e}")
                    count += 1
                    continue
                    
                frames.append(frame_filename)
                frame_count += 1
                
                # Report frame extraction progress
                extract_progress = 5 + int(frame_count / max_frames * 25)
                update_progress_and_log(extract_progress, f"Extracted frame {frame_count}/{max_frames}")

            count += 1

        cap.release()
        
        if not frames:
            error_msg = "No frames extracted for analysis."
            update_progress_and_log(100, error_msg, "warning")
            logging.warning(error_msg)
            return {"error": error_msg}

        # Run analysis sequence on frames
        update_progress_and_log(30, f"Analyzing {len(frames)} frames from {video_filename}...")
        logging.info(f"Starting analysis of {len(frames)} frames")
            
        analyses = []
        for i, frame_path in enumerate(frames):
            logging.info(f"Analyzing frame {i+1}/{len(frames)}: {os.path.basename(frame_path)}")
            
            try:
                # Load and preprocess the image
                image = cv2.imread(frame_path)
                if image is None:
                    raise ValueError(f"Could not load image: {frame_path}")
                
                # Convert to RGB for better analysis
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Prepare the prompt for the model
                prompt = f"Analyze this image and describe what you see in detail. Focus on:\n"
                prompt += "1. Objects and people present\n"
                prompt += "2. Actions or activities\n"
                prompt += "3. Scene context and setting\n"
                prompt += "4. Notable details or unusual elements\n"
                
                # Get analysis from Llama model
                try:
                    analysis_result = llama_engine.analyze_image(image_rgb, prompt)
                    if not analysis_result:
                        raise ValueError("No analysis result returned from LlamaEngine")
                        
                    analyses.append({
                        'frame_path': frame_path,
                        'analysis': analysis_result,
                        'timestamp': time.time()
                    })
                    logging.info(f"Successfully analyzed frame {i+1}/{len(frames)}")
                except Exception as analysis_e:
                    error_msg = f"Error during LlamaEngine analysis: {str(analysis_e)}"
                    logging.error(error_msg)
                    update_progress_and_log(30 + int((i + 1) / len(frames) * 60), 
                                         f"Analysis failed for frame {i+1}/{len(frames)}: {str(analysis_e)}", "error")
                    continue
                    
            except Exception as e:
                error_msg = f"Error processing frame {frame_path}: {str(e)}"
                logging.error(error_msg)
                update_progress_and_log(30 + int((i + 1) / len(frames) * 60), 
                                     f"Error processing frame {i+1}/{len(frames)}: {str(e)}", "error")
                continue
                
            # Report analysis progress
            analysis_progress = 30 + int((i + 1) / len(frames) * 60)
            update_progress_and_log(analysis_progress, f"Analyzed frame {i+1}/{len(frames)}")
                
        if not analyses:
            error_msg = "No analysis results obtained. All frame analyses failed."
            update_progress_and_log(100, error_msg, "error")
            logging.error(error_msg)
            return {"error": error_msg}

        # Clean up frames
        try:
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            logging.info("Cleaned up temporary frames")
            update_progress_and_log(95, "Cleaned up temporary files")
        except Exception as cleanup_e:
            error_msg = f"Error during cleanup: {cleanup_e}"
            logging.error(error_msg)
            update_progress_and_log(95, error_msg, "warning")

        # Add video metadata to results
        result = {
            "video_filename": video_filename,
            "analysis_date": datetime.now().isoformat(),
            "total_frames_processed": len(analyses),
            "analyses": analyses
        }
        
        update_progress_and_log(100, f"Analysis of {video_filename} complete.", "success")
        return result
        
    except Exception as e:
        error_msg = f"Full video analysis failed: {str(e)}"
        logging.error(error_msg)
        update_progress_and_log(100, error_msg, "error")
        return {"error": error_msg}

def main():
    # This main function is for command-line testing and can be kept separate
    # from the Flask application's usage of VideoAnalyzer.
    print("Video Analyzer Command Line Interface (for testing)")
    video_file = input("Enter path to video file: ")
    
    analyzer = VideoAnalyzer()
    if not analyzer.llama_engine:
        print("Error: LlamaEngine could not be initialized. Analysis aborted.")
        return

    if not os.path.exists(video_file):
        print(f"Error: File not found at {video_file}")
        return

    print("Starting analysis...")
    result = analyzer.analyze_video(video_file)
    
    print("\nAnalysis Complete.")
    print("Status:", result.get('status'))
    if result.get('error'):
        print("Error:", result.get('error'))
    
    if result.get('results_data'):
        # Print the full serializable results data
        print("\nAnalysis Data:\n", json.dumps(result['results_data'], indent=2))

if __name__ == '__main__':
    main() 