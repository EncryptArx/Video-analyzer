import traceback
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import torch
from ultralytics import YOLO
import mediapipe as mp
from deepface import DeepFace
import supervision as sv
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils.ops import non_max_suppression
import re
from types import SimpleNamespace
from llama_cpp import Llama
import json
from transformers import pipeline
import uuid
import math
from collections import namedtuple
from PIL import Image
import time
import signal
from contextlib import contextmanager
import threading

# Configure logging for debugging purposes
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

class SpeedEstimator:
    def __init__(self):
        self.prev_positions = {}
        self.fps = 30  # Assuming 30 FPS video
        
    def estimate_speed(self, track: dict, log_callback=None) -> float:
        """Estimate speed of a tracked object."""
        try:
            current_pos = track['xyxy']
            track_id = track['tracker_id']
            if track_id in self.prev_positions:
                prev_pos = self.prev_positions[track_id]
                distance = np.sqrt(
                    (current_pos[0] - prev_pos[0])**2 + 
                    (current_pos[1] - prev_pos[1])**2
                )
                speed = (distance * 0.1 * 3.6) / (1/self.fps)
                self.prev_positions[track_id] = current_pos
                return speed
            else:
                self.prev_positions[track_id] = current_pos
                return 0.0
        except Exception as e:
            error_msg = f"Speed estimation failed: {e}"
            if log_callback:
                log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Speed estimation failed: {e}") # Removed conflicting logging
            return 0.0

class BehaviorAnalyzer:
    def __init__(self):
        self.suspicious_patterns = {
            'rapid_movement': 10.0,  # Speed threshold
            'loitering_time': 30,    # Seconds
            'aggressive_pose': 0.8   # Confidence threshold
        }
    
    def analyze_behavior(self, pose: Dict, speed: float, face: Dict, log_callback=None) -> Dict:
        """Analyze behavior based on pose, speed, and facial expressions."""
        try:
            behavior = {
                'activity': 'unknown',
                'is_suspicious': False,
                'is_aggressive': False,
                'description': ''
            }
            
            # Analyze movement
            if speed > self.suspicious_patterns['rapid_movement']:
                behavior['activity'] = 'running'
                behavior['is_suspicious'] = True
                behavior['description'] = 'Rapid movement detected'
            
            # Analyze pose
            if pose.get('pose_type') == 'aggressive':
                behavior['is_aggressive'] = True
                behavior['description'] = 'Aggressive pose detected'
            
            # Analyze facial expressions
            if face.get('emotion') in ['angry', 'fear']:
                behavior['is_suspicious'] = True
                behavior['description'] = f'Suspicious emotion detected: {face["emotion"]}'
            
            return behavior
        except Exception as e:
            error_msg = f"Behavior analysis failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Behavior analysis failed: {e}") # Removed conflicting logging
            return {'activity': 'unknown', 'is_suspicious': False, 'is_aggressive': False}

class SpoofDetector:
    """Detects various types of spoofing attempts in video."""
    def __init__(self):
        self.spoof_types = {
            'mask': False,
            'photo': False,
            'video': False,
            'deepfake': False
        }
    
    def detect_spoofing(self, frame: np.ndarray, face_region: np.ndarray, log_callback=None) -> Dict:
        """Detect various types of spoofing attempts."""
        try:
            results = {
                'is_spoofed': False,
                'spoof_type': None,
                'confidence': 0.0,
                'details': []
            }
            
            # Check for mask spoofing
            if self._detect_mask_spoofing(face_region, log_callback):
                results['is_spoofed'] = True
                results['spoof_type'] = 'mask'
                results['confidence'] = 0.85
                results['details'].append("Potential mask spoofing detected")
            
            # Check for photo spoofing
            if self._detect_photo_spoofing(frame, face_region, log_callback):
                results['is_spoofed'] = True
                results['spoof_type'] = 'photo'
                results['confidence'] = 0.90
                results['details'].append("Potential photo spoofing detected")
            
            # Check for video spoofing
            if self._detect_video_spoofing(frame, log_callback):
                results['is_spoofed'] = True
                results['spoof_type'] = 'video'
                results['confidence'] = 0.95
                results['details'].append("Potential video spoofing detected")
            
            # Check for deepfake
            if self._detect_deepfake(face_region, log_callback):
                results['is_spoofed'] = True
                results['spoof_type'] = 'deepfake'
                results['confidence'] = 0.88
                results['details'].append("Potential deepfake detected")
            
            return results
        except Exception as e:
            error_msg = f"Spoof detection failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Spoof detection failed: {e}") # Removed conflicting logging
            return {'is_spoofed': False, 'spoof_type': None, 'confidence': 0.0, 'details': []}

    def _detect_mask_spoofing(self, face_region: np.ndarray, log_callback=None) -> bool:
        """Detect if face is covered by a mask."""
        try:
            # Implement mask detection logic
            return False
        except Exception as e:
            error_msg = f"Mask spoofing detection failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Mask spoofing detection failed: {e}") # Removed conflicting logging
            return False

    def _detect_photo_spoofing(self, frame: np.ndarray, face_region: np.ndarray, log_callback=None) -> bool:
        """Detect if face is from a photo."""
        try:
            # Implement photo spoofing detection logic
            return False
        except Exception as e:
            error_msg = f"Photo spoofing detection failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Photo spoofing detection failed: {e}") # Removed conflicting logging
            return False

    def _detect_video_spoofing(self, frame: np.ndarray, log_callback=None) -> bool:
        """Detect if video is being replayed."""
        try:
            # Implement video spoofing detection logic
            return False
        except Exception as e:
            error_msg = f"Video spoofing detection failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Video spoofing detection failed: {e}") # Removed conflicting logging
            return False

    def _detect_deepfake(self, face_region: np.ndarray, log_callback=None) -> bool:
        """Detect if face is a deepfake."""
        try:
            # Implement deepfake detection logic
            return False
        except Exception as e:
            error_msg = f"Deepfake detection failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Deepfake detection failed: {e}") # Removed conflicting logging
            return False

class ThreatAnalyzer:
    """Analyzes potential threats and suspicious behaviors."""
    def __init__(self):
        self.threat_levels = {
            'low': 0,
            'medium': 1,
            'high': 2,
            'critical': 3
        }
    
    def analyze_threat(self, behavior: Dict, detections: List[Dict], log_callback=None) -> Dict:
        """Analyze potential threats based on behavior and detections."""
        try:
            # Initialize threat with default values
            threat = {
                'level': 'low',
                'confidence': 0.0,
                'type': 'none',
                'details': []
            }
            
            if not behavior or not isinstance(behavior, dict):
                return threat
            
            # Check for aggressive behavior
            if behavior.get('is_aggressive', False):
                threat['level'] = 'high'
                threat['type'] = 'aggressive_behavior'
                threat['confidence'] = 0.8
                threat['details'].append("Aggressive behavior detected")
            
            # Check for suspicious behavior
            if behavior.get('is_suspicious', False):
                threat['level'] = 'medium'
                threat['type'] = 'suspicious_behavior'
                threat['confidence'] = 0.6
                threat['details'].append("Suspicious behavior detected")
            
            # Check for weapons in detections
            if detections:
                weapons = [d for d in detections if d.get('class') in ['knife', 'gun']]
                if weapons:
                    threat['level'] = 'critical'
                    threat['type'] = 'weapon_detected'
                    threat['confidence'] = 0.9
                    threat['details'].append(f"Potential weapon detected: {weapons[0].get('class')}")
            
            return threat
            
        except Exception as e:
            error_msg = f"Threat analysis failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Threat analysis failed: {e}") # Removed conflicting logging
            return {
                'level': 'low',
                'confidence': 0.0,
                'type': 'none',
                'details': []
            }

class TrafficAnalyzer:
    """Analyzes traffic patterns and violations."""
    def __init__(self):
        self.violation_types = {
            'speeding': False,
            'red_light': False,
            'wrong_way': False,
            'illegal_parking': False
        }
    
    def analyze_traffic(self, frame: np.ndarray, detections: List[Dict], log_callback=None) -> Dict:
        """Analyzes traffic patterns and detect violations."""
        try:
            analysis = {
                'violations': [],
                'traffic_flow': 'normal',
                'density': 'low',
                'details': []
            }
            
            # Analyze vehicle speeds
            for detection in detections:
                if detection['type'] in ['car', 'truck', 'motorcycle', 'bus']:
                    speed = self._estimate_speed(detection)
                    if speed > 50:  # Speed threshold
                        analysis['violations'].append({
                            'type': 'speeding',
                            'vehicle': detection['type'],
                            'speed': speed
                        })
            
            return analysis
        except Exception as e:
            error_msg = f"Traffic analysis failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Traffic analysis failed: {e}") # Removed conflicting logging
            return {'violations': [], 'traffic_flow': 'unknown', 'density': 'unknown', 'details': []}

    def _estimate_speed(self, detection: Dict, log_callback=None) -> float:
        """Estimate vehicle speed based on detection bbox."""
        try:
            # Implement speed estimation logic
            return 0.0
        except Exception as e:
            error_msg = f"Vehicle speed estimation failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Vehicle speed estimation failed: {e}") # Removed conflicting logging
            return 0.0

    def _estimate_direction(self, detection: Dict, log_callback=None) -> str:
        """Estimate vehicle direction based on bbox position."""
        try:
            # Calculate center point
            center_x = (detection['bbox'][0] + detection['bbox'][2]) / 2
            
            # Estimate direction based on position
            if center_x < 0.3:
                return 'left'
            elif center_x > 0.7:
                return 'right'
            else:
                return 'straight'
        except Exception as e:
            error_msg = f"Vehicle direction estimation failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Vehicle direction estimation failed: {e}") # Removed conflicting logging
            return 'unknown'

class ObjectTracker:
    """Advanced object tracking with behavior analysis."""
    def __init__(self):
        self.track_history = {}
        self.movement_patterns = {}
    
    def track_object(self, detection: Dict, frame_idx: int, log_callback=None) -> Dict:
        """Track an object across frames."""
        try:
            track_info = {
                'id': detection['id'],
                'type': detection['type'],
                'position': detection['bbox'],
                'movement': 'unknown',
                'pattern': 'unknown',
                'history': []
            }
            
            # Update track history
            if detection['id'] not in self.track_history:
                self.track_history[detection['id']] = []
            self.track_history[detection['id']].append({
                'frame': frame_idx,
                'position': detection['bbox']
            })
            
            return track_info
        except Exception as e:
            error_msg = f"Object tracking failed: {e}"
            if log_callback:
                 log_callback(error_msg, 'error') # Use log_callback
            # logging.error(f"Object tracking failed: {e}") # Removed conflicting logging
            return {}

class LlamaEngine:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('model_path')
        if not self.model_path:
            raise ValueError("Model path not provided in config")

        # Initialize the model with larger context window
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Increased context window
                n_batch=512,  # Increased batch size
                n_threads=4   # Use multiple threads
            )
            logging.info(f"Initialized Llama model from {self.model_path} with 4096 context window")
        except Exception as e:
            logging.error(f"Failed to initialize Llama model: {e}")
            raise e # Re-raise the exception after logging

    def analyze_image(self, image: np.ndarray, prompt: str, log_callback=None) -> str:
        """Analyze an image using the Llama model."""
        try:
            # Drastic reduction in image processing for speed
            import base64
            from io import BytesIO
            from PIL import Image
            
            pil_image = Image.fromarray(image)
            # Very small resize
            max_size = 32 # Even smaller
            ratio = min(max_size/pil_image.width, max_size/pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.NEAREST) # Faster resampling
            
            # Convert to RGB if not already
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Super aggressive compression
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=5, optimize=True) # Lowest quality
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Minimal prompt for speed
            full_prompt = f"<image>{image_base64}</image>\nAnalyze: objects?"
            
            # Shorter timeout and tokens
            try:
                timer = threading.Timer(10.0, lambda: None) # Shorter timeout
                timer.start()
                response = self.model(
                    full_prompt,
                    max_tokens=10, # Very few tokens
                    temperature=0.5,
                    top_p=0.9,
                    stop=["\n"], # Stop after first line
                    echo=False
                )
                timer.cancel()

                if response and 'choices' in response and response['choices']:
                    # Return a simple confirmation
                    return "Analysis OK"
                else:
                    if log_callback:
                         # Use log_callback for specific model response failure
                         log_callback(None, "Llama model analysis failed to produce a response.", 'warning')
                    return "Analysis failed"
                    
            except TimeoutError:
                if log_callback:
                     # Use log_callback for timeout
                     log_callback(None, "Llama model analysis timed out.", 'warning')
                return "Analysis timeout"
            except Exception as e:
                 error_msg = f"Llama model inference error: {e}"
                 if log_callback:
                      # Use log_callback for inference errors
                      log_callback(None, error_msg, 'error')
                 return "Analysis error"
                
        except Exception as e:
            error_msg = f"Image processing error before Llama model: {e}"
            if log_callback:
                 # Use log_callback for pre-inference errors
                 log_callback(None, error_msg, 'error')
            return "Image processing error"

    def _analyze_frame(self, frame_path: str, progress_callback=None) -> dict:
        """Analyze a single frame using the Llama model."""
        # progress_callback here also acts as log_callback with levels
        log_callback = progress_callback # Alias for clarity
        try:
            # Load and preprocess the image
            image = cv2.imread(frame_path)
            if image is None:
                # Report load error via callback
                if log_callback:
                     log_callback(None, f"Error: Could not load image {os.path.basename(frame_path)}", 'error')
                # logging.error(f"Could not load image: {frame_path}") # Removed conflicting logging
                return {'frame_path': os.path.basename(frame_path), 'error': "Could not load image", 'timestamp': time.time()}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform simplified analysis
            # No progress_callback here; parent loop handles frame progress
            analysis_result = self.analyze_image(image_rgb, "Describe", log_callback=log_callback)

            # Report specific analysis outcomes via callback if needed, otherwise just return result.
            if analysis_result == "Image processing error":
                if log_callback:
                     log_callback(None, f"Warning: Image processing failed for {os.path.basename(frame_path)}", 'warning')
            elif analysis_result == "Analysis timeout":
                 if log_callback:
                      log_callback(None, f"Warning: Analysis timed out for {os.path.basename(frame_path)}", 'warning')
            elif analysis_result == "Analysis error":
                 if log_callback:
                      log_callback(None, f"Warning: Analysis failed for {os.path.basename(frame_path)}", 'warning')
            # else:
            #     if progress_callback:
            #          progress_callback(None, f"Analysis successful for {os.path.basename(frame_path)}", 'info')

            return {
                'frame_path': os.path.basename(frame_path),
                'analysis': analysis_result,
                'timestamp': time.time()
            }

        except Exception as e:
            error_msg = f"Error analyzing {os.path.basename(frame_path)}: {str(e)}"
            # Report analysis error via callback
            if log_callback:
                log_callback(None, error_msg, 'error')
            return {'frame_path': os.path.basename(frame_path), 'error': str(e), 'timestamp': time.time()}

    def _get_timestamp_from_frame(self, frame_filename: str) -> float:
        """Extract timestamp from frame filename."""
        try:
            # Extract frame number from filename (e.g., "frame_0001.jpg" -> 1)
            frame_num = int(frame_filename.split('_')[1].split('.')[0])
            # Convert frame number to timestamp (assuming 30fps)
            return frame_num / 30.0
        except Exception as e:
            logging.warning(f"Could not extract timestamp from {frame_filename}: {e}")
            return 0.0

    def _get_detection_description(self, class_name: str, confidence: float) -> str:
        """Generate a description for a detection."""
        try:
            confidence_percent = int(confidence * 100)
            return f"{class_name.capitalize()} detected with {confidence_percent}% confidence"
        except Exception as e:
            logging.warning(f"Error generating detection description: {e}")
            return f"{class_name.capitalize()} detected"

    def _analyze_movement(self, keypoints, log_callback=None) -> str:
        """Analyze movement based on pose keypoints."""
        try:
            # Calculate velocity between keypoints
            velocities = []
            for i in range(len(keypoints) - 1):
                if keypoints[i] is not None and keypoints[i+1] is not None:
                    v = keypoints[i+1] - keypoints[i]
                    velocities.append(np.linalg.norm(v))
            
            if not velocities:
                return 'unknown'
            
            avg_velocity = np.mean(velocities)
            
            # Classify movement
            if avg_velocity > 0.5:
                return 'running'
            elif avg_velocity > 0.2:
                return 'walking'
            else:
                return 'standing'
                
        except Exception as e:
            error_msg = f"Movement analysis failed: {e}"
            if log_callback:
                 log_callback(None, error_msg, 'error')
            return 'error analyzing movement'

    def _analyze_appearance(self, roi: np.ndarray, log_callback=None) -> Dict:
        """Analyze person's appearance."""
        try:
            if roi.size == 0:
                return {'clothing': 'unknown', 'height': 'unknown', 'build': 'unknown'}
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Basic color analysis
            colors = []
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'green': ([40, 50, 50], [80, 255, 255]),
                'black': ([0, 0, 0], [180, 255, 30]),
                'white': ([0, 0, 200], [180, 30, 255])
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if np.sum(mask) > 1000:
                    colors.append(color_name)
            
            return {
                'clothing': ', '.join(colors) if colors else 'unknown',
                'height': 'unknown',
                'build': 'unknown'
            }
        except Exception as e:
            error_msg = f"Error analyzing appearance: {e}"
            if log_callback:
                 log_callback(None, error_msg, 'error')
            return {'clothing': 'unknown', 'height': 'unknown', 'build': 'unknown'}

    def _analyze_pose(self, roi: np.ndarray, log_callback=None) -> Dict:
        """Analyze pose in the given region of interest."""
        try:
            if roi is None or roi.size == 0:
                return {
                    'has_pose': False,
                    'pose_type': 'unknown',
                    'confidence': 0.0,
                    'key_points': {},
                    'movement': 'unknown',
                    'posture': 'unknown'
                }

            # Run pose detection using pose_model directly
            pose_results = self.models['pose'](roi, verbose=False)[0]
            
            if pose_results.keypoints is None:
                return {
                    'has_pose': False,
                    'pose_type': 'unknown',
                    'confidence': 0.0,
                    'key_points': {},
                    'movement': 'unknown',
                    'posture': 'unknown'
                }
            
            # Extract keypoints
            keypoints = pose_results.keypoints.data[0]
            key_points = self._get_key_landmarks_from_list(keypoints)
            
            # Analyze movement
            movement = self._analyze_movement(keypoints)
            
            # Analyze posture
            posture = self._analyze_posture(key_points)
            
            return {
                'has_pose': True,
                'pose_type': self._classify_pose(key_points),
                'confidence': float(pose_results.boxes.conf[0]) if pose_results.boxes.conf.numel() > 0 else 0.0,
                'key_points': key_points,
                'movement': movement,
                'posture': posture
            }
                
        except Exception as e:
            error_msg = f"Error in pose analysis: {e}"
            if log_callback:
                 log_callback(None, error_msg, 'error')
            return {
                'has_pose': False,
                'pose_type': 'unknown',
                'confidence': 0.0,
                'key_points': {},
                'movement': 'unknown',
                'posture': 'unknown'
            }

    def _get_key_landmarks_from_list(self, landmarks, log_callback=None) -> Dict:
        """Extract key landmarks from pose detection results with improved error handling."""
        try:
            # Initialize empty key points dictionary with default structure
            key_points = {
                'nose': {'x': 0.0, 'y': 0.0},
                'left_shoulder': {'x': 0.0, 'y': 0.0},
                'right_shoulder': {'x': 0.0, 'y': 0.0},
                'left_elbow': {'x': 0.0, 'y': 0.0},
                'right_elbow': {'x': 0.0, 'y': 0.0},
                'left_wrist': {'x': 0.0, 'y': 0.0},
                'right_wrist': {'x': 0.0, 'y': 0.0},
                'left_hip': {'x': 0.0, 'y': 0.0},
                'right_hip': {'x': 0.0, 'y': 0.0},
                'left_knee': {'x': 0.0, 'y': 0.0},
                'right_knee': {'x': 0.0, 'y': 0.0},
                'left_ankle': {'x': 0.0, 'y': 0.0},
                'right_ankle': {'x': 0.0, 'y': 0.0}
            }

            if landmarks is None or not hasattr(landmarks, 'cpu'):
                logging.debug("No pose landmarks detected")
                return key_points

            # Convert to numpy array and ensure proper shape
            try:
                landmarks_np = landmarks.cpu().numpy()
                if len(landmarks_np.shape) < 2:
                    logging.debug("Invalid landmarks shape")
                    return key_points
            except Exception as e:
                logging.debug(f"Error converting landmarks to numpy: {e}")
                return key_points

            # Define landmark indices and names
            landmark_indices = {
                'nose': 0,
                'left_shoulder': 5,
                'right_shoulder': 6,
                'left_elbow': 7,
                'right_elbow': 8,
                'left_wrist': 9,
                'right_wrist': 10,
                'left_hip': 11,
                'right_hip': 12,
                'left_knee': 13,
                'right_knee': 14,
                'left_ankle': 15,
                'right_ankle': 16
            }

            # Extract each landmark with proper error handling
            for name, idx in landmark_indices.items():
                    if idx < len(landmarks_np):
                        # Ensure we're getting a single point (x,y) pair
                        point = landmarks_np[idx]
                        if len(point) >= 2:  # Must have at least x,y coordinates
                            key_points[name] = {
                                'x': float(point[0]),
                                'y': float(point[1])
                            }
        except Exception as e:
            error_msg = f"Error in _get_key_landmarks_from_list: {str(e)}"
            if log_callback:
                 log_callback(None, error_msg, 'error')
            # Return the default key_points structure even on error
            return key_points

    def _calculate_angle(self, point1: Dict, point2: Dict, log_callback=None) -> float:
        """Calculate angle between two points."""
        try:
            dx = point2['x'] - point1['x']
            dy = point2['y'] - point1['y']
            return abs(math.degrees(math.atan2(dy, dx)))
        except Exception as e:
            error_msg = f"Error calculating angle: {e}"
            if log_callback:
                 log_callback(None, error_msg, 'error')
            return 0.0

    def _calculate_movement_vector(self, point1: Dict, point2: Dict, log_callback=None) -> Dict:
        """Calculate movement vector between two points."""
        try:
            dx = point2['x'] - point1['x']
            dy = point2['y'] - point1['y']
            
            # Calculate direction
            angle = np.degrees(np.arctan2(dy, dx))
            direction = self._get_direction_from_angle(angle)
            
            # Calculate speed (magnitude of movement)
            speed = np.sqrt(dx*dx + dy*dy)
            
            # Calculate stability (how straight the movement is)
            stability = 1.0 - (abs(dx) / (abs(dx) + abs(dy))) if (abs(dx) + abs(dy)) > 0 else 0.0
            
            return {
                'direction': direction,
                'speed': float(speed),
                'stability': float(stability)
            }
        except Exception as e:
            logging.error(f"Error calculating movement vector: {e}")
            return {'direction': 'unknown', 'speed': 0.0, 'stability': 0.0}
            
    def _get_direction_from_angle(self, angle: float) -> str:
        """Convert angle to cardinal direction."""
        try:
            # Normalize angle to 0-360
            angle = angle % 360
            
            if 337.5 <= angle or angle < 22.5:
                return "right"
            elif 22.5 <= angle < 67.5:
                return "down-right"
            elif 67.5 <= angle < 112.5:
                return "down"
            elif 112.5 <= angle < 157.5:
                return "down-left"
            elif 157.5 <= angle < 202.5:
                return "left"
            elif 202.5 <= angle < 247.5:
                return "up-left"
            elif 247.5 <= angle < 292.5:
                return "up"
            else:  # 292.5 <= angle < 337.5
                return "up-right"
        except Exception as e:
            logging.error(f"Error getting direction from angle: {e}")
            return "unknown"

    def _classify_pose(self, key_points: Dict[str, Tuple[float, float]]) -> str:
        """Classify the overall pose type."""
        try:
            if not key_points:
                return 'unknown'
                
            # Check for aggressive poses
            if self._is_aggressive_pose(key_points):
                return 'aggressive'
                
            # Check for hiding poses
            if self._is_hiding_pose(key_points):
                return 'hiding'
                
            # Check for unusual poses
            if self._is_unusual_pose(key_points):
                return 'unusual'
                
            return 'normal'
        except Exception as e:
            logging.error(f"Error classifying pose: {e}")
            return 'unknown'

    def _is_aggressive_pose(self, key_points: Dict[str, Tuple[float, float]]) -> bool:
        """Check if the pose indicates aggressive behavior."""
        try:
            if not key_points:
                return False
                
            # Check for raised arms
            if self._has_raised_arms(key_points):
                return True
                
            # Check for forward lean
            if self._has_forward_lean(key_points):
                return True
                
            # Check for unusual arm positions
            if self._has_unusual_arm_positions(key_points):
                return True
                
            return False
        except Exception as e:
            logging.error(f"Error checking aggressive pose: {e}")
            return False

    def _is_hiding_pose(self, key_points: Dict[str, Tuple[float, float]]) -> bool:
        """Check if the pose indicates hiding behavior."""
        try:
            if not key_points:
                return False
                
            # Check for crouching
            if self._is_crouching(key_points):
                return True
                
            # Check for face covering
            if self._is_covering_face(key_points):
                return True
                
            return False
        except Exception as e:
            logging.error(f"Error checking hiding pose: {e}")
            return False

    def _is_unusual_pose(self, key_points: Dict[str, Tuple[float, float]]) -> bool:
        """Check if the pose is unusual."""
        try:
            if not key_points:
                return False
                
            # Check for unusual arm positions
            if self._has_unusual_arm_positions(key_points):
                return True
                
            # Check for extreme angles
            if self._has_extreme_angles(key_points):
                return True
                
            return False
        except Exception as e:
            logging.error(f"Error checking unusual pose: {e}")
            return False

    def _has_extreme_angles(self, key_points: Dict[str, Tuple[float, float]]) -> bool:
        """Check if the pose has extreme angles."""
        try:
            if not key_points:
                return False
                
            # Check shoulder angle
            if 'left_shoulder' in key_points and 'right_shoulder' in key_points and 'neck' in key_points:
                shoulder_angle = self._calculate_angle(
                    key_points['left_shoulder'],
                    key_points['neck'],
                    key_points['right_shoulder']
                )
                if shoulder_angle > 150 or shoulder_angle < 30:
                    return True
                    
            # Check hip angle
            if 'left_hip' in key_points and 'right_hip' in key_points and 'pelvis' in key_points:
                hip_angle = self._calculate_angle(
                    key_points['left_hip'],
                    key_points['pelvis'],
                    key_points['right_hip']
                )
                if hip_angle > 150 or hip_angle < 30:
                    return True
                    
            return False
        except Exception as e:
            logging.error(f"Error checking extreme angles: {e}")
            return False

    def _analyze_behavior(self, roi, pose_results, detections) -> Dict:
        """Analyze behavior based on pose and context."""
        try:
            behavior = {
                'behavior_type': 'normal',
                'confidence': 0.0
            }
            
            if pose_results and pose_results.keypoints is not None:
                keypoints = pose_results.keypoints.data[0]
                
                # Check for unusual poses
                if self._is_unusual_pose(keypoints):
                    behavior['behavior_type'] = 'suspicious'
                    behavior['confidence'] = 0.7
                
                # Check for aggressive poses
                if self._is_aggressive_pose(keypoints):
                    behavior['behavior_type'] = 'aggressive'
                    behavior['confidence'] = 0.8
                
                # Check for hiding behavior
                if self._is_hiding_pose(keypoints):
                    behavior['behavior_type'] = 'hiding'
                    behavior['confidence'] = 0.6
            
            return behavior
            
        except Exception as e:
            logging.error(f"Error in behavior analysis: {e}")
            return {'behavior_type': 'normal', 'confidence': 0.0}

    def _is_unusual_pose(self, keypoints) -> bool:
        """Check if pose is unusual."""
        try:
            # Check for crouching
            if self._is_crouching(keypoints):
                return True
                
            # Check for unusual arm positions
            if self._has_unusual_arm_positions(keypoints):
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error in unusual pose check: {e}")
            return False

    def _is_aggressive_pose(self, keypoints) -> bool:
        """Check if pose indicates aggressive behavior."""
        try:
            # Check for raised arms
            if self._has_raised_arms(keypoints):
                return True
                
            # Check for forward lean
            if self._has_forward_lean(keypoints):
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error in aggressive pose check: {e}")
            return False

    def _is_hiding_pose(self, keypoints) -> bool:
        """Check if pose indicates hiding behavior."""
        try:
            # Check for crouching
            if self._is_crouching(keypoints):
                return True
                
            # Check for covering face
            if self._is_covering_face(keypoints):
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error in hiding pose check: {e}")
            return False

    def _estimate_vehicle_speed(self, bbox) -> str:
        """Estimate vehicle speed based on bbox size and position."""
        try:
            # Calculate bbox size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = width * height
            
            # Estimate speed based on size
            if size > 10000:
                return 'fast'
            elif size > 5000:
                return 'medium'
            else:
                return 'slow'
                
        except Exception as e:
            logging.error(f"Error in vehicle speed estimation: {e}")
            return 'unknown'

    def _estimate_vehicle_direction(self, bbox) -> str:
        """Estimate vehicle direction based on bbox position."""
        try:
            # Calculate center point
            center_x = (bbox[0] + bbox[2]) / 2
            
            # Estimate direction based on position
            if center_x < 0.3:
                return 'left'
            elif center_x > 0.7:
                return 'right'
            else:
                return 'center'
                
        except Exception as e:
            logging.error(f"Error in vehicle direction estimation: {e}")
            return 'unknown'

    def _is_carrying_item(self, detection, all_detections) -> bool:
        """Check if a person is carrying an item."""
        try:
            if detection['class'] not in ['backpack', 'bag', 'suitcase', 'laptop']:
                return False
                
            # Check if item is near a person
            for det in all_detections:
                if det[5] == 0:  # person class
                    person_bbox = det[:4]
                    item_bbox = detection['bbox']
                    
                    # Calculate IoU
                    iou = self._calculate_iou(person_bbox, item_bbox)
                    if iou > 0.3:
                        return True
                        
            return False
            
        except Exception as e:
            logging.error(f"Error in carrying item check: {e}")
            return False

    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        try:
            # Calculate intersection
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error in IoU calculation: {e}")
            return 0.0

    def _analyze_posture(self, key_points: Dict[str, Tuple[float, float]]) -> str:
        """Analyze posture based on key points."""
        try:
            if not key_points:
                return 'unknown'
                
            # Check for different postures
            if self._is_crouching(key_points):
                return 'crouching'
            elif self._has_raised_arms(key_points):
                return 'arms_raised'
            elif self._is_covering_face(key_points):
                return 'covering_face'
            elif self._has_unusual_arm_positions(key_points):
                return 'unusual_arms'
            elif self._has_forward_lean(key_points):
                return 'leaning_forward'
            else:
                return 'standing'
        except Exception as e:
            logging.error(f"Error analyzing posture: {e}")
            return 'unknown'

    def _analyze_movement_dynamics(self, key_points: Dict) -> Dict:
        """Analyze movement dynamics for behavior detection."""
        try:
            dynamics = {
                'is_fleeing': False,
                'is_loitering': False
            }
            
            # Analyze leg movement
            knees_bent = (abs(key_points['knees']['left'].y - key_points['hips']['left'].y) > 0.2 and 
                         abs(key_points['knees']['right'].y - key_points['hips']['right'].y) > 0.2)
            
            stride_wide = (abs(key_points['ankles']['left'].x - key_points['ankles']['right'].x) > 0.3)
            
            # Analyze body movement
            forward_lean = (key_points['hips']['left'].x < key_points['ankles']['left'].x and 
                          key_points['hips']['right'].x < key_points['ankles']['right'].x)
            
            # Determine behaviors
            if knees_bent and (stride_wide or forward_lean):
                dynamics['is_fleeing'] = True
            
            if not knees_bent and not stride_wide and not forward_lean:
                dynamics['is_loitering'] = True
            
            return dynamics
        except Exception as e:
            logging.warning(f"Movement dynamics analysis failed: {e}")
            return {}

    def _analyze_interaction_patterns(self, key_points: Dict) -> Dict:
        """Analyze interaction patterns with other objects/people."""
        try:
            patterns = {
                'is_interacting': False,
                'is_carrying': False
            }
            
            # Analyze arm positions for interaction
            arms_extended = (abs(key_points['wrists']['left'].x - key_points['shoulders']['left'].x) > 0.3 or 
                           abs(key_points['wrists']['right'].x - key_points['shoulders']['right'].x) > 0.3)
            
            arms_raised = (key_points['wrists']['left'].y < key_points['shoulders']['left'].y and 
                          key_points['wrists']['right'].y < key_points['shoulders']['right'].y)
            
            # Analyze body orientation for interaction
            body_forward = (key_points['shoulders']['left'].x < key_points['hips']['left'].x and 
                          key_points['shoulders']['right'].x < key_points['hips']['right'].x)
            
            # Determine behaviors
            if arms_extended and body_forward:
                patterns['is_interacting'] = True
            
            if arms_raised and not body_forward:
                patterns['is_carrying'] = True
            
            return patterns
        except Exception as e:
            logging.warning(f"Interaction pattern analysis failed: {e}")
            return {}

    def _is_interacting(self, landmarks, detections: List[Dict]) -> bool:
        """Detect interactions with other objects or people."""
        try:
            # Get wrist positions
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            
            # Check for nearby objects/people
            for detection in detections:
                if detection['type'] != 'person':
                    # Calculate distance to object
                    obj_center = [(detection['bbox'][0] + detection['bbox'][2])/2,
                                (detection['bbox'][1] + detection['bbox'][3])/2]
                    
                    # Check if wrists are near object
                    left_dist = np.sqrt((left_wrist.x - obj_center[0])**2 + 
                                      (left_wrist.y - obj_center[1])**2)
                    right_dist = np.sqrt((right_wrist.x - obj_center[0])**2 + 
                                       (right_wrist.y - obj_center[1])**2)
                    
                    if left_dist < 0.2 or right_dist < 0.2:
                        return True
            
            return False
        except Exception as e:
            logging.warning(f"Interaction detection failed: {e}")
            return False

    def _analyze_vehicle_characteristics(self, vehicle_roi: np.ndarray) -> Dict:
        """Analyze vehicle characteristics."""
        try:
            if vehicle_roi.size == 0:
                return {'color': 'unknown', 'size': 'unknown', 'condition': 'unknown'}
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            
            # Basic color detection
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'green': ([40, 50, 50], [80, 255, 255]),
                'black': ([0, 0, 0], [180, 255, 30]),
                'white': ([0, 0, 200], [180, 30, 255])
            }
            
            colors = []
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if np.sum(mask) > 1000:
                    colors.append(color_name)
            
            return {
                'color': ', '.join(colors) if colors else 'unknown',
                'size': 'unknown',
                'condition': 'unknown'
            }
        except Exception as e:
            logging.warning(f"Error analyzing vehicle characteristics: {e}")
            return {'color': 'unknown', 'size': 'unknown', 'condition': 'unknown'}

    def _assess_risks(self, analyses: List[Dict]) -> Dict:
        """Assess potential risks and threats in the video."""
        return {
            'aggressive': self._detect_aggressive_behavior(analyses),
            'theft_cases': self._detect_potential_theft(analyses),
            'unusual_zones': self._identify_unusual_activity_zones(analyses)
        }

    def _generate_timelines(self, analyses: List[Dict], timestamps: List[str]) -> List[Dict]:
        """Generate a timeline of significant events."""
        timelines = []
        for i, analysis in enumerate(analyses):
            if analysis.get('significant_events'):
                timelines.append({
                    'timestamp': timestamps[i],
                    'event': analysis['significant_events']
                })
        return timelines

    def _assess_overall_risk(self, analyses: List[Dict]) -> str:
        """Assess the overall risk level based on analysis results."""
        try:
            risk_score = 0
            
            # Check for aggressive behavior
            if self._detect_aggressive_behavior(analyses):
                risk_score += 3
            
            # Check for theft cases
            theft_cases = self._detect_potential_theft(analyses)
            risk_score += theft_cases * 2
            
            # Check for unusual zones
            unusual_zones = self._identify_unusual_activity_zones(analyses)
            risk_score += len(unusual_zones)
            
            # Check for critical events
            critical_events = sum(1 for a in analyses if a.get('alerts'))
            risk_score += critical_events
            
            # Determine risk level based on score
            if risk_score >= 5:
                return "HIGH"
            elif risk_score >= 3:
                return "MEDIUM"
            elif risk_score >= 1:
                return "LOW"
            else:
                return "MINIMAL"
            
        except Exception as e:
            logging.error(f"Error assessing overall risk: {e}")
            return "UNKNOWN"

    def _has_raised_arms(self, keypoints: Dict) -> bool:
        """Check if a person has raised arms based on keypoint positions."""
        try:
            if not all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist', 
                                              'right_shoulder', 'right_elbow', 'right_wrist']):
                return False
                
            # Get y-coordinates of keypoints
            left_shoulder_y = keypoints['left_shoulder']['y']
            left_wrist_y = keypoints['left_wrist']['y']
            right_shoulder_y = keypoints['right_shoulder']['y']
            right_wrist_y = keypoints['right_wrist']['y']
            
            # Check if wrists are above shoulders
            return (left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y)
        except Exception as e:
            logging.error(f"Error in raised arms check: {e}")
            return False

    def _is_crouching(self, keypoints: Dict) -> bool:
        """Check if a person is crouching based on keypoint positions."""
        try:
            if not all(k in keypoints for k in ['left_hip', 'left_knee', 'left_ankle',
                                              'right_hip', 'right_knee', 'right_ankle']):
                return False
                
            # Get y-coordinates of keypoints
            left_hip_y = keypoints['left_hip']['y']
            left_knee_y = keypoints['left_knee']['y']
            right_hip_y = keypoints['right_hip']['y']
            right_knee_y = keypoints['right_knee']['y']
            
            # Calculate knee angles
            left_knee_angle = self._calculate_angle(
                {'x': keypoints['left_hip']['x'], 'y': left_hip_y},
                {'x': keypoints['left_knee']['x'], 'y': left_knee_y}
            )
            right_knee_angle = self._calculate_angle(
                {'x': keypoints['right_hip']['x'], 'y': right_hip_y},
                {'x': keypoints['right_knee']['x'], 'y': right_knee_y}
            )
            
            # Person is crouching if knee angles are less than 90 degrees
            return (left_knee_angle < 90 and right_knee_angle < 90)
        except Exception as e:
            logging.error(f"Error in crouching check: {e}")
            return False

    def _is_covering_face(self, keypoints: Dict) -> bool:
        """Check if a person is covering their face."""
        try:
            if not all(k in keypoints for k in ['left_hand', 'right_hand', 'nose', 'left_eye', 'right_eye']):
                return False
                
            # Get coordinates
            left_hand = keypoints['left_hand']
            right_hand = keypoints['right_hand']
            nose = keypoints['nose']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Calculate distances between hands and face
            left_hand_to_face = self._calculate_distance(left_hand, nose)
            right_hand_to_face = self._calculate_distance(right_hand, nose)
            
            # Check if hands are close to face
            return (left_hand_to_face < 50 or right_hand_to_face < 50)
        except Exception as e:
            logging.error(f"Error in face covering check: {e}")
            return False

    def _has_unusual_arm_positions(self, keypoints: Dict) -> bool:
        """Check for unusual arm positions."""
        try:
            if not all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist',
                                          'right_shoulder', 'right_elbow', 'right_wrist']):
                return False
                
            # Get coordinates
            left_shoulder = keypoints['left_shoulder']
            left_elbow = keypoints['left_elbow']
            left_wrist = keypoints['left_wrist']
            right_shoulder = keypoints['right_shoulder']
            right_elbow = keypoints['right_elbow']
            right_wrist = keypoints['right_wrist']
            
            # Calculate arm angles
            left_arm_angle = self._calculate_angle(left_shoulder, left_elbow)
            right_arm_angle = self._calculate_angle(right_shoulder, right_elbow)
            
            # Check for unusual angles
            return (left_arm_angle > 150 or right_arm_angle > 150)
        except Exception as e:
            logging.error(f"Error in unusual arm positions check: {e}")
            return False

    def _has_forward_lean(self, keypoints: Dict) -> bool:
        """Check if a person is leaning forward aggressively."""
        try:
            if not all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                return False
                
            # Get coordinates
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            left_hip = keypoints['left_hip']
            right_hip = keypoints['right_hip']
            
            # Calculate angles
            shoulder_angle = self._calculate_angle(left_shoulder, right_shoulder)
            hip_angle = self._calculate_angle(left_hip, right_hip)
            
            # Check for forward lean
            return abs(shoulder_angle - hip_angle) > 30
        except Exception as e:
            logging.error(f"Error in forward lean check: {e}")
            return False

    def _calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate Euclidean distance between two points."""
        try:
            return math.sqrt((point2['x'] - point1['x'])**2 + (point2['y'] - point1['y'])**2)
        except Exception as e:
            logging.error(f"Error calculating distance: {e}")
            return float('inf')

    def _analyze_scene(self, analyses: List[Dict]) -> Dict:
        """Analyze scene characteristics from video analysis."""
        try:
            if not analyses:
                return {
                    'location_type': 'Unknown',
                    'lighting': 'Unknown',
                    'crowd_density': 'Unknown',
                    'activity_level': 'Low',
                    'weather_conditions': 'Unknown',
                    'time_of_day': 'Unknown',
                    'conditions': 'Unknown'
                }
            
            # Aggregate scene data from all analyses
            location_types = []
            lighting_conditions = []
            crowd_densities = []
            activity_levels = []
            weather_conditions = []
            time_of_day = []
            conditions = []
            
            for analysis in analyses:
                scene_context = analysis.get('scene_context', {})
                location_types.append(scene_context.get('location_type', 'Unknown'))
                lighting_conditions.append(scene_context.get('lighting', 'Unknown'))
                crowd_densities.append(scene_context.get('crowd_density', 'Unknown'))
                activity_levels.append(scene_context.get('activity_level', 'Low'))
                weather_conditions.append(scene_context.get('weather_conditions', 'Unknown'))
                time_of_day.append(scene_context.get('time_of_day', 'Unknown'))
                conditions.append(scene_context.get('conditions', 'Unknown'))
            
            # Determine most common values
            def most_common(lst):
                return max(set(lst), key=lst.count) if lst else 'Unknown'
            
            # Calculate average number of people for crowd density
            total_people = sum(len([d for d in analysis.get('detections', []) if d.get('class') == 'person']) 
                             for analysis in analyses)
            avg_people = total_people / len(analyses) if analyses else 0
            
            # Determine crowd density based on average people count
            if avg_people > 20:
                crowd_density = 'High'
            elif avg_people > 10:
                crowd_density = 'Medium'
            elif avg_people > 0:
                crowd_density = 'Low'
            else:
                crowd_density = 'None'
            
            # Analyze activity level based on movement and interactions
            total_movements = sum(len([d for d in analysis.get('detections', []) 
                                     if d.get('movement', {}).get('is_moving', False)]) 
                                for analysis in analyses)
            avg_movements = total_movements / len(analyses) if analyses else 0
            
            if avg_movements > 15:
                activity_level = 'High'
            elif avg_movements > 5:
                activity_level = 'Medium'
            else:
                activity_level = 'Low'
            
            # Analyze lighting conditions
            lighting = most_common(lighting_conditions)
            if lighting == 'Unknown':
                # Try to determine lighting from image brightness
                brightness_values = []
                for analysis in analyses:
                    if 'frame_info' in analysis and 'brightness' in analysis['frame_info']:
                        brightness_values.append(analysis['frame_info']['brightness'])
                
                if brightness_values:
                    avg_brightness = sum(brightness_values) / len(brightness_values)
                    if avg_brightness > 0.7:
                        lighting = 'Bright'
                    elif avg_brightness > 0.4:
                        lighting = 'Moderate'
                    else:
                        lighting = 'Low'
            
            # Analyze weather conditions
            weather = most_common(weather_conditions)
            if weather == 'Unknown':
                # Try to determine weather from scene analysis
                weather_indicators = {
                    'rain': 0,
                    'snow': 0,
                    'fog': 0,
                    'clear': 0
                }
                
                for analysis in analyses:
                    scene_context = analysis.get('scene_context', {})
                    for condition in weather_indicators:
                        if scene_context.get(condition, False):
                            weather_indicators[condition] += 1
                
                if weather_indicators['rain'] > len(analyses) * 0.3:
                    weather = 'Rainy'
                elif weather_indicators['snow'] > len(analyses) * 0.3:
                    weather = 'Snowy'
                elif weather_indicators['fog'] > len(analyses) * 0.3:
                    weather = 'Foggy'
                else:
                    weather = 'Clear'
            
            # Determine time of day
            time = most_common(time_of_day)
            if time == 'Unknown':
                # Try to determine time from lighting and scene analysis
                if lighting == 'Low':
                    time = 'Night'
                elif lighting == 'Bright':
                    time = 'Day'
                else:
                    time = 'Dawn/Dusk'
            
            # Combine conditions
            conditions_list = []
            if weather != 'Clear':
                conditions_list.append(weather.lower())
            if lighting != 'Moderate':
                conditions_list.append(lighting.lower())
            if crowd_density != 'Low':
                conditions_list.append(f"{crowd_density.lower()} crowd")
            if activity_level != 'Low':
                conditions_list.append(f"{activity_level.lower()} activity")
            
            conditions_str = ', '.join(conditions_list) if conditions_list else 'Normal'
            
            return {
                'location_type': most_common(location_types),
                'lighting': lighting,
                'crowd_density': crowd_density,
                'activity_level': activity_level,
                'weather_conditions': weather,
                'time_of_day': time,
                'conditions': conditions_str,
                'average_people_count': int(avg_people),
                'total_detections': sum(len(analysis.get('detections', [])) for analysis in analyses),
                'movement_analysis': {
                    'total_movements': total_movements,
                    'average_movements': avg_movements,
                    'movement_patterns': self._analyze_movement_patterns(analyses)
                }
            }
        except Exception as e:
            logging.error(f"Error analyzing scene: {e}")
            return {
                'location_type': 'Unknown',
                'lighting': 'Unknown',
                'crowd_density': 'Unknown',
                'activity_level': 'Low',
                'weather_conditions': 'Unknown',
                'time_of_day': 'Unknown',
                'conditions': 'Unknown'
            }

    def _analyze_movement_patterns(self, analyses: List[Dict]) -> Dict:
        """Analyze movement patterns in the scene."""
        try:
            patterns = {
                'linear': 0,
                'circular': 0,
                'random': 0,
                'stationary': 0
            }
            
            for analysis in analyses:
                for detection in analysis.get('detections', []):
                    movement = detection.get('movement', {})
                    if movement.get('is_moving', False):
                        pattern = movement.get('pattern', 'random')
                        patterns[pattern] = patterns.get(pattern, 0) + 1
                    else:
                        patterns['stationary'] += 1
            
            # Normalize patterns
            total = sum(patterns.values())
            if total > 0:
                patterns = {k: v/total for k, v in patterns.items()}
            
            return patterns
        except Exception as e:
            logging.error(f"Error analyzing movement patterns: {e}")
            return {}

    def _is_unusual_pose(self, keypoints: Dict) -> bool:
        """Check if a pose is unusual."""
        try:
            if not isinstance(keypoints, dict):
                return False
                
            # Check for various unusual poses
            return any([
                self._is_crouching(keypoints),
                self._has_raised_arms(keypoints),
                self._is_covering_face(keypoints),
                self._has_unusual_arm_positions(keypoints),
                self._has_forward_lean(keypoints)
            ])
        except Exception as e:
            logging.error(f"Error in unusual pose check: {e}")
            return False

    def _is_aggressive_pose(self, keypoints: Dict) -> bool:
        """Check if a pose indicates aggressive behavior."""
        try:
            if not isinstance(keypoints, dict):
                return False
                
            # Check for aggressive indicators
            return any([
                self._has_raised_arms(keypoints),
                self._has_forward_lean(keypoints),
                self._has_unusual_arm_positions(keypoints)
            ])
        except Exception as e:
            logging.error(f"Error in aggressive pose check: {e}")
            return False

    def _is_hiding_pose(self, keypoints: Dict) -> bool:
        """Check if a pose indicates hiding behavior."""
        try:
            if not isinstance(keypoints, dict):
                return False
                
            # Check for hiding indicators
            return any([
                self._is_crouching(keypoints),
                self._is_covering_face(keypoints)
            ])
        except Exception as e:
            logging.error(f"Error in hiding pose check: {e}")
            return False

    def _analyze_lighting(self, analyses: List[Dict]) -> str:
        """Analyze lighting conditions across frames."""
        try:
            lighting_values = []
            for analysis in analyses:
                if 'scene_context' in analysis and 'lighting_conditions' in analysis['scene_context']:
                    lighting = analysis['scene_context']['lighting_conditions']
                    if lighting != 'unknown':
                        lighting_values.append(lighting)
            
            if not lighting_values:
                return 'unknown'
                
            # Return most common lighting condition
            return max(set(lighting_values), key=lighting_values.count)
        except Exception as e:
            logging.error(f"Error analyzing lighting: {e}")
            return 'unknown'

    def _analyze_crowd_density(self, analyses: List[Dict]) -> str:
        """Analyze crowd density across frames."""
        try:
            density_values = []
            for analysis in analyses:
                if 'scene_context' in analysis and 'crowd_density' in analysis['scene_context']:
                    density = analysis['scene_context']['crowd_density']
                    if density != 'unknown':
                        density_values.append(density)
            
            if not density_values:
                return 'unknown'
                
            # Return most common density level
            return max(set(density_values), key=density_values.count)
        except Exception as e:
            logging.error(f"Error analyzing crowd density: {e}")
            return 'unknown'

    def _analyze_activity_level(self, analyses: List[Dict]) -> str:
        """Analyze activity level across frames."""
        try:
            activity_counts = {'low': 0, 'medium': 0, 'high': 0}
            
            for analysis in analyses:
                detections = analysis.get('detections', [])
                moving_objects = sum(1 for d in detections if d.get('movement') in ['running', 'walking'])
                
                if moving_objects > 5:
                    activity_counts['high'] += 1
                elif moving_objects > 2:
                    activity_counts['medium'] += 1
                else:
                    activity_counts['low'] += 1
            
            # Return most common activity level
            return max(activity_counts.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logging.error(f"Error analyzing activity level: {e}")
            return 'unknown'

    def _identify_notable_objects(self, analyses: List[Dict]) -> List[str]:
        """Identify notable objects across frames."""
        try:
            notable_objects = set()
            for analysis in analyses:
                detections = analysis.get('detections', [])
                for detection in detections:
                    if detection.get('confidence', 0) > 0.8:  # High confidence detections
                        notable_objects.add(detection.get('class', ''))
            
            return list(notable_objects)
        except Exception as e:
            logging.error(f"Error identifying notable objects: {e}")
            return []

    def _analyze_critical_events(self, analyses: List[Dict], timestamps: List[str]) -> List[Dict]:
        """Analyze and extract critical events from the analysis results.
        
        Args:
            analyses: List of analysis results
            timestamps: List of timestamps corresponding to each analysis
            
        Returns:
            List of critical events with timestamps and descriptions
        """
        try:
            critical_events = []
            if not analyses or not timestamps:
                return []
                
            for i, analysis in enumerate(analyses):
                if not isinstance(analysis, dict):
                    continue
                    
                timestamp = timestamps[i] if i < len(timestamps) else 'Unknown'
                events = analysis.get('events', {})
                
                # Check for specific event types
                if events.get('suspicious_activity', {}).get('detected'):
                    critical_events.append({
                        'timestamp': timestamp,
                        'type': 'suspicious_activity',
                        'confidence': events['suspicious_activity'].get('confidence', 0.0),
                        'description': events['suspicious_activity'].get('description', 'Suspicious activity detected')
                    })
                
                # Add more event types as needed
                
            return critical_events
            
        except Exception as e:
            logging.error(f"Error analyzing critical events: {e}", exc_info=True)
            return []
            
    def _detect_anomalies(self, analyses: List[Dict]) -> List[Dict]:
        """Detect anomalies in the analysis results.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            if not analyses:
                return []
                
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    continue
                    
                # Check for anomalies in detections
                for detection in analysis.get('detections', []):
                    if detection.get('confidence', 0) > 0.9 and detection.get('class') in ['weapon', 'suspicious_object']:
                        anomalies.append({
                            'type': 'suspicious_object',
                            'class': detection['class'],
                            'confidence': detection.get('confidence', 0.0),
                            'frame_info': analysis.get('frame_info', 'unknown')
                        })
                
                # Add more anomaly detection logic here
                
            return anomalies
            
        except Exception as e:
            logging.error(f"Error detecting anomalies: {e}", exc_info=True)
            return []
            
    def _analyze_critical_events(self, analyses: List[Dict], timestamps: List[str]) -> List[Dict]:
        """Analyze and extract critical events from the video analysis.
        
        Args:
            analyses: List of analysis results
            timestamps: List of timestamps corresponding to each analysis
            
        Returns:
            List of detected critical events with details
        """
        try:
            if not analyses or not timestamps or len(analyses) != len(timestamps):
                return []
                
            events = []
            
            for i, analysis in enumerate(analyses):
                if not isinstance(analysis, dict):
                    continue
                    
                timestamp = timestamps[i] if i < len(timestamps) else f"frame_{i}"
                frame_events = []
                
                # Check for high-confidence detections of important objects
                for detection in analysis.get('detections', []):
                    if not isinstance(detection, dict):
                        continue
                        
                    confidence = detection.get('confidence', 0)
                    class_name = str(detection.get('class', '')).lower()
                    
                    # Define critical objects and their thresholds
                    critical_objects = {
                        'weapon': 0.7,
                        'knife': 0.7,
                        'gun': 0.7,
                        'fire': 0.8,
                        'explosion': 0.8,
                        'accident': 0.75,
                        'fight': 0.8,
                        'fall': 0.8
                    }
                    
                    if class_name in critical_objects and confidence >= critical_objects[class_name]:
                        frame_events.append({
                            'type': class_name,
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'description': f"Detected {class_name} with {confidence:.0%} confidence"
                        })
                
                # Check for unusual activity
                if 'activity' in analysis and isinstance(analysis['activity'], dict):
                    activity = analysis['activity']
                    if activity.get('is_unusual', False):
                        frame_events.append({
                            'type': 'unusual_activity',
                            'timestamp': timestamp,
                            'confidence': activity.get('confidence', 0.7),
                            'description': activity.get('description', 'Unusual activity detected')
                        })
                
                # Add frame events to main events list
                events.extend(frame_events)
            
            # Remove duplicate events that are too close in time
            unique_events = []
            last_event = None
            for event in sorted(events, key=lambda x: x.get('timestamp', '')):
                if last_event is None or \
                   (event['type'] != last_event['type'] or 
                    str(event['timestamp']) != str(last_event['timestamp'])):
                    unique_events.append(event)
                    last_event = event
            
            return unique_events
            
        except Exception as e:
            logging.error(f"Error analyzing critical events: {e}", exc_info=True)
            return []
    
    def _analyze_scene(self, analyses: List[Dict]) -> Dict:
        """Analyze the scene context from the video analysis.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Dictionary containing scene analysis results
        """
        try:
            if not analyses:
                return {}
                
            scene_data = {
                'location_type': 'unknown',
                'lighting_conditions': 'unknown',
                'crowd_density': 'low',
                'weather_conditions': 'clear',
                'time_of_day': 'day',
                'activity_level': 'low'
            }
            
            # Count frames with different scene attributes
            location_types = {}
            lighting_conditions = {}
            crowd_densities = {}
            weather_conditions = {}
            time_of_days = {}
            activity_levels = {}
            
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    continue
                    
                # Get scene context
                scene = analysis.get('scene_context', {}) or {}
                
                # Update location type
                loc = str(scene.get('location_type', 'unknown')).lower()
                location_types[loc] = location_types.get(loc, 0) + 1
                
                # Update lighting conditions
                light = str(scene.get('lighting_conditions', 'unknown')).lower()
                lighting_conditions[light] = lighting_conditions.get(light, 0) + 1
                
                # Update crowd density
                crowd = str(scene.get('crowd_density', 'low')).lower()
                crowd_densities[crowd] = crowd_densities.get(crowd, 0) + 1
                
                # Update weather conditions
                weather = str(scene.get('weather_conditions', 'clear')).lower()
                weather_conditions[weather] = weather_conditions.get(weather, 0) + 1
                
                # Update time of day
                tod = str(scene.get('time_of_day', 'day')).lower()
                time_of_days[tod] = time_of_days.get(tod, 0) + 1
                
                # Update activity level
                activity = str(analysis.get('activity_level', 'low')).lower()
                activity_levels[activity] = activity_levels.get(activity, 0) + 1
            
            # Get most common values
            def get_most_common(d):
                return max(d.items(), key=lambda x: x[1])[0] if d else 'unknown'
            
            scene_data.update({
                'location_type': get_most_common(location_types),
                'lighting_conditions': get_most_common(lighting_conditions),
                'crowd_density': get_most_common(crowd_densities),
                'weather_conditions': get_most_common(weather_conditions),
                'time_of_day': get_most_common(time_of_days),
                'activity_level': get_most_common(activity_levels)
            })
            
            return scene_data
            
        except Exception as e:
            logging.error(f"Error analyzing scene: {e}", exc_info=True)
            return {
                'location_type': 'unknown',
                'lighting_conditions': 'unknown',
                'crowd_density': 'low',
                'weather_conditions': 'clear',
                'time_of_day': 'day',
                'activity_level': 'low',
                'error': str(e)
            }
            
    def _identify_environmental_factors(self, analyses: List[Dict]) -> List[str]:
        """Identify environmental factors from the analysis."""
        try:
            factors = []
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    continue
                    
                # Check lighting conditions
                if analysis.get('lighting_conditions'):
                    factors.append(f"Lighting: {analysis['lighting_conditions']}")
                    
                # Check weather conditions
                if analysis.get('weather_conditions'):
                    factors.append(f"Weather: {analysis['weather_conditions']}")
                    
                # Check crowd density
                if analysis.get('crowd_density'):
                    factors.append(f"Crowd: {analysis['crowd_density']}")
                    
            return list(set(factors))  # Remove duplicates
            
        except Exception as e:
            logging.error(f"Error identifying environmental factors: {e}", exc_info=True)
            return []

    def _limit_context_size(self, analyses: List[Dict], max_frames: int = 30) -> List[Dict]:
        """Limit the context size by selecting the most relevant frames."""
        if len(analyses) <= max_frames:
            return analyses
            
        # Sort analyses by confidence scores
        scored_analyses = []
        for i, analysis in enumerate(analyses):
            score = 0
            # Add scores from detections
            for det in analysis.get('detections', []):
                score += det.get('confidence', 0)
            # Add scores from events
            for event in analysis.get('events', {}).values():
                if isinstance(event, dict) and event.get('detected'):
                    score += event.get('confidence', 0)
            scored_analyses.append((score, i, analysis))
        
        # Sort by score and take top max_frames
        scored_analyses.sort(reverse=True)
        selected_indices = sorted([i for _, i, _ in scored_analyses[:max_frames]])
        
        return [analyses[i] for i in selected_indices]

def perform_video_analysis(video_path: str, config: Dict[str, Any], progress_callback: callable = None) -> Dict:
    """Performs a significantly simplified video analysis sequence with real-time frontend progress/log updates."""
    # progress_callback here handles both progress updates and log messages with levels.
    log_callback = progress_callback # Alias for clarity
    try:
        # Initial status via progress_callback
        if log_callback:
            log_callback(0, "Starting video analysis process...")

        # Initialize LlamaEngine (report status), pass log_callback for potential init errors
        try:
            if log_callback:
                log_callback(2, "Initializing LlamaEngine...")
            # Pass log_callback during initialization if LlamaEngine init can report
            llama_engine = LlamaEngine(config)
            if log_callback:
                log_callback(5, "LlamaEngine initialized.", 'success') # Report success level
        except Exception as e:
            error_msg = f"Error initializing LlamaEngine: {str(e)}"
            if log_callback:
                log_callback(100, error_msg, 'error')
            return {"error": error_msg, "analyses": []}

        # Prepare frames directory
        frame_dir = "static/frames"
        os.makedirs(frame_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Error: Could not open video file: {video_path}"
            if log_callback:
                log_callback(100, error_msg, 'error')
            return {"error": error_msg, "analyses": []}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            error_msg = f"Error: No frames found in video file: {video_path}"
            if log_callback:
                log_callback(100, error_msg, 'error')
            return {"error": error_msg, "analyses": []}

        # Extract a very small number of frames for speed
        max_frames = min(5, total_frames) # Analyze only 5 frames
        interval = max(1, total_frames // max_frames)
        frames = []
        frame_count = 0
        count = 0

        if log_callback:
             log_callback(10, f"Extracting {max_frames} frames...")

        # Frame extraction loop (10-50% progress)
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                # Report end of video early via callback
                if log_callback:
                    current_progress = 10 + int((frame_count / max_frames) * 40)
                    log_callback(current_progress, f"Warning: End of video reached early. Extracted {frame_count} frames.", 'warning')
                break

            if count % interval == 0:
                frame_filename = f"frame_{frame_count+1:02d}.jpg"
                frame_path = os.path.join(frame_dir, frame_filename)
                try:
                    frame_resized = cv2.resize(frame, (40, 30))
                    # Use imwrite with error check
                    success = cv2.imwrite(frame_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 10])
                    if not success:
                         raise IOError(f"Could not save frame to {frame_path}") # Raise exception to be caught below

                    frames.append(frame_filename)
                    frame_count += 1
                    # Report extraction progress (10-50%)
                    if log_callback:
                        progress = 10 + int((frame_count / max_frames) * 40)
                        log_callback(progress, f"Extracted frame {frame_count}/{max_frames}")
                except Exception as e:
                     # Report frame save errors via progress_callback
                     current_progress = 10 + int(((frame_count + 1) / max_frames) * 40)
                     error_msg = f"Warning: Could not save frame {frame_count+1}/{max_frames}: {str(e)}"
                     if log_callback:
                          log_callback(current_progress, error_msg, 'warning')
                     pass # Continue if saving fails
            count += 1
        cap.release()

        # Handle case where no frames were extracted
        if not frames: # Use original max_frames for consistent messages
            error_msg = f"Error: No frames extracted for analysis. Expected {max_frames} but got 0."
            if log_callback:
                log_callback(100, error_msg, 'error')
            return {"error": error_msg, "analyses": []}

        # Start analysis reporting (50-95%)
        if log_callback:
             log_callback(50, f"Analyzing {len(frames)} extracted frames...")

        # Frame analysis loop
        analyses = []
        for i, frame_filename in enumerate(frames):
            frame_path = os.path.join(frame_dir, frame_filename)

            # Report analysis progress BEFORE analyzing the frame
            if log_callback:
                 progress = 50 + int((i / len(frames)) * 45)
                 log_callback(progress, f"Analyzing frame {i+1}/{len(frames)}")

            try:
                 # Pass progress_callback (log_callback) to _analyze_frame for error reporting
                 analysis_result = llama_engine._analyze_frame(frame_path, progress_callback=log_callback)
                 if analysis_result and 'error' not in analysis_result:
                     analysis_result['frame_path'] = frame_filename
                     analyses.append(analysis_result)
                 else:
                     # Analysis failed or returned error. Message handled by _analyze_frame or analyze_image.
                     # Ensure progress is updated for this frame even if it failed.
                     current_progress = 50 + int(((i + 1) / len(frames)) * 45)
                     if log_callback:
                          # Send a generic failure message if no specific error/warning was reported by downstream functions
                          if not (analysis_result and analysis_result.get('error')): # Check if _analyze_frame returned an error dict
                               # Check if the analysis result string itself indicates an issue from analyze_image
                              if analysis_result and not (analysis_result.get('analysis') in ['Analysis timeout', 'Analysis error', 'Image processing error', 'Analysis failed']):
                                   # If _analyze_frame didn't return an error and analyze_image didn't give a specific warning,
                                   # then analysis_result is likely None or unexpected. Log a warning.
                                   log_callback(current_progress, f"Warning: Analysis skipped or failed for frame {i+1}/{len(frames)} with no specific error.", 'warning')

                     pass # Message handled by _analyze_frame or analyze_image

            except Exception as e:
                 # This block should ideally not be hit if _analyze_frame catches errors, but for safety:
                 error_msg = f"Critical Error during frame analysis loop for {frame_filename}: {str(e)}"
                 if log_callback:
                      current_progress = 50 + int(((i + 1) / len(frames)) * 45)
                      log_callback(current_progress, error_msg, 'error')
                 pass

        # Handle case where no analysis results were obtained
        if not analyses and frames: # Only error if frames were extracted but none analyzed
            error_msg = "Error: No analysis results obtained from any frame."
            if log_callback:
                log_callback(100, error_msg, 'error')
            return {"error": error_msg, "analyses": []}

        # Final completion status (Progress reaches 100% here)
        if log_callback:
            log_callback(100, "Analysis complete.", 'success')

        # Note: Some low-level library output (e.g., from OpenCV or Llama-cpp) might still appear in the terminal
        # and is outside the control of this progress_callback mechanism.

        # Clean up frames (Optional, based on need to display them later)
        # try:
        #     for frame_filename in frames:
        #         frame_path = os.path.join(frame_dir, frame_filename)
        #         if os.path.exists(frame_path):
        #             os.remove(frame_path)
        # except Exception as cleanup_e:
        #     if log_callback:
        #          log_callback(100, f"Warning: Frame cleanup failed: {cleanup_e}", 'warning')


        return {"analyses": analyses}

    except Exception as e:
        # Catch any unhandled exceptions at the top level
        error_msg = f"Critical Error during analysis process: {str(e)}"
        if log_callback:
            log_callback(100, error_msg, 'error')
        return {"error": error_msg, "analyses": []}
