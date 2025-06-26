import argparse
import os
import tempfile
import cv2
from datetime import datetime
from tkinter import Tk, filedialog
# from llama_engine import LlamaEngine # Removed since LlamaEngine is used internally by perform_video_analysis
import logging
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json # Import json to handle structured results
import time
# from video_analyzer import perform_video_analysis # Removed duplicate import if video_analyzer is llama_engine

# Removing the logging basicConfig from here to avoid interference with app.py's logging setup.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_video_file() -> str:
    """Open file dialog to select video file."""
    # This function seems unused by the Flask app, but keeping it if it has other purposes.
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path if file_path and os.path.exists(file_path) else None

def extract_frames(video_path: str, output_dir: str, max_frames: int = 30) -> list:
    """Extract frames from video file with optimized sampling."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Using print instead of logging here, assuming this function might be called outside Flask context
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: No frames found in video file: {video_path}", file=sys.stderr)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate optimal frame sampling interval
    interval = max(1, total_frames // max_frames)

    frames = []
    count = 0
    frame_count = 0

    # Set video capture properties for faster reading
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {count} from video {video_path}", file=sys.stderr)
            break

        if count % interval == 0:
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            try:
                # Resize frame for analysis if needed (optional, depends on model input size)
                # frame = cv2.resize(frame, (640, 480))
                success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                     print(f"Warning: Could not save frame {frame_count} to {frame_path}", file=sys.stderr)
                     continue
                frames.append({'path': frame_path, 'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0})
                frame_count += 1
            except Exception as e:
                 print(f"Error saving frame {frame_count} to {frame_path}: {e}", file=sys.stderr)
                 continue

        count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}", file=sys.stderr)
    return frames

def get_user_instruction() -> str:
    """Prompt user for analysis instructions."""
    # This function seems unused by the Flask app, keeping it if needed elsewhere.
    print("\nEnter specific instructions (or press Enter for default analysis):")
    return input(">> ").strip() or None

def save_report(report_content: str, video_name: str) -> str:
    """Save the report to a file with timestamp and return the file path."""
    # This function seems unused by the Flask app analysis flow, keeping it if needed elsewhere.
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("reports", f"analysis_report_{os.path.splitext(video_name)[0]}_{timestamp}.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_path

# Removed the analyze_video_backend function from main.py
# This function contained logging that conflicted with the Flask app's logging.
# The analysis logic is now handled by perform_video_analysis in llama_engine.py
# and orchestrated by the background thread in app.py.

# Removed the main command-line entry point function from main.py
# This function contained print and logging calls that caused unwanted terminal output
# when running the Flask app, as it was likely being executed in some environments.
# The Flask app in app.py provides the web interface entry point.

# Removed the __main__ block that calls main()
# This prevents the command-line interface from running automatically
# when the module is imported or run directly.

# Keeping other potentially useful functions/classes if they exist outside the removed parts
# Assuming any other code in main.py (like utility functions or class definitions) is kept if present.

# Note: This edit focuses on removing the duplicate analysis logic and command-line interface
# that caused the conflicting terminal output. The core analysis logic should now be
# driven solely by app.py calling llama_engine.py.

# If other functions/classes were present in the original main.py and are needed elsewhere,
# they would be preserved, but the core analysis and main execution block are removed.
