# Video Analyzer

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/EncryptArx/Video-analyzer.git)

A comprehensive video analysis tool that leverages advanced AI models (YOLO, LLaMA) to process videos, extract frames, analyze content, and generate detailed reports. The project features a Python backend with Flask and a web-based frontend for easy interaction.

---

## Features
- **Video Upload & Processing**: Upload videos through web interface
- **Frame Extraction**: Automatic frame extraction for analysis
- **Object Detection**: YOLO-based object detection and classification
- **Pose Estimation**: Human pose detection and analysis
- **Emotion Recognition**: Facial emotion analysis using ResNet-50
- **AI-Powered Reports**: LLaMA-7B powered natural language report generation
- **PDF Reports**: Downloadable detailed analysis reports
- **Real-time Progress**: Live progress tracking during analysis
- **Web Interface**: Modern, responsive web UI

---

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: At least 10GB free space for models and temporary files
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)
- **CPU**: Multi-core processor recommended

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, macOS, or Linux
- **Git**: For cloning the repository

---

## Required Models

### Download Required Model Files
Before running the application, you need to download the following model files:

1. **LLaMA-7B Model** (Required for report generation):
   - Download: `llama-7b.Q4_K_M.gguf`
   - Size: ~4GB
   - Place in: `backend/models/`
   - **Download Sources**:
     - **Primary**: [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
     - **Alternative**: [Google Drive](https://drive.google.com/file/d/1e5YvbUykVnlb9YdxAY4j0q7MOgaXPCbY/view?usp=sharing)
     - **GitHub**: [LLaMA.cpp releases](https://github.com/ggerganov/llama.cpp/releases)

2. **YOLO Models** (Required for object detection and pose estimation):
   - `yolov8n.pt` (Object detection) - ~6MB
   - `yolov8n-pose.pt` (Pose estimation) - ~6MB
   - Place in: `backend/` directory
   - Source: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

### Model File Structure
```
backend/
├── models/
│   └── llama-7b.Q4_K_M.gguf    # LLaMA model for report generation
├── yolov8n.pt                  # YOLO object detection model
└── yolov8n-pose.pt             # YOLO pose estimation model
```

---

## Dependencies

### Python Packages
The following packages are automatically installed via `requirements.txt`:

```txt
flask                    # Web framework
opencv-python           # Computer vision
ultralytics             # YOLO models
numpy                   # Numerical computing
pillow                  # Image processing
torch                   # PyTorch (ML framework)
torchvision             # PyTorch vision
torchaudio              # PyTorch audio
llama-cpp-python        # LLaMA model inference
deepface                # Face analysis
tensorflow              # ML framework
scipy                   # Scientific computing
matplotlib              # Plotting
seaborn                 # Statistical visualization
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/EncryptArx/Video-analyzer.git
cd Video-analyzer
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Required Models
```bash
# Create models directory
mkdir -p models

# Download LLaMA model (4GB)
# Option 1: Using wget/curl from Hugging Face
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf -O models/llama-7b.Q4_K_M.gguf

# Option 2: Manual download from Hugging Face
# Visit: https://huggingface.co/TheBloke/Llama-2-7B-GGUF
# Download: llama-2-7b.Q4_K_M.gguf
# Rename to: llama-7b.Q4_K_M.gguf
# Place in: backend/models/

# Option 3: Download from Google Drive (if Hugging Face is unavailable)
# Visit: https://drive.google.com/file/d/1e5YvbUykVnlb9YdxAY4j0q7MOgaXPCbY/view?usp=sharing
# Download the file and rename to: llama-7b.Q4_K_M.gguf
# Place in: backend/models/

# Download YOLO models (will be auto-downloaded on first run)
# The models will be automatically downloaded when you first run the application
```

### 4. Configuration
The application uses `config.json` for configuration:
```json
{
  "model_path": "models/llama-7b.Q4_K_M.gguf",
  "detection_model": "yolov8n.pt",
  "face_model": "emotion",
  "pose_model": "yolov8n-pose.pt",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.4
}
```

### 5. Running the Application
```bash
# Start the Flask server
python app.py

# Or if you have a main.py entry point
python main.py
```

### 6. Access the Application
- Open your web browser
- Navigate to: `http://localhost:5000`
- Upload a video file and start analysis

---

## Project Structure
```
video_analyzer/
├── backend/
│   ├── app.py                 # Flask web application
│   ├── main.py                # Alternative entry point
│   ├── config.json            # Configuration file
│   ├── requirements.txt       # Python dependencies
│   ├── video_analyzer.py      # Core video analysis logic
│   ├── llama_engine.py        # LLaMA model interface
│   ├── video_utils.py         # Video processing utilities
│   ├── report_utils.py        # Report generation utilities
│   ├── create_favicon.py      # Favicon creation script
│   ├── models/                # AI models directory
│   │   └── llama-7b.Q4_K_M.gguf
│   ├── frames/                # Extracted video frames (auto-created)
│   ├── uploads/               # Uploaded videos (auto-created)
│   ├── reports/               # Generated reports (auto-created)
│   │   └── report templates/  # Report templates
│   ├── yolov8n.pt            # YOLO object detection model
│   └── yolov8n-pose.pt       # YOLO pose estimation model
├── frontend/
│   ├── templates/
│   │   └── index.html        # Main web interface
│   └── static/
│       ├── css/
│       │   └── styles.css    # Styling
│       ├── js/
│       │   └── script.js     # Frontend JavaScript
│       └── favicon.ico       # Website icon
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## Usage

### 1. Upload Video
- Click "Choose File" and select a video file
- Supported formats: MP4, AVI, MOV, MKV
- Maximum file size: 500MB (configurable)

### 2. Start Analysis
- Click "Analyze Video" to begin processing
- Monitor real-time progress in the interface
- Analysis includes:
  - Object detection
  - Pose estimation
  - Emotion recognition
  - Frame extraction

### 3. View Results
- Analysis results are displayed in real-time
- Download PDF report when complete
- View detailed statistics and insights

---

## API Endpoints

### Web Interface
- `GET /` - Main application page
- `POST /upload` - Upload video file
- `POST /analyze` - Start video analysis
- `GET /analysis_status` - Get analysis progress
- `GET /download_report/<filename>` - Download generated report

---

## Troubleshooting

### Common Issues

1. **Model Download Errors**
   ```bash
   # If YOLO models fail to download automatically:
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

2. **LLaMA Model Download Issues**
   - If Hugging Face is unavailable, use the [Google Drive link](https://drive.google.com/file/d/1e5YvbUykVnlb9YdxAY4j0q7MOgaXPCbY/view?usp=sharing)
   - Ensure the file is named exactly: `llama-7b.Q4_K_M.gguf`
   - Place in: `backend/models/` directory

3. **CUDA/GPU Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install CPU-only version if needed
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Memory Issues**
   - Reduce video resolution before upload
   - Close other applications
   - Increase system swap space

5. **Port Already in Use**
   ```bash
   # Change port in app.py
   app.run(host='0.0.0.0', port=5001, debug=True)
   ```

### Performance Optimization
- Use GPU for faster processing
- Process shorter videos for testing
- Adjust confidence thresholds in `config.json`
- Monitor system resources during analysis

---

## Development

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Structure
- `video_analyzer.py`: Core analysis logic
- `llama_engine.py`: LLaMA model interface
- `app.py`: Flask web application
- `report_utils.py`: Report generation

---

## License
This project is licensed under the Apache-2.0 License. See the [LICENSE](https://github.com/EncryptArx/Video-analyzer/blob/main/LICENSE) file for details.

---

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Support
- **Issues**: [GitHub Issues](https://github.com/EncryptArx/Video-analyzer/issues)
- **Documentation**: This README and inline code comments
- **Community**: GitHub Discussions (if enabled)

---

## Repository
[https://github.com/EncryptArx/Video-analyzer.git](https://github.com/EncryptArx/Video-analyzer.git)

---

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) for LLaMA model inference
- [Flask](https://flask.palletsprojects.com/) for web framework
- [OpenCV](https://opencv.org/) for computer vision 