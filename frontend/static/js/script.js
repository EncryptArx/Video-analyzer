document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const selectButton = document.getElementById('selectButton');
    const analyzeButton = document.getElementById('analyzeButton');
    const uploadStatus = document.getElementById('uploadStatus');
    const analysisSection = document.querySelector('.analysis-section');
    const analysisContentDiv = document.getElementById('analysisContent');
    const progressBar = document.getElementById('progressBar');
    const uploadProgress = document.getElementById('uploadProgress');
    const statusMessage = document.getElementById('statusMessage');
    const videoPreview = document.getElementById('videoPreview');
    const dropZone = document.getElementById('dropZone');
    const logOutput = document.getElementById('logOutput');
    const logContainer = document.querySelector('.log-container');
    
    // Metrics elements
    const durationMetric = document.getElementById('durationMetric');
    const framesMetric = document.getElementById('framesMetric');
    const objectsMetric = document.getElementById('objectsMetric');
    const emotionsMetric = document.getElementById('emotionsMetric');

    let currentAnalysisId = null;
    let uploadedFilename = null;
    let eventSource = null;
    let statusCheckInterval;

    // Set default placeholders for status and log
    statusMessage.textContent = 'Waiting to start analysis...';
    statusMessage.className = 'status-message';
    logOutput.innerHTML = '<div class="log-entry info"><span class="timestamp">[System]</span> <span class="message">Ready to process video. Click "Start Analysis" to begin.</span></div>';

    // Function to add a log entry with improved performance and auto-scrolling
    function addLogEntry(message, type = 'info', timestamp = null) {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        const ts = timestamp || new Date().toLocaleTimeString();
        logEntry.innerHTML = `<span class="timestamp">[${ts}]</span> <span class="message">${message}</span>`;
        logOutput.appendChild(logEntry);

        // Auto-scroll the log container to the latest entry
        if (logContainer) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }

    // Function to update progress bar
    function updateProgress(progress, message) {
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
    }

    // Function to update metrics
    function updateMetrics(data) {
        if (data.duration) durationMetric.textContent = formatDuration(data.duration);
        if (data.frames) framesMetric.textContent = data.frames;
        if (data.objects) objectsMetric.textContent = data.objects;
        if (data.emotions) emotionsMetric.textContent = data.emotions;
    }

    // Function to format duration
    function formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Function to display the analysis results
    function displayAnalysisResults(analysisData) {
        if (!analysisData) return;
        
        // Clear previous content
        analysisContentDiv.innerHTML = '';
        
        // Create analysis results container
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'analysis-results';
        
        // Add analysis data
        const analysisJson = JSON.stringify(analysisData, null, 2);
        const pre = document.createElement('pre');
        pre.textContent = analysisJson;
        resultsContainer.appendChild(pre);
        
        // Add to page
        analysisContentDiv.appendChild(resultsContainer);
    }

    // Function to handle file upload
    async function handleFileUpload(file) {
        try {
            // Show loading state
            uploadStatus.textContent = 'Uploading...';
            uploadStatus.style.color = 'blue';
            uploadProgress.style.width = '0%';
            addLogEntry('Starting video upload...');
            
            // Create FormData
            const formData = new FormData();
            formData.append('video', file);

            // Upload file
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            
            if (data.success) {
                uploadedFilename = data.filename;
                uploadStatus.textContent = 'Upload successful!';
                uploadStatus.style.color = 'green';
                uploadProgress.style.width = '100%';
                analyzeButton.style.display = 'block';
                addLogEntry('Video upload completed successfully', 'success');

                // Show video preview
                videoPreview.src = URL.createObjectURL(file);
                videoPreview.style.display = 'block';
                addLogEntry('Video preview generated');
            }
        } catch (error) {
            console.error('Error:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.style.color = 'red';
            addLogEntry(`Error: ${error.message}`, 'error');
        }
    }

    // Function to handle analysis start
    function handleAnalysisStart() {
        if (!uploadedFilename) {
            addLogEntry('No video file selected for analysis', 'error');
            return;
        }

        // Clear previous logs and results
        logOutput.innerHTML = '';
        document.getElementById('analysisContent').innerHTML = '';
        
        // Disable analyze button and show loading state
        analyzeButton.disabled = true;
        analyzeButton.classList.add('loading');
        statusMessage.textContent = 'Starting analysis...';
        statusMessage.className = 'status-message processing';
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        addLogEntry('Starting video analysis...');
        
        // Clear any existing polling
        isPolling = false;
        if (currentPollingTimeout) {
            clearTimeout(currentPollingTimeout);
            currentPollingTimeout = null;
        }
        
        // Start analysis
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename: uploadedFilename })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Analysis request failed');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            addLogEntry('Analysis request accepted, starting processing...');
            // Start checking status
            isPolling = true;
            checkStatus();
        })
        .catch(error => {
            console.error('Analysis error:', error);
            isPolling = false;
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.className = 'status-message error';
            analyzeButton.disabled = false;
            analyzeButton.classList.remove('loading');
            addLogEntry(`Error: ${error.message}`, 'error');
        });
    }

    let isPolling = false;
    let currentPollingTimeout = null;

    function checkStatus() {
        if (!isPolling) return;
        
        fetch('/analysis_status')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (!data) {
                    throw new Error('No data received from server');
                }

                // Update progress
                const progress = data.progress || 0;
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${progress}%`;
                statusMessage.textContent = data.message || `Progress: ${progress}%`;

                // Update log
                if (data.logs && Array.isArray(data.logs)) {
                    // Only update logs if they've changed to prevent flickering
                    const currentLogs = Array.from(logOutput.children).map(div => div.textContent);
                    const newLogs = data.logs.map(log => 
                        `[${log.timestamp || new Date().toISOString()}] ${log.message}`
                    );
                    
                    if (JSON.stringify(currentLogs) !== JSON.stringify(newLogs)) {
                        logOutput.innerHTML = '';
                        data.logs.forEach(log => {
                            addLogEntry(
                                log.message, 
                                log.level || 'info', 
                                log.timestamp
                            );
                        });
                    }
                }
                
                // Handle different statuses
                switch (data.status) {
                    case 'complete':
                        handleAnalysisComplete(data);
                        return;
                        
                    case 'error':
                        throw new Error(data.error || data.message || 'Analysis failed');
                        
                    case 'processing':
                        // Continue polling after a delay
                        currentPollingTimeout = setTimeout(checkStatus, 1000);
                        break;
                        
                    default:
                        // If we get an unknown status, log it but keep polling
                        console.warn('Unknown status:', data.status);
                        currentPollingTimeout = setTimeout(checkStatus, 1000);
                        break;
                }
            })
            .catch(error => {
                console.error('Status check error:', error);
                handleAnalysisError(error);
            });
    }
    
    function handleAnalysisComplete(data) {
        isPolling = false;
        clearTimeout(currentPollingTimeout);
        currentPollingTimeout = null;
        
        statusMessage.textContent = 'Analysis complete!';
        statusMessage.className = 'status-message success';
        analyzeButton.disabled = false;
        analyzeButton.classList.remove('loading');
        
        if (data.results) {
            displayResults(data.results);
            
            // Add download report button if report is available
            if (data.results.report && (data.results.report.download_url || data.results.report.filename)) {
                // Remove any existing download button
                const existingBtn = document.querySelector('.download-report-btn');
                if (existingBtn) {
                    existingBtn.remove();
                }
                
                const downloadBtn = document.createElement('button');
                downloadBtn.className = 'button download-report-btn';
                downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Report (PDF)';
                
                // Use the server-provided download URL
                const downloadUrl = data.results.report.download_url || 
                                 `/download_report/${encodeURIComponent(data.results.report.filename)}`;
                
                downloadBtn.onclick = () => {
                    // Open in a new tab to avoid leaving the page
                    window.open(downloadUrl, '_blank');
                };
                
                // Add some styling to the button
                downloadBtn.style.marginTop = '20px';
                downloadBtn.style.padding = '10px 20px';
                downloadBtn.style.fontSize = '16px';
                
                const resultsSection = document.querySelector('.results-section');
                if (resultsSection) {
                    // Insert the download button at the top of the results section
                    resultsSection.insertBefore(downloadBtn, resultsSection.firstChild);
                } else {
                    // If results section doesn't exist, add it to the analysis section
                    analysisSection.appendChild(downloadBtn);
                }
                
                // Also log the availability of the report
                addLogEntry('Analysis completed successfully - Report is ready for download', 'success');
            } else {
                addLogEntry('Analysis completed successfully - Results are available in the interface', 'success');
            }
        }
    }
    
    function handleAnalysisError(error) {
        isPolling = false;
        clearTimeout(currentPollingTimeout);
        currentPollingTimeout = null;
        
        const errorMessage = error.message || 'An unknown error occurred';
        console.error('Analysis error:', errorMessage);
        
        statusMessage.textContent = `Error: ${errorMessage}`;
        statusMessage.className = 'status-message error';
        analyzeButton.disabled = false;
        analyzeButton.classList.remove('loading');
        addLogEntry(`Error: ${errorMessage}`, 'error');
    }

    function displayResults(results) {
        analysisContentDiv.innerHTML = '';
        
        // Check if we have a report to display
        if (results.report && results.report.content) {
            // Create a container for the report
            const reportContainer = document.createElement('div');
            reportContainer.className = 'report-container';
            
            // Add the report content as pre-formatted text
            const reportContent = document.createElement('pre');
            reportContent.className = 'report-content';
            reportContent.textContent = results.report.content;
            
            // Add download button for the report
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'button download-report-btn';
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Report (PDF)';
            downloadBtn.onclick = () => {
                // Use the server-provided download URL
                const downloadUrl = results.report.download_url || 
                                 `/download_report/${encodeURIComponent(results.report.filename)}`;
                window.open(downloadUrl, '_blank');
            };
            
            // Style the download button
            downloadBtn.style.margin = '20px 0';
            downloadBtn.style.padding = '10px 20px';
            downloadBtn.style.fontSize = '16px';
            
            // Add elements to the container
            reportContainer.appendChild(downloadBtn);
            reportContainer.appendChild(document.createElement('hr'));
            reportContainer.appendChild(reportContent);
            
            // Add the report container to the page
            analysisContentDiv.appendChild(reportContainer);
        } 
        // Fallback to showing analysis results if no report is available
        else if (results.analyses && results.analyses.length > 0) {
            const resultsContainer = document.createElement('div');
            resultsContainer.className = 'results-container';
            
            results.analyses.forEach((analysis, index) => {
                const resultCard = document.createElement('div');
                resultCard.className = 'result-card';
                
                // Create frame preview if available
                if (analysis.frame_path) {
                    const framePreview = document.createElement('img');
                    framePreview.src = `/frames/${analysis.frame_path}`;
                    framePreview.alt = `Frame ${index + 1}`;
                    framePreview.className = 'frame-preview';
                    resultCard.appendChild(framePreview);
                }
                
                // Create analysis content
                const content = document.createElement('div');
                content.className = 'analysis-content';
                
                // Add timestamp if available
                if (analysis.timestamp) {
                    const timestamp = document.createElement('div');
                    timestamp.className = 'timestamp';
                    timestamp.textContent = new Date(analysis.timestamp * 1000).toLocaleString();
                    content.appendChild(timestamp);
                }
                
                // Add analysis text
                if (analysis.analysis) {
                    const analysisText = document.createElement('p');
                    analysisText.className = 'analysis-text';
                    analysisText.textContent = analysis.analysis;
                    content.appendChild(analysisText);
                }
                
                // Add error message if present
                if (analysis.error) {
                    const errorMessage = document.createElement('p');
                    errorMessage.className = 'error-message';
                    errorMessage.textContent = `Error: ${analysis.error}`;
                    content.appendChild(errorMessage);
                }
                
                resultCard.appendChild(content);
                resultsContainer.appendChild(resultCard);
            });
            
            analysisContentDiv.appendChild(resultsContainer);
        } 
        // If no report or analysis results are available
        else {
            analysisContentDiv.innerHTML = '<p class="no-results">No report or analysis results available.</p>';
        }
    }

    // Helper functions for drag and drop
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropZone.classList.add('highlight');
    }

    function unhighlight() {
        dropZone.classList.remove('highlight');
    }

    // Add event listeners
    selectButton.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    analyzeButton.addEventListener('click', handleAnalysisStart);

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Initial state setup
    analyzeButton.style.display = 'none';
    analysisSection.style.display = 'none';
    videoPreview.style.display = 'none';
});