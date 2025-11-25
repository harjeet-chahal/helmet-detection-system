"""
FastAPI Backend for Helmet Detection System
Real-time video processing with WebSocket streaming
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import cv2
import numpy as np
import json
import asyncio
from datetime import datetime
import io
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.yolo_detector import HelmetDetector, Detection
from detection.tracker import DeepSORTTracker
from risk_assessment.risk_engine import RiskAssessmentEngine, HazardZone, HazardLevel


# Pydantic models for API
class DetectionResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_name: str
    track_id: Optional[int] = None
    risk_score: float = 0.0


class FrameAnalysis(BaseModel):
    frame_id: int
    timestamp: float
    detections: List[DetectionResult]
    violations_count: int
    total_workers: int
    overall_risk: float
    alert_level: str


class ViolationAlert(BaseModel):
    alert_id: str
    timestamp: float
    track_id: int
    violation_duration: float
    risk_score: float
    alert_level: str
    location: Dict[str, int]


class SiteStatistics(BaseModel):
    total_frames_processed: int
    total_violations: int
    active_workers: int
    average_compliance_rate: float
    high_risk_areas: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Helmet Detection API",
    description="Advanced helmet detection system with real-time risk assessment",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state (in production, use Redis or database)
class ApplicationState:
    def __init__(self):
        self.detector: Optional[HelmetDetector] = None
        self.tracker: Optional[DeepSORTTracker] = None
        self.risk_engine: Optional[RiskAssessmentEngine] = None
        self.active_streams: Dict[str, bool] = {}
        self.statistics = {
            'total_frames': 0,
            'total_violations': 0,
            'total_alerts': 0
        }


state = ApplicationState()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Initializing Helmet Detection System...")
    
    # Initialize detector
    state.detector = HelmetDetector(
        model_path='yolov8n.pt',  # Replace with trained model
        conf_threshold=0.5,
        device='auto'
    )
    
    # Initialize tracker
    state.tracker = DeepSORTTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # Initialize risk engine
    state.risk_engine = RiskAssessmentEngine(
        frame_width=1920,
        frame_height=1080,
        fps=30.0
    )
    
    print("System initialized successfully!")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Helmet Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "detect_image": "/api/detect/image",
            "detect_video": "/api/detect/video",
            "websocket": "/ws/stream",
            "statistics": "/api/statistics",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_loaded": state.detector is not None,
        "tracker_loaded": state.tracker is not None,
        "risk_engine_loaded": state.risk_engine is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/detect/image", response_model=FrameAnalysis)
async def detect_image(file: UploadFile = File(...)):
    """
    Detect helmets in a single image
    """
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Detect
    detections = state.detector.detect(frame)
    
    # Convert to response format
    detection_results = [
        DetectionResult(
            bbox=list(d.bbox),
            confidence=d.confidence,
            class_name=d.class_name,
            track_id=d.track_id,
            risk_score=d.risk_score
        )
        for d in detections
    ]
    
    violations_count = sum(1 for d in detections if d.class_name == 'no_helmet')
    total_workers = len(detections)
    
    return FrameAnalysis(
        frame_id=0,
        timestamp=datetime.now().timestamp(),
        detections=detection_results,
        violations_count=violations_count,
        total_workers=total_workers,
        overall_risk=0.0,
        alert_level="info"
    )


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process video file for helmet detection
    Returns job ID for tracking progress
    """
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    # Save uploaded file
    video_path = f"/tmp/upload_{datetime.now().timestamp()}.mp4"
    contents = await file.read()
    
    with open(video_path, 'wb') as f:
        f.write(contents)
    
    # Process video in background
    job_id = f"job_{datetime.now().timestamp()}"
    
    if background_tasks:
        background_tasks.add_task(process_video, video_path, job_id)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video processing started"
    }


async def process_video(video_path: str, job_id: str):
    """Background task to process video"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track
        detections = state.detector.detect(frame)
        tracks = state.tracker.update(detections, frame)
        
        # Assess risk
        frame_results = {
            'frame_id': frame_count,
            'detections': len(detections),
            'violations': sum(1 for d in detections if d.class_name == 'no_helmet')
        }
        
        results.append(frame_results)
        frame_count += 1
    
    cap.release()
    
    # Save results (in production, save to database)
    print(f"Job {job_id} completed: {frame_count} frames processed")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream processing
    Client sends frames, server returns detections
    """
    await websocket.accept()
    
    stream_id = f"stream_{datetime.now().timestamp()}"
    state.active_streams[stream_id] = True
    
    print(f"WebSocket connection established: {stream_id}")
    
    try:
        while state.active_streams.get(stream_id, False):
            # Receive frame from client
            data = await websocket.receive_bytes()
            
            # Decode frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue
            
            # Process frame
            detections = state.detector.detect(frame)
            tracks = state.tracker.update(detections, frame)
            
            # Calculate risk
            site_summary = state.risk_engine.get_site_risk_summary(tracks)
            
            # Update statistics
            state.statistics['total_frames'] += 1
            state.statistics['total_violations'] += site_summary['active_violations']
            
            # Prepare response
            response = {
                'timestamp': datetime.now().isoformat(),
                'frame_id': state.statistics['total_frames'],
                'detections': [
                    {
                        'bbox': list(d.bbox),
                        'class_name': d.class_name,
                        'confidence': float(d.confidence),
                        'track_id': d.track_id if hasattr(d, 'track_id') else None
                    }
                    for d in detections
                ],
                'summary': site_summary,
                'active_tracks': len(tracks)
            }
            
            # Send response
            await websocket.send_json(response)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {stream_id}")
        state.active_streams[stream_id] = False
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        state.active_streams[stream_id] = False


@app.get("/api/statistics", response_model=SiteStatistics)
async def get_statistics():
    """Get overall system statistics"""
    if state.tracker is None:
        raise HTTPException(status_code=503, detail="Tracker not initialized")
    
    summary = state.tracker.get_violation_summary()
    
    return SiteStatistics(
        total_frames_processed=state.statistics['total_frames'],
        total_violations=state.statistics['total_violations'],
        active_workers=summary['total_active_tracks'],
        average_compliance_rate=summary['average_compliance_rate'],
        high_risk_areas=[]  # Populate from risk engine
    )


@app.post("/api/zones/add")
async def add_hazard_zone(
    zone_id: str,
    name: str,
    polygon: List[Dict[str, int]],
    hazard_level: str,
    description: str = ""
):
    """Add a new hazard zone"""
    if state.risk_engine is None:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    try:
        level = HazardLevel[hazard_level.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid hazard level")
    
    zone = HazardZone(
        zone_id=zone_id,
        name=name,
        polygon=[(p['x'], p['y']) for p in polygon],
        hazard_level=level,
        description=description
    )
    
    state.risk_engine.add_hazard_zone(zone)
    
    return {
        "message": "Hazard zone added successfully",
        "zone_id": zone_id
    }


@app.get("/api/zones/list")
async def list_hazard_zones():
    """List all configured hazard zones"""
    if state.risk_engine is None:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    zones = [
        {
            'zone_id': zone.zone_id,
            'name': zone.name,
            'hazard_level': zone.hazard_level.name,
            'description': zone.description
        }
        for zone in state.risk_engine.hazard_zones
    ]
    
    return {"zones": zones}


@app.get("/api/alerts/recent")
async def get_recent_alerts(limit: int = 10):
    """Get recent violation alerts"""
    # In production, fetch from database
    return {
        "alerts": [],
        "message": "Alert system operational"
    }


@app.post("/api/model/reload")
async def reload_model(model_path: str = "yolov8n.pt"):
    """Reload detection model with new weights"""
    try:
        state.detector = HelmetDetector(
            model_path=model_path,
            conf_threshold=0.5,
            device='auto'
        )
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
