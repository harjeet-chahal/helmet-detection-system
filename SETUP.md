# Setup Guide - Helmet Detection System

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Model Training](#model-training)
5. [API Usage](#api-usage)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- **Minimum (CPU only):**
  - 8GB RAM
  - 4 CPU cores
  - 20GB disk space

- **Recommended (GPU):**
  - 16GB RAM
  - NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
  - CUDA 11.8+
  - cuDNN 8.6+
  - 50GB disk space

### Software Requirements
- Python 3.9-3.11
- Docker & Docker Compose (for containerized deployment)
- Git
- PostgreSQL 14+ (or use Docker)
- Redis 7+ (or use Docker)

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/helmet-detection-system.git
cd helmet-detection-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For GPU support (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**.env Configuration:**
```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=helmet_detection
DATABASE_URL=postgresql://postgres:your_secure_password@localhost:5432/helmet_detection

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your_secret_key_here

# Model
MODEL_PATH=data/models/helmet_detector_best.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# Alerts
ENABLE_EMAIL_ALERTS=false
ENABLE_SMS_ALERTS=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Logging
LOG_LEVEL=INFO
```

### 5. Database Setup
```bash
# Start PostgreSQL (if using Docker)
docker run -d \
  --name helmet_detection_db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=helmet_detection \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15

# Run migrations
python -m src.database.models

# Or use Alembic for migrations
alembic upgrade head
```

### 6. Download Pre-trained Model (Optional)
```bash
# Create models directory
mkdir -p data/models

# Download YOLOv8 base model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt \
  -O data/models/yolov8n.pt
```

### 7. Start Development Server
```bash
# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Celery worker (optional)
celery -A src.tasks.celery_app worker --loglevel=info

# In another terminal, start Celery beat (optional)
celery -A src.tasks.celery_app beat --loglevel=info
```

### 8. Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# Or visit in browser
open http://localhost:8000/docs
```

## Docker Deployment

### 1. Build and Start Services
```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 2. Initialize Database
```bash
# Run database migrations
docker-compose exec api python -m src.database.models

# Or connect to database directly
docker-compose exec postgres psql -U postgres -d helmet_detection
```

### 3. Access Services
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Flower (Celery Monitor)**: http://localhost:5555
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 4. Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Model Training

### 1. Prepare Dataset
```bash
# Download datasets
# Option 1: Kaggle Safety Helmet Detection Dataset
kaggle datasets download andrewmvd/hard-hat-detection
unzip hard-hat-detection.zip -d data/raw/

# Option 2: Roboflow (requires account)
# Download from: https://roboflow.com/

# Organize dataset structure
python scripts/prepare_dataset.py --input data/raw --output data/processed
```

### 2. Train Model
```bash
# Option 1: Using Jupyter notebook
jupyter notebook notebooks/02_model_training.ipynb

# Option 2: Using Python script
python scripts/train_model.py \
  --data data/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640

# Option 3: Using CLI
yolo train \
  model=yolov8n.pt \
  data=data/dataset.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs \
  name=helmet_detection
```

### 3. Evaluate Model
```bash
# Validate on test set
python scripts/evaluate_model.py \
  --model data/models/helmet_detector_best.pt \
  --data data/dataset.yaml

# Or use YOLO CLI
yolo val \
  model=data/models/helmet_detector_best.pt \
  data=data/dataset.yaml
```

### 4. Export Model
```bash
# Export to ONNX
python scripts/export_model.py \
  --model data/models/helmet_detector_best.pt \
  --format onnx

# Export to TensorRT (requires NVIDIA GPU)
python scripts/export_model.py \
  --model data/models/helmet_detector_best.pt \
  --format engine \
  --half
```

## API Usage

### 1. Image Detection
```bash
# Using cURL
curl -X POST "http://localhost:8000/api/detect/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Using Python
import requests

url = "http://localhost:8000/api/detect/image"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### 2. Video Processing
```bash
# Upload video for processing
curl -X POST "http://localhost:8000/api/detect/video" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_video.mp4"

# Check processing status
curl "http://localhost:8000/api/jobs/{job_id}"
```

### 3. WebSocket Streaming
```python
import asyncio
import websockets
import cv2
import numpy as np

async def stream_video():
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # Webcam
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Send frame
            await websocket.send(buffer.tobytes())
            
            # Receive detections
            response = await websocket.recv()
            print(response)
            
            await asyncio.sleep(0.033)  # ~30 FPS
        
        cap.release()

asyncio.run(stream_video())
```

### 4. Add Hazard Zone
```bash
curl -X POST "http://localhost:8000/api/zones/add" \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "zone_1",
    "name": "Excavation Area",
    "polygon": [
      {"x": 100, "y": 100},
      {"x": 500, "y": 100},
      {"x": 500, "y": 400},
      {"x": 100, "y": 400}
    ],
    "hazard_level": "CRITICAL",
    "description": "Deep excavation zone"
  }'
```

## Production Deployment

### 1. AWS Deployment
```bash
# Install AWS CLI
pip install awscli

# Configure AWS
aws configure

# Deploy using ECS/Fargate
aws ecs create-cluster --cluster-name helmet-detection

# Or use Elastic Beanstalk
eb init -p docker helmet-detection-system
eb create helmet-detection-prod
```

### 2. Google Cloud Platform
```bash
# Install gcloud CLI
# Follow: https://cloud.google.com/sdk/docs/install

# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/helmet-detection

# Deploy to Cloud Run
gcloud run deploy helmet-detection \
  --image gcr.io/PROJECT_ID/helmet-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 3. Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -n helmet-detection

# View logs
kubectl logs -f deployment/helmet-detection-api -n helmet-detection
```

### 4. Edge Deployment (NVIDIA Jetson)
```bash
# Install JetPack SDK on Jetson device
# Follow: https://developer.nvidia.com/embedded/jetpack

# Export model to TensorRT
python scripts/export_model.py \
  --model data/models/helmet_detector_best.pt \
  --format engine \
  --device 0

# Deploy on Jetson
scp -r src/ jetson@192.168.1.100:/home/jetson/helmet-detection/
ssh jetson@192.168.1.100
cd helmet-detection
python src/api/main.py
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train_model.py --batch 8  # Instead of 16

# Use gradient accumulation
python scripts/train_model.py --accumulate 2
```

#### 2. Model Loading Fails
```bash
# Check model path
ls -la data/models/

# Re-download base model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt \
  -O data/models/yolov8n.pt
```

#### 3. Database Connection Error
```bash
# Check PostgreSQL status
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

#### 4. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in .env
API_PORT=8001
```

#### 5. Permission Denied
```bash
# Fix file permissions
chmod -R 755 src/
chmod -R 777 data/

# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock
```

### Performance Optimization

#### 1. Enable GPU Acceleration
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

#### 2. Optimize Inference
```bash
# Use TensorRT for NVIDIA GPUs
python scripts/export_model.py --format engine

# Use ONNX Runtime
pip install onnxruntime-gpu
```

#### 3. Database Indexing
```sql
-- Connect to PostgreSQL
psql -U postgres -d helmet_detection

-- Create indexes
CREATE INDEX idx_detections_timestamp ON detections(timestamp DESC);
CREATE INDEX idx_detections_camera_time ON detections(camera_id, timestamp DESC);
CREATE INDEX idx_violations_status ON violations(status);
```

## Getting Help

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/helmet-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/helmet-detection-system/discussions)

## License

MIT License - see [LICENSE](LICENSE) for details
