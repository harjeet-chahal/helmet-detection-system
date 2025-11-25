# Quick Start Guide

Get the Helmet Detection System running in under 10 minutes!

## Option 1: Docker (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- 8GB RAM minimum
- 20GB disk space

### Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd helmet-detection-system
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your preferred settings (optional for demo)
```

3. **Start all services**
```bash
docker-compose up -d
```

4. **Check status**
```bash
docker-compose ps
```

5. **Access the API**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin)
- Flower: http://localhost:5555

6. **Test the system**
```bash
# Health check
curl http://localhost:8000/health

# Upload an image for detection
curl -X POST "http://localhost:8000/api/detect/image" \
  -F "file=@your_image.jpg"
```

That's it! The system is now running.

---

## Option 2: Local Development

### Prerequisites
- Python 3.9-3.11
- PostgreSQL 14+
- Redis 7+

### Steps

1. **Clone and setup virtual environment**
```bash
git clone <your-repo-url>
cd helmet-detection-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start PostgreSQL and Redis**
```bash
# Using Docker
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=helmet_detection \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15

docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit DATABASE_URL and REDIS_URL if needed
```

5. **Initialize database**
```bash
python -m src.database.models
```

6. **Start the API**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

7. **Open in browser**
- http://localhost:8000/docs

---

## Option 3: Quick Demo (No Installation)

Run a demo with synthetic data to see how the system works:

```bash
# Clone the repo
git clone <your-repo-url>
cd helmet-detection-system

# Install minimal dependencies
pip install numpy opencv-python scipy scikit-learn

# Run demo
python scripts/demo_test.py
```

This will test all core components with synthetic data and show you the system capabilities.

---

## Testing the System

### 1. Image Detection

**Using the Web UI:**
1. Go to http://localhost:8000/docs
2. Find `POST /api/detect/image`
3. Click "Try it out"
4. Upload an image
5. Click "Execute"

**Using Python:**
```python
import requests

url = "http://localhost:8000/api/detect/image"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/api/detect/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### 2. Video Processing

**Upload a video:**
```bash
curl -X POST "http://localhost:8000/api/detect/video" \
  -F "file=@construction_video.mp4"
```

**Or use the complete pipeline script:**
```bash
python scripts/process_video.py \
  --video path/to/video.mp4 \
  --output outputs \
  --conf 0.5
```

### 3. Real-time Webcam Detection

```bash
python scripts/process_video.py \
  --realtime \
  --camera 0
```

Press 'q' to quit.

---

## Adding Hazard Zones

Define hazardous areas for risk assessment:

```bash
curl -X POST "http://localhost:8000/api/zones/add" \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "excavation_1",
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

---

## Viewing Results

### Statistics Dashboard
```bash
curl http://localhost:8000/api/statistics
```

### Recent Alerts
```bash
curl http://localhost:8000/api/alerts/recent
```

### Grafana Dashboards
1. Open http://localhost:3000
2. Login: admin/admin
3. View pre-configured dashboards

---

## Stopping the System

### Docker
```bash
docker-compose down
```

### Local Development
```bash
# Stop API (Ctrl+C in terminal)
# Stop Docker services
docker stop postgres redis
```

---

## Next Steps

### For Development:
1. **Train Custom Model** - See `notebooks/02_model_training.ipynb`
2. **Customize Risk Factors** - Edit `src/risk_assessment/risk_engine.py`
3. **Add New API Endpoints** - Extend `src/api/main.py`

### For Production:
1. **Configure Alerts** - Set up email/SMS in `.env`
2. **Deploy to Cloud** - See `SETUP.md` for AWS/GCP/Azure
3. **Enable Monitoring** - Configure Prometheus alerts
4. **Set up SSL** - Add certificates to Nginx

### For Research:
1. **Experiment with Models** - Try different YOLO versions
2. **Tune Risk Weights** - Optimize risk assessment formula
3. **Add Features** - Implement behavior analysis, pose estimation

---

## Common Issues

### Port Already in Use
```bash
# Change port in docker-compose.yml or .env
API_PORT=8001
```

### CUDA Out of Memory
```bash
# Use CPU mode
export DEVICE=cpu

# Or reduce batch size
python scripts/process_video.py --batch 8
```

### Database Connection Failed
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Restart if needed
docker-compose restart postgres
```

### Model Not Found
```bash
# Download YOLOv8 base model
mkdir -p data/models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt \
  -O data/models/yolov8n.pt
```

---

## Getting Help

- **Documentation**: See [README.md](README.md), [SETUP.md](SETUP.md), [ARCHITECTURE.md](ARCHITECTURE.md)
- **Issues**: Open an issue on GitHub
- **Questions**: Contact [your-email]

---

## Example Results

After running the system, you'll find:

```
outputs/
├── video_annotated.mp4          # Processed video with detections
├── video_metrics.json           # Performance metrics
└── reports/
    └── daily_compliance.pdf     # Generated reports
```

**Sample Metrics JSON:**
```json
{
  "video_info": {
    "frames_processed": 1000,
    "duration_seconds": 33.3
  },
  "detection_stats": {
    "total_violations": 15,
    "unique_violators": 5,
    "compliance_rate": 0.85
  },
  "risk_assessment": {
    "total_alerts": 3,
    "high_risk_incidents": 1
  },
  "performance": {
    "avg_fps": 30.5,
    "avg_inference_time_ms": 32.8
  }
}
```

---

**Happy detecting! 🎉**

For questions or issues, please refer to the full documentation or reach out to the team.
