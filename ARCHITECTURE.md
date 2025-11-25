# System Architecture Documentation

## Overview

The Helmet Detection System is designed as a scalable, production-ready application with clear separation of concerns and modern MLOps practices.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ RTSP Cameras │  │  Video Files │  │   Webcams    │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 1. YOLOv8 Detection Engine                              │  │
│  │    - Helmet detection                                    │  │
│  │    - No-helmet detection                                 │  │
│  │    - Person detection                                    │  │
│  │    - 45 FPS on GPU / 22 FPS on CPU                       │  │
│  └────────────────┬────────────────────────────────────────┘  │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 2. DeepSORT Tracker                                      │  │
│  │    - Multi-object tracking                               │  │
│  │    - Feature extraction (color + HOG)                    │  │
│  │    - Temporal consistency                                │  │
│  │    - Violation duration tracking                         │  │
│  └────────────────┬────────────────────────────────────────┘  │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 3. Risk Assessment Engine (NOVEL)                        │  │
│  │    ┌─────────────────────────────────────────────────┐  │  │
│  │    │ Risk Factors:                                    │  │  │
│  │    │  • Zone proximity (35% weight)                   │  │  │
│  │    │  • Violation duration (25% weight)               │  │  │
│  │    │  • Worker density (20% weight)                   │  │  │
│  │    │  • Compliance history (20% weight)               │  │  │
│  │    └─────────────────────────────────────────────────┘  │  │
│  │    Output: Risk Score (0-100) + Alert Level             │  │
│  └────────────────┬────────────────────────────────────────┘  │
│                   │                                             │
└───────────────────┼─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API & SERVICES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   FastAPI REST   │  │   WebSocket      │  │   Celery    │ │
│  │                  │  │   Streaming      │  │   Workers   │ │
│  │  • /detect/image │  │                  │  │             │ │
│  │  • /detect/video │  │  • Real-time     │  │  • Async    │ │
│  │  • /statistics   │  │    detections    │  │    tasks    │ │
│  │  • /zones/*      │  │  • Low latency   │  │  • Reports  │ │
│  │  • /alerts       │  │                  │  │  • Cleanup  │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬──────┘ │
│           │                     │                     │         │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐           │
│  │ PostgreSQL +         │  │      Redis           │           │
│  │ TimescaleDB          │  │                      │           │
│  │                      │  │  • Caching           │           │
│  │ • Detections         │  │  • Session store     │           │
│  │ • Violations         │  │  • Task queue        │           │
│  │ • Alerts             │  │  • Real-time data    │           │
│  │ • Reports            │  │                      │           │
│  │ • Time-series opt.   │  │                      │           │
│  └──────────────────────┘  └──────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MONITORING & ALERTS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Prometheus   │  │   Grafana    │  │    Flower    │        │
│  │              │  │              │  │              │        │
│  │ • Metrics    │  │ • Dashboards │  │ • Celery     │        │
│  │ • Alerting   │  │ • Analytics  │  │   monitor    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐         │
│  │           Alert Channels                         │         │
│  │  • Email (SMTP)                                  │         │
│  │  • SMS (Twilio)                                  │         │
│  │  • Dashboard notifications                       │         │
│  │  • Webhooks                                      │         │
│  └──────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Detection Engine

**YOLOv8 Detector** (`src/detection/yolo_detector.py`)

```python
Features:
├── Multi-class detection (helmet/no_helmet/person)
├── Configurable confidence threshold
├── GPU/CPU support with FP16 optimization
├── Batch inference support
├── ONNX/TensorRT export
└── Performance monitoring
```

**Key Methods:**
- `detect(frame)` - Single frame detection
- `detect_batch(frames)` - Batch processing
- `filter_helmet_violations()` - Extract violations
- `associate_person_with_helmet()` - Validation logic

### 2. Tracking System

**DeepSORT Tracker** (`src/detection/tracker.py`)

```python
Features:
├── Multi-object tracking with Kalman filter
├── Feature-based appearance matching
├── Track lifecycle management
├── Violation duration tracking
├── Compliance rate calculation
└── Trajectory visualization
```

**Novel Contributions:**
- Custom feature extractor (color histograms + HOG)
- Temporal violation tracking per worker
- Compliance history maintenance
- Persistent violator identification

### 3. Risk Assessment Engine (NOVEL)

**Multi-Factor Risk Scoring** (`src/risk_assessment/risk_engine.py`)

```python
Risk Factors:
├── Zone Risk (35% weight)
│   ├── Hazard zone proximity
│   ├── Zone hazard level (SAFE → CRITICAL)
│   └── Distance-based decay
│
├── Duration Risk (25% weight)
│   ├── Violation persistence
│   ├── Time-weighted scoring
│   └── Alert escalation
│
├── Density Risk (20% weight)
│   ├── Worker count in scene
│   ├── Incident impact potential
│   └── Crowd safety factor
│
└── History Risk (20% weight)
    ├── Past compliance rate
    ├── Repeat offender detection
    └── Pattern analysis
```

**Output:**
```python
RiskScore {
    total_score: 0-100,
    zone_risk: 0-100,
    duration_risk: 0-100,
    density_risk: 0-100,
    history_risk: 0-100,
    alert_level: INFO | WARNING | CRITICAL | EMERGENCY
}
```

### 4. API Layer

**FastAPI Application** (`src/api/main.py`)

```
Endpoints:
├── GET  /              - API info
├── GET  /health        - Health check
├── POST /api/detect/image
├── POST /api/detect/video
├── GET  /api/statistics
├── POST /api/zones/add
├── GET  /api/zones/list
├── GET  /api/alerts/recent
└── WS   /ws/stream     - Real-time streaming
```

**WebSocket Protocol:**
```
Client → Server: Binary frame data (JPEG encoded)
Server → Client: JSON detection results
{
  "timestamp": "2025-11-24T...",
  "frame_id": 123,
  "detections": [...],
  "summary": {
    "overall_risk": 45.2,
    "alert_level": "warning",
    "active_violations": 2
  }
}
```

### 5. Database Schema

**PostgreSQL Tables:**

```sql
-- Cameras (video sources)
cameras
├── id (PK)
├── camera_id (unique)
├── name
├── location
├── rtsp_url
└── configuration (JSON)

-- Detections (time-series hypertable)
detections
├── id (PK)
├── camera_id (FK)
├── timestamp (indexed)
├── frame_id
├── track_id
├── bbox (x1, y1, x2, y2)
├── class_name
├── confidence
└── risk_score

-- Violations (aggregated)
violations
├── id (PK)
├── violation_id (unique)
├── camera_id (FK)
├── track_id
├── start_time
├── end_time
├── duration_seconds
├── max_risk_score
├── status
└── alert_sent

-- Hazard Zones
hazard_zones
├── id (PK)
├── zone_id (unique)
├── name
├── polygon (JSON)
├── hazard_level
└── risk_multiplier

-- Alerts
alerts
├── id (PK)
├── alert_id (unique)
├── timestamp
├── alert_type
├── alert_level
├── message
└── acknowledged
```

**TimescaleDB Optimization:**
```sql
-- Convert detections to hypertable
SELECT create_hypertable('detections', 'timestamp');

-- Create indexes
CREATE INDEX idx_detections_camera_time 
  ON detections (camera_id, timestamp DESC);
```

## Data Flow

### Real-Time Processing Flow

```
Video Frame
    ↓
[YOLOv8 Detection]
    ↓
Detection Objects
    ↓
[DeepSORT Tracking]
    ↓
Track Objects + IDs
    ↓
[Risk Assessment]
    ↓
Risk Scores + Alerts
    ↓
[Database Write] → PostgreSQL
    ↓
[Alert Generation] → Redis Queue
    ↓
[Celery Worker] → Send Notifications
    ↓
[WebSocket Broadcast] → Dashboard
```

### Video Processing Flow

```
Upload Video
    ↓
[Store in /uploads]
    ↓
[Create Celery Task]
    ↓
[Process Each Frame]
    ├→ Detect
    ├→ Track
    ├→ Assess Risk
    └→ Log to Database
    ↓
[Generate Annotated Video]
    ↓
[Create Metrics Report]
    ↓
[Store in /outputs]
    ↓
Return Job Status
```

## Deployment Architecture

### Docker Compose Stack

```
┌─────────────────────────────────────────┐
│         Docker Host                     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  nginx (reverse proxy)            │ │
│  │  Port: 80, 443                    │ │
│  └─────────────┬─────────────────────┘ │
│                │                         │
│  ┌─────────────▼─────────────────────┐ │
│  │  api (FastAPI)                    │ │
│  │  Port: 8000                       │ │
│  │  GPU: Optional                    │ │
│  └─────────────┬─────────────────────┘ │
│                │                         │
│  ┌─────────────▼─────────────────────┐ │
│  │  celery_worker                    │ │
│  │  Workers: 4                       │ │
│  └─────────────┬─────────────────────┘ │
│                │                         │
│  ┌─────────────┴─────────────────────┐ │
│  │  postgres + timescaledb           │ │
│  │  Port: 5432                       │ │
│  │  Volume: postgres_data            │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  redis                            │ │
│  │  Port: 6379                       │ │
│  │  Volume: redis_data               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  prometheus                       │ │
│  │  Port: 9090                       │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  grafana                          │ │
│  │  Port: 3000                       │ │
│  └───────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

### Production Kubernetes Architecture

```
┌─────────────────────────────────────────────────┐
│              Kubernetes Cluster                 │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Ingress Controller                      │  │
│  │  (NGINX)                                 │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  API Service                             │  │
│  │  Deployment: 3 replicas                  │  │
│  │  HPA: 2-10 pods (CPU > 70%)              │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  Worker Service                          │  │
│  │  Deployment: 5 replicas                  │  │
│  │  GPU NodeSelector: true                  │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  StatefulSet: PostgreSQL                 │  │
│  │  PVC: 100Gi                              │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  StatefulSet: Redis                      │  │
│  │  PVC: 10Gi                               │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Scalability Considerations

### Horizontal Scaling
- **API Servers:** Stateless, can scale to N instances
- **Celery Workers:** Scale based on queue depth
- **Database:** Read replicas for queries

### Vertical Scaling
- **GPU Workers:** RTX 3060 → A100 for higher throughput
- **Database:** Increase connection pool, RAM

### Performance Optimization
1. **Model Optimization:**
   - ONNX Runtime for CPU
   - TensorRT for NVIDIA GPU
   - FP16 quantization

2. **Database Optimization:**
   - TimescaleDB hypertables
   - Partitioning by camera_id
   - Index optimization

3. **Caching Strategy:**
   - Redis for session data
   - Result caching for repeated queries
   - CDN for static assets

## Security Architecture

```
Security Layers:
├── Network Level
│   ├── VPC isolation
│   ├── Security groups
│   └── SSL/TLS encryption
│
├── Application Level
│   ├── JWT authentication
│   ├── Rate limiting
│   ├── Input validation
│   └── CORS policies
│
├── Data Level
│   ├── Database encryption at rest
│   ├── Encrypted backups
│   └── Access control
│
└── Monitoring
    ├── Sentry error tracking
    ├── Audit logs
    └── Intrusion detection
```

## Disaster Recovery

**Backup Strategy:**
- PostgreSQL: Daily automated backups
- Redis: Snapshot every 6 hours
- Model weights: Version control in S3

**Recovery Time Objective (RTO):** < 1 hour
**Recovery Point Objective (RPO):** < 15 minutes

---

**Last Updated:** November 2025
**Version:** 1.0.0
