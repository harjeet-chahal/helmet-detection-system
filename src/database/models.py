"""
Database Models for Helmet Detection System
Using PostgreSQL with TimescaleDB for time-series optimization
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from typing import Optional
import os


Base = declarative_base()


class Camera(Base):
    """Camera/video source configuration"""
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    location = Column(String(500))
    rtsp_url = Column(String(500))
    status = Column(String(20), default='active')  # active, inactive, maintenance
    fps = Column(Float, default=30.0)
    resolution = Column(String(20))  # e.g., "1920x1080"
    
    # Configuration
    conf_threshold = Column(Float, default=0.5)
    enable_tracking = Column(Boolean, default=True)
    enable_risk_assessment = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="camera", cascade="all, delete-orphan")
    violations = relationship("Violation", back_populates="camera", cascade="all, delete-orphan")


class Detection(Base):
    """Individual detection record (converted to time-series with TimescaleDB)"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey('cameras.camera_id'), nullable=False, index=True)
    
    # Detection details
    timestamp = Column(DateTime, nullable=False, index=True)
    frame_id = Column(Integer, nullable=False)
    track_id = Column(Integer, nullable=True, index=True)
    
    # Bounding box
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    
    # Classification
    class_name = Column(String(50), nullable=False, index=True)  # helmet, no_helmet, person
    confidence = Column(Float, nullable=False)
    
    # Risk assessment
    risk_score = Column(Float, default=0.0)
    alert_level = Column(String(20), default='info')  # info, warning, critical, emergency
    zone_id = Column(String(50), nullable=True)
    
    # Relationships
    camera = relationship("Camera", back_populates="detections")
    
    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
            'frame_id': self.frame_id,
            'track_id': self.track_id,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'class_name': self.class_name,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'alert_level': self.alert_level,
            'zone_id': self.zone_id
        }


class Violation(Base):
    """Violation records with temporal tracking"""
    __tablename__ = 'violations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(String(100), unique=True, nullable=False, index=True)
    camera_id = Column(String(50), ForeignKey('cameras.camera_id'), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    
    # Timing
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    
    # Status
    status = Column(String(20), default='active')  # active, resolved, escalated
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    
    # Risk assessment
    max_risk_score = Column(Float, default=0.0)
    avg_risk_score = Column(Float, default=0.0)
    alert_level = Column(String(20), default='info')
    
    # Location
    zone_id = Column(String(50), nullable=True)
    location = Column(JSON, nullable=True)  # Store bbox or polygon
    
    # Worker information
    worker_id = Column(String(100), nullable=True)
    
    # Actions taken
    alert_sent = Column(Boolean, default=False)
    alert_sent_at = Column(DateTime, nullable=True)
    escalated = Column(Boolean, default=False)
    escalated_at = Column(DateTime, nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Relationships
    camera = relationship("Camera", back_populates="violations")
    
    def to_dict(self):
        return {
            'id': self.id,
            'violation_id': self.violation_id,
            'camera_id': self.camera_id,
            'track_id': self.track_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'resolved': self.resolved,
            'max_risk_score': self.max_risk_score,
            'avg_risk_score': self.avg_risk_score,
            'alert_level': self.alert_level,
            'zone_id': self.zone_id,
            'worker_id': self.worker_id,
            'alert_sent': self.alert_sent,
            'escalated': self.escalated
        }


class HazardZoneDB(Base):
    """Hazard zone definitions"""
    __tablename__ = 'hazard_zones'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Zone geometry (stored as JSON)
    polygon = Column(JSON, nullable=False)  # List of {x, y} points
    
    # Hazard configuration
    hazard_level = Column(String(20), nullable=False)  # SAFE, LOW, MEDIUM, HIGH, CRITICAL
    risk_multiplier = Column(Float, default=1.0)
    
    # Status
    active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'zone_id': self.zone_id,
            'name': self.name,
            'description': self.description,
            'polygon': self.polygon,
            'hazard_level': self.hazard_level,
            'risk_multiplier': self.risk_multiplier,
            'active': self.active
        }


class Alert(Base):
    """Alert/notification records"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Alert details
    timestamp = Column(DateTime, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # violation, system, maintenance
    alert_level = Column(String(20), nullable=False, index=True)  # info, warning, critical, emergency
    
    # Content
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    
    # Related entities
    camera_id = Column(String(50), nullable=True)
    violation_id = Column(String(100), nullable=True)
    track_id = Column(Integer, nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Status
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(200), nullable=True)
    
    # Delivery
    channels_sent = Column(JSON, nullable=True)  # List of channels (email, sms, dashboard)
    
    def to_dict(self):
        return {
            'id': self.id,
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'alert_level': self.alert_level,
            'title': self.title,
            'message': self.message,
            'camera_id': self.camera_id,
            'violation_id': self.violation_id,
            'acknowledged': self.acknowledged,
            'channels_sent': self.channels_sent
        }


class ComplianceReport(Base):
    """Daily/hourly compliance reports"""
    __tablename__ = 'compliance_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Time period
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # hourly, daily, weekly, monthly
    
    # Camera
    camera_id = Column(String(50), nullable=True, index=True)
    
    # Statistics
    total_detections = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    unique_workers = Column(Integer, default=0)
    compliance_rate = Column(Float, default=0.0)
    
    # Risk metrics
    avg_risk_score = Column(Float, default=0.0)
    max_risk_score = Column(Float, default=0.0)
    high_risk_incidents = Column(Integer, default=0)
    
    # Violations breakdown
    short_violations = Column(Integer, default=0)  # < 3 seconds
    medium_violations = Column(Integer, default=0)  # 3-10 seconds
    long_violations = Column(Integer, default=0)  # > 10 seconds
    
    # Top violators
    top_violators = Column(JSON, nullable=True)  # List of track IDs with counts
    
    # Generated at
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'report_id': self.report_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'period_type': self.period_type,
            'camera_id': self.camera_id,
            'total_detections': self.total_detections,
            'total_violations': self.total_violations,
            'unique_workers': self.unique_workers,
            'compliance_rate': self.compliance_rate,
            'avg_risk_score': self.avg_risk_score,
            'max_risk_score': self.max_risk_score,
            'high_risk_incidents': self.high_risk_incidents
        }


# Database connection management
class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: PostgreSQL connection URL
                         Format: postgresql://user:password@host:port/dbname
        """
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/helmet_detection'
            )
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        print("Database tables created successfully")
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def setup_timescaledb(self):
        """
        Setup TimescaleDB hypertables for time-series optimization
        Run this after creating tables
        """
        session = self.get_session()
        try:
            # Convert detections table to hypertable
            session.execute(
                "SELECT create_hypertable('detections', 'timestamp', if_not_exists => TRUE);"
            )
            
            # Create indexes for better query performance
            session.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_camera_time ON detections (camera_id, timestamp DESC);"
            )
            session.execute(
                "CREATE INDEX IF NOT EXISTS idx_detections_class_time ON detections (class_name, timestamp DESC);"
            )
            
            session.commit()
            print("TimescaleDB hypertables configured successfully")
        except Exception as e:
            print(f"Error setting up TimescaleDB: {e}")
            session.rollback()
        finally:
            session.close()


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    db.create_tables()
    
    # Uncomment if using TimescaleDB
    # db.setup_timescaledb()
    
    print("Database setup complete")
