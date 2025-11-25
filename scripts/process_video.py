"""
Complete Video Processing Pipeline
Integrates detection, tracking, risk assessment, and alert generation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import json
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.yolo_detector import HelmetDetector
from detection.tracker import DeepSORTTracker
from risk_assessment.risk_engine import RiskAssessmentEngine, AlertLevel


class VideoPipeline:
    """
    Complete pipeline for processing video with helmet detection
    """
    
    def __init__(
        self,
        detector: HelmetDetector,
        tracker: DeepSORTTracker,
        risk_engine: RiskAssessmentEngine,
        output_dir: str = "outputs",
        save_video: bool = True,
        save_metrics: bool = True
    ):
        """
        Initialize video processing pipeline
        
        Args:
            detector: Helmet detector instance
            tracker: Multi-object tracker instance
            risk_engine: Risk assessment engine instance
            output_dir: Directory to save outputs
            save_video: Save annotated video
            save_metrics: Save metrics JSON
        """
        self.detector = detector
        self.tracker = tracker
        self.risk_engine = risk_engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.save_video = save_video
        self.save_metrics = save_metrics
        
        # Statistics
        self.frame_count = 0
        self.violation_history = []
        self.alert_history = []
        self.performance_metrics = defaultdict(list)
        
    def process_video(
        self,
        video_path: str,
        output_name: Optional[str] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_name: Name for output files (default: input filename)
            skip_frames: Number of frames to skip between processing
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            Dictionary with processing results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Setup output video writer
        if output_name is None:
            output_name = Path(video_path).stem
        
        output_video_path = None
        video_writer = None
        
        if self.save_video:
            output_video_path = self.output_dir / f"{output_name}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps / (skip_frames + 1),
                (width, height)
            )
        
        # Update risk engine with video properties
        self.risk_engine.frame_width = width
        self.risk_engine.frame_height = height
        self.risk_engine.fps = fps
        
        # Processing loop
        self.frame_count = 0
        frames_processed = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Skip frames if requested
                if skip_frames > 0 and self.frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Process frame
                annotated_frame = self._process_frame(frame)
                
                # Write to output video
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                frames_processed += 1
                
                # Progress update
                if frames_processed % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
                
                # Check max frames limit
                if max_frames is not None and frames_processed >= max_frames:
                    break
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
        
        # Generate summary report
        summary = self._generate_summary(
            video_path=video_path,
            output_video_path=output_video_path,
            frames_processed=frames_processed,
            fps=fps
        )
        
        # Save metrics
        if self.save_metrics:
            metrics_path = self.output_dir / f"{output_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nMetrics saved to: {metrics_path}")
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Frames processed: {frames_processed}")
        print(f"Total violations: {summary['total_violations']}")
        print(f"High-risk incidents: {summary['high_risk_incidents']}")
        if output_video_path:
            print(f"Output video: {output_video_path}")
        print(f"{'='*60}\n")
        
        return summary
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline"""
        import time
        start_time = time.time()
        
        # 1. Detect helmets
        detections = self.detector.detect(frame)
        
        # 2. Update tracker
        tracks = self.tracker.update(detections, frame)
        
        # 3. Assess risk for violations
        violations = [t for t in tracks if t.class_name == 'no_helmet']
        
        for violation in violations:
            risk_score = self.risk_engine.assess_violation_risk(
                violation,
                tracks,
                frame.shape[:2]
            )
            
            # Update track with risk score
            violation.risk_score = risk_score.total_score
            
            # Record violation
            self.violation_history.append({
                'frame': self.frame_count,
                'track_id': violation.track_id,
                'risk_score': risk_score.total_score,
                'alert_level': risk_score.alert_level.value,
                'duration': violation.violation_duration
            })
            
            # Generate alert if critical
            if risk_score.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                self._generate_alert(violation, risk_score)
        
        # 4. Visualize
        annotated = frame.copy()
        
        # Draw hazard zones
        annotated = self.risk_engine.visualize_zones(annotated)
        
        # Draw tracks
        annotated = self.tracker.visualize_tracks(
            annotated,
            show_trajectory=True,
            show_ids=True
        )
        
        # Add statistics overlay
        annotated = self._add_statistics_overlay(annotated, tracks, violations)
        
        # Record performance
        inference_time = time.time() - start_time
        self.performance_metrics['inference_time'].append(inference_time)
        self.performance_metrics['num_detections'].append(len(detections))
        self.performance_metrics['num_violations'].append(len(violations))
        
        return annotated
    
    def _add_statistics_overlay(
        self,
        frame: np.ndarray,
        tracks: List,
        violations: List
    ) -> np.ndarray:
        """Add statistics overlay to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Statistics text
        stats = [
            f"Frame: {self.frame_count}",
            f"Active Workers: {len(tracks)}",
            f"Violations: {len(violations)}",
            f"Total Violations: {len(self.violation_history)}",
            f"Alerts: {len(self.alert_history)}",
        ]
        
        # Get site risk summary
        if tracks:
            site_summary = self.risk_engine.get_site_risk_summary(tracks)
            stats.extend([
                f"Site Risk: {site_summary['overall_risk']:.1f}",
                f"Alert Level: {site_summary['alert_level'].upper()}"
            ])
        
        y = 40
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y += 25
        
        return frame
    
    def _generate_alert(self, violation_track, risk_score):
        """Generate alert for high-risk violation"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'frame': self.frame_count,
            'track_id': violation_track.track_id,
            'alert_level': risk_score.alert_level.value,
            'risk_score': risk_score.total_score,
            'violation_duration': violation_track.violation_duration,
            'location': violation_track.bbox,
            'factors': risk_score.factors
        }
        
        self.alert_history.append(alert)
        
        # In production, send to notification system
        print(f"\n⚠️  ALERT: {risk_score.alert_level.value.upper()}")
        print(f"   Track ID: {violation_track.track_id}")
        print(f"   Risk Score: {risk_score.total_score:.1f}")
        print(f"   Duration: {violation_track.violation_duration:.1f}s\n")
    
    def _generate_summary(
        self,
        video_path: str,
        output_video_path: Optional[str],
        frames_processed: int,
        fps: float
    ) -> Dict:
        """Generate processing summary"""
        # Violation statistics
        unique_violators = set(v['track_id'] for v in self.violation_history)
        
        # Alert statistics by level
        alerts_by_level = defaultdict(int)
        for alert in self.alert_history:
            alerts_by_level[alert['alert_level']] += 1
        
        # Performance statistics
        avg_inference_time = np.mean(self.performance_metrics['inference_time'])
        avg_detections = np.mean(self.performance_metrics['num_detections'])
        
        # Tracker statistics
        tracker_summary = self.tracker.get_violation_summary()
        
        summary = {
            'video_info': {
                'input_path': video_path,
                'output_path': str(output_video_path) if output_video_path else None,
                'frames_processed': frames_processed,
                'fps': fps,
                'duration_seconds': frames_processed / fps
            },
            'detection_stats': {
                'total_violations': len(self.violation_history),
                'unique_violators': len(unique_violators),
                'avg_detections_per_frame': float(avg_detections),
                'compliance_rate': tracker_summary['average_compliance_rate']
            },
            'risk_assessment': {
                'total_alerts': len(self.alert_history),
                'alerts_by_level': dict(alerts_by_level),
                'high_risk_incidents': alerts_by_level.get('critical', 0) + alerts_by_level.get('emergency', 0)
            },
            'performance': {
                'avg_inference_time_ms': avg_inference_time * 1000,
                'avg_fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
                'total_processing_time_seconds': sum(self.performance_metrics['inference_time'])
            },
            'tracker_stats': tracker_summary,
            'violation_details': self.violation_history[:100],  # First 100 violations
            'alerts': self.alert_history
        }
        
        return summary
    
    def process_realtime(
        self,
        camera_id: int = 0,
        display: bool = True
    ):
        """
        Process real-time video stream from camera
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            display: Display annotated video
        """
        cap = cv2.VideoCapture(camera_id)
        
        print(f"Starting real-time processing from camera {camera_id}")
        print("Press 'q' to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process frame
                annotated_frame = self._process_frame(frame)
                
                # Display
                if display:
                    cv2.imshow('Helmet Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Helmet Detection Video Processing')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to model weights')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--skip', type=int, default=0, help='Skip frames')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing system...")
    detector = HelmetDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device='auto'
    )
    
    tracker = DeepSORTTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )
    
    risk_engine = RiskAssessmentEngine()
    
    # Create pipeline
    pipeline = VideoPipeline(
        detector=detector,
        tracker=tracker,
        risk_engine=risk_engine,
        output_dir=args.output
    )
    
    # Process video
    summary = pipeline.process_video(
        video_path=args.video,
        skip_frames=args.skip,
        max_frames=args.max_frames
    )
    
    print("\nProcessing Summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
