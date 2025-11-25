"""
DeepSORT Multi-Object Tracker for Helmet Detection
Tracks individuals across frames for temporal violation analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import time


@dataclass
class Track:
    """Represents a tracked object across frames"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    feature_vector: Optional[np.ndarray] = None
    
    # Tracking state
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = 'tentative'  # tentative, confirmed, deleted
    
    # History
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Optional[np.ndarray] = None
    
    # Violation tracking (Novel feature)
    violation_count: int = 0
    violation_duration: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    compliance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        self.bbox_history.append(self.bbox)
        
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, class_name: str):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = time.time()
        
        # Update history
        self.bbox_history.append(bbox)
        
        # Calculate velocity
        if len(self.bbox_history) >= 2:
            prev_center = self._get_center(self.bbox_history[-2])
            curr_center = self._get_center(bbox)
            self.velocity = np.array([
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1]
            ])
        
        # Update state
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
        
        # Track violations
        if class_name == 'no_helmet':
            self.violation_count += 1
            self.violation_duration += 1/30.0  # Assuming 30 FPS
        
        self.compliance_history.append(1 if class_name == 'helmet' else 0)
    
    def predict(self):
        """Predict next position using velocity"""
        self.age += 1
        self.time_since_update += 1
        
        if self.velocity is not None and self.time_since_update < 5:
            x1, y1, x2, y2 = self.bbox
            cx, cy = self._get_center(self.bbox)
            
            # Predict new center
            new_cx = cx + self.velocity[0]
            new_cy = cy + self.velocity[1]
            
            # Update bbox
            w = x2 - x1
            h = y2 - y1
            self.bbox = (
                int(new_cx - w/2),
                int(new_cy - h/2),
                int(new_cx + w/2),
                int(new_cy + h/2)
            )
    
    def mark_missed(self):
        """Mark frame as missed"""
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > 30:  # 1 second at 30 FPS
            self.state = 'deleted'
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate bbox center"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def get_compliance_rate(self) -> float:
        """Calculate compliance rate from history"""
        if not self.compliance_history:
            return 0.0
        return sum(self.compliance_history) / len(self.compliance_history)
    
    def to_dict(self) -> Dict:
        """Convert track to dictionary"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_name': self.class_name,
            'hits': self.hits,
            'age': self.age,
            'state': self.state,
            'violation_count': self.violation_count,
            'violation_duration': self.violation_duration,
            'compliance_rate': self.get_compliance_rate(),
            'first_seen': self.first_seen,
            'last_seen': self.last_seen
        }


class SimpleFeatureExtractor:
    """
    Lightweight feature extractor for tracking
    Uses color histograms and HOG features
    """
    
    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract features from detection
        
        Args:
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Feature vector
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(256)
        
        roi = frame[y1:y2, x1:x2]
        
        # Resize for consistency
        roi = cv2.resize(roi, (64, 128))
        
        # Color histogram (3 channels * 32 bins = 96 dims)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([roi], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        # HOG features (simplified)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        hog_features = hog.compute(gray).flatten()
        
        # Combine features (take subset of HOG to keep dimension reasonable)
        features = np.concatenate([
            np.array(hist_features),
            hog_features[:160]
        ])
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features


class DeepSORTTracker:
    """
    DeepSORT tracker with enhanced violation tracking capabilities
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        feature_threshold: float = 0.7
    ):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits to confirm a track
            iou_threshold: IoU threshold for matching
            feature_threshold: Feature similarity threshold
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.feature_extractor = SimpleFeatureExtractor()
        
        # Statistics
        self.frame_count = 0
        self.total_tracks = 0
        
    def update(
        self,
        detections: List,  # List of Detection objects
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of Detection objects from detector
            frame: Current frame (needed for feature extraction)
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched_indices, unmatched_detections, unmatched_tracks = \
            self._match_detections_to_tracks(detections, frame)
        
        # Update matched tracks
        for det_idx, track_idx in matched_indices:
            detection = detections[det_idx]
            track = self.tracks[track_idx]
            
            # Extract features if frame available
            if frame is not None:
                features = self.feature_extractor.extract(frame, detection.bbox)
                track.feature_vector = features
            
            track.update(detection.bbox, detection.confidence, detection.class_name)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Extract features
            features = None
            if frame is not None:
                features = self.feature_extractor.extract(frame, detection.bbox)
            
            track = Track(
                track_id=self.next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_name=detection.class_name,
                feature_vector=features
            )
            
            self.tracks.append(track)
            self.next_id += 1
            self.total_tracks += 1
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.state == 'confirmed']
    
    def _match_detections_to_tracks(
        self,
        detections: List,
        frame: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU and features
        
        Returns:
            matched_indices: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Build cost matrix (lower is better)
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d_idx, detection in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                # IoU cost
                iou = self._calculate_iou(detection.bbox, track.bbox)
                iou_cost = 1 - iou
                
                # Feature cost (if available)
                feature_cost = 0.0
                if frame is not None and track.feature_vector is not None:
                    det_features = self.feature_extractor.extract(frame, detection.bbox)
                    feature_cost = cosine(det_features, track.feature_vector)
                
                # Combined cost (weighted)
                cost_matrix[d_idx, t_idx] = 0.6 * iou_cost + 0.4 * feature_cost
        
        # Hungarian algorithm for optimal assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on threshold
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for d_idx, t_idx in zip(det_indices, track_indices):
            if cost_matrix[d_idx, t_idx] < (1 - self.iou_threshold):
                matched_indices.append((d_idx, t_idx))
                unmatched_detections.remove(d_idx)
                unmatched_tracks.remove(t_idx)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_violation_summary(self) -> Dict:
        """
        Get summary of violations across all tracks (Novel feature)
        
        Returns:
            Dictionary with violation statistics
        """
        active_tracks = [t for t in self.tracks if t.state == 'confirmed']
        
        total_violations = sum(t.violation_count for t in active_tracks)
        persistent_violators = [
            t for t in active_tracks 
            if t.violation_duration > 3.0  # 3+ seconds
        ]
        
        compliance_rates = [t.get_compliance_rate() for t in active_tracks]
        avg_compliance = np.mean(compliance_rates) if compliance_rates else 0.0
        
        return {
            'total_active_tracks': len(active_tracks),
            'total_violations': total_violations,
            'persistent_violators': len(persistent_violators),
            'persistent_violator_ids': [t.track_id for t in persistent_violators],
            'average_compliance_rate': avg_compliance,
            'frames_processed': self.frame_count
        }
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def visualize_tracks(
        self,
        frame: np.ndarray,
        show_trajectory: bool = True,
        show_ids: bool = True
    ) -> np.ndarray:
        """
        Visualize tracks on frame
        
        Args:
            frame: Input frame
            show_trajectory: Draw trajectory lines
            show_ids: Show track IDs
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for track in self.tracks:
            if track.state != 'confirmed':
                continue
            
            x1, y1, x2, y2 = track.bbox
            
            # Color based on compliance
            if track.class_name == 'no_helmet':
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            if show_ids:
                label = f"ID:{track.track_id}"
                if track.violation_count > 0:
                    label += f" V:{track.violation_count}"
                
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Draw trajectory
            if show_trajectory and len(track.bbox_history) > 1:
                points = [
                    ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                    for b in track.bbox_history
                ]
                for i in range(len(points) - 1):
                    cv2.line(annotated, points[i], points[i+1], color, 2)
        
        return annotated


if __name__ == "__main__":
    # Example usage
    tracker = DeepSORTTracker()
    
    print("DeepSORT Tracker initialized")
    print(f"Max age: {tracker.max_age} frames")
    print(f"Min hits: {tracker.min_hits}")
