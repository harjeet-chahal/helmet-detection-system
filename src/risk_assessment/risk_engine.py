"""
Risk Assessment Engine - Novel Feature
Multi-factor risk scoring for helmet violations based on:
- Proximity to hazardous zones
- Duration of violation
- Number of workers in scene
- Historical compliance patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class HazardLevel(Enum):
    """Hazard level enumeration"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HazardZone:
    """Defines a hazardous zone in the construction site"""
    zone_id: str
    name: str
    polygon: List[Tuple[int, int]]  # Polygon vertices
    hazard_level: HazardLevel
    description: str
    multiplier: float = 1.0  # Risk multiplier for this zone
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(self.polygon)
        inside = False
        
        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


@dataclass
class RiskScore:
    """Risk score with breakdown of contributing factors"""
    total_score: float
    zone_risk: float
    duration_risk: float
    density_risk: float
    history_risk: float
    alert_level: AlertLevel
    factors: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_score': self.total_score,
            'zone_risk': self.zone_risk,
            'duration_risk': self.duration_risk,
            'density_risk': self.density_risk,
            'history_risk': self.history_risk,
            'alert_level': self.alert_level.value,
            'factors': self.factors
        }


class RiskAssessmentEngine:
    """
    Advanced risk assessment engine for helmet violations
    Novel contribution: Multi-factor contextual risk scoring
    """
    
    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        fps: float = 30.0
    ):
        """
        Initialize risk assessment engine
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Hazard zones (can be configured via API)
        self.hazard_zones: List[HazardZone] = []
        
        # Risk thresholds
        self.thresholds = {
            'warning': 30.0,
            'critical': 60.0,
            'emergency': 80.0
        }
        
        # Weights for risk factors
        self.weights = {
            'zone': 0.35,
            'duration': 0.25,
            'density': 0.20,
            'history': 0.20
        }
        
        # Statistics
        self.risk_history = []
        
    def add_hazard_zone(self, zone: HazardZone):
        """Add a hazard zone to the site"""
        self.hazard_zones.append(zone)
        
    def load_zones_from_config(self, config_path: str):
        """Load hazard zones from JSON configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for zone_data in config.get('hazard_zones', []):
            zone = HazardZone(
                zone_id=zone_data['id'],
                name=zone_data['name'],
                polygon=[(p['x'], p['y']) for p in zone_data['polygon']],
                hazard_level=HazardLevel[zone_data['hazard_level']],
                description=zone_data.get('description', ''),
                multiplier=zone_data.get('multiplier', 1.0)
            )
            self.add_hazard_zone(zone)
    
    def assess_violation_risk(
        self,
        track: 'Track',  # Track object from tracker
        all_tracks: List['Track'],
        frame_shape: Tuple[int, int]
    ) -> RiskScore:
        """
        Assess risk for a single violation
        
        Args:
            track: Track with violation (no_helmet)
            all_tracks: All active tracks in scene
            frame_shape: (height, width) of frame
            
        Returns:
            RiskScore object
        """
        factors = {}
        
        # 1. Zone Risk - proximity to hazardous zones
        zone_risk = self._calculate_zone_risk(track)
        factors['zone_proximity'] = zone_risk
        
        # 2. Duration Risk - how long violation persists
        duration_risk = self._calculate_duration_risk(track)
        factors['violation_duration'] = duration_risk
        
        # 3. Density Risk - number of people in scene
        density_risk = self._calculate_density_risk(all_tracks, frame_shape)
        factors['worker_density'] = density_risk
        
        # 4. History Risk - compliance history
        history_risk = self._calculate_history_risk(track)
        factors['compliance_history'] = history_risk
        
        # Calculate weighted total
        total_score = (
            self.weights['zone'] * zone_risk +
            self.weights['duration'] * duration_risk +
            self.weights['density'] * density_risk +
            self.weights['history'] * history_risk
        )
        
        # Determine alert level
        alert_level = self._determine_alert_level(total_score)
        
        risk_score = RiskScore(
            total_score=total_score,
            zone_risk=zone_risk,
            duration_risk=duration_risk,
            density_risk=density_risk,
            history_risk=history_risk,
            alert_level=alert_level,
            factors=factors
        )
        
        self.risk_history.append(risk_score)
        
        return risk_score
    
    def _calculate_zone_risk(self, track: 'Track') -> float:
        """
        Calculate risk based on proximity to hazardous zones
        Returns: 0-100 score
        """
        if not self.hazard_zones:
            return 0.0
        
        # Get track center
        x1, y1, x2, y2 = track.bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        max_risk = 0.0
        
        for zone in self.hazard_zones:
            if zone.contains_point(center):
                # Inside hazardous zone
                zone_risk = zone.hazard_level.value * 20.0 * zone.multiplier
                max_risk = max(max_risk, zone_risk)
            else:
                # Calculate distance to zone
                min_dist = self._min_distance_to_polygon(center, zone.polygon)
                
                # Risk decreases with distance (50px = 0 risk)
                if min_dist < 50:
                    proximity_factor = 1.0 - (min_dist / 50.0)
                    zone_risk = zone.hazard_level.value * 15.0 * proximity_factor * zone.multiplier
                    max_risk = max(max_risk, zone_risk)
        
        return min(max_risk, 100.0)
    
    def _calculate_duration_risk(self, track: 'Track') -> float:
        """
        Calculate risk based on violation duration
        Returns: 0-100 score
        """
        # Risk increases with duration
        # 0 seconds = 0, 10+ seconds = 100
        duration_seconds = track.violation_duration
        
        if duration_seconds < 1.0:
            return 10.0  # Initial warning
        elif duration_seconds < 3.0:
            return 30.0
        elif duration_seconds < 5.0:
            return 50.0
        elif duration_seconds < 10.0:
            return 75.0
        else:
            return 100.0
    
    def _calculate_density_risk(
        self,
        all_tracks: List['Track'],
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Calculate risk based on worker density
        More workers = higher risk if incident occurs
        Returns: 0-100 score
        """
        num_workers = len(all_tracks)
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Calculate density (workers per megapixel)
        density = num_workers / (frame_area / 1e6)
        
        # Risk increases with density
        if density < 2:
            return 10.0
        elif density < 5:
            return 30.0
        elif density < 10:
            return 60.0
        else:
            return 90.0
    
    def _calculate_history_risk(self, track: 'Track') -> float:
        """
        Calculate risk based on compliance history
        Repeat offenders get higher risk
        Returns: 0-100 score
        """
        if track.hits < 10:
            # Not enough history
            return 20.0
        
        compliance_rate = track.get_compliance_rate()
        
        # Lower compliance = higher risk
        if compliance_rate > 0.9:
            return 10.0
        elif compliance_rate > 0.7:
            return 30.0
        elif compliance_rate > 0.5:
            return 60.0
        else:
            return 90.0  # Persistent violator
    
    def _determine_alert_level(self, risk_score: float) -> AlertLevel:
        """Determine alert level from risk score"""
        if risk_score >= self.thresholds['emergency']:
            return AlertLevel.EMERGENCY
        elif risk_score >= self.thresholds['critical']:
            return AlertLevel.CRITICAL
        elif risk_score >= self.thresholds['warning']:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _min_distance_to_polygon(
        self,
        point: Tuple[int, int],
        polygon: List[Tuple[int, int]]
    ) -> float:
        """Calculate minimum distance from point to polygon"""
        x, y = point
        min_dist = float('inf')
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            # Distance to line segment
            dist = self._point_to_segment_distance((x, y), p1, p2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(
        self,
        point: Tuple[int, int],
        seg_start: Tuple[int, int],
        seg_end: Tuple[int, int]
    ) -> float:
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def get_site_risk_summary(self, all_tracks: List['Track']) -> Dict:
        """
        Get overall site risk summary
        
        Returns:
            Dictionary with site-wide risk metrics
        """
        if not all_tracks:
            return {
                'overall_risk': 0.0,
                'alert_level': AlertLevel.INFO.value,
                'active_violations': 0,
                'total_workers': 0
            }
        
        violations = [t for t in all_tracks if t.class_name == 'no_helmet']
        
        if not violations:
            return {
                'overall_risk': 0.0,
                'alert_level': AlertLevel.INFO.value,
                'active_violations': 0,
                'total_workers': len(all_tracks)
            }
        
        # Calculate average risk across violations
        violation_risks = []
        for violation in violations:
            risk = self.assess_violation_risk(
                violation,
                all_tracks,
                (self.frame_height, self.frame_width)
            )
            violation_risks.append(risk.total_score)
        
        overall_risk = np.mean(violation_risks)
        alert_level = self._determine_alert_level(overall_risk)
        
        return {
            'overall_risk': overall_risk,
            'alert_level': alert_level.value,
            'active_violations': len(violations),
            'total_workers': len(all_tracks),
            'violation_rate': len(violations) / len(all_tracks),
            'max_individual_risk': max(violation_risks),
            'high_risk_violations': sum(1 for r in violation_risks if r > 60)
        }
    
    def visualize_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize hazard zones on frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with zones drawn
        """
        annotated = frame.copy()
        
        # Color mapping for hazard levels
        colors = {
            HazardLevel.SAFE: (0, 255, 0),
            HazardLevel.LOW: (0, 255, 255),
            HazardLevel.MEDIUM: (0, 165, 255),
            HazardLevel.HIGH: (0, 100, 255),
            HazardLevel.CRITICAL: (0, 0, 255)
        }
        
        for zone in self.hazard_zones:
            color = colors[zone.hazard_level]
            
            # Draw polygon
            pts = np.array(zone.polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Semi-transparent overlay
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            
            # Draw boundary
            cv2.polylines(annotated, [pts], True, color, 2)
            
            # Add label
            center_x = int(np.mean([p[0] for p in zone.polygon]))
            center_y = int(np.mean([p[1] for p in zone.polygon]))
            
            cv2.putText(
                annotated,
                f"{zone.name} ({zone.hazard_level.name})",
                (center_x - 50, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated


if __name__ == "__main__":
    # Example usage
    engine = RiskAssessmentEngine()
    
    # Add sample hazard zone
    excavation_zone = HazardZone(
        zone_id="zone_1",
        name="Excavation Area",
        polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
        hazard_level=HazardLevel.CRITICAL,
        description="Deep excavation zone",
        multiplier=1.5
    )
    
    engine.add_hazard_zone(excavation_zone)
    print(f"Added hazard zone: {excavation_zone.name}")
    print(f"Risk thresholds: {engine.thresholds}")
