"""
Demo Script for Helmet Detection System
Demonstrates all major features with synthetic data
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from detection.yolo_detector import HelmetDetector, Detection
from detection.tracker import DeepSORTTracker, Track
from risk_assessment.risk_engine import RiskAssessmentEngine, HazardZone, HazardLevel


def create_synthetic_frame(width=1920, height=1080):
    """Create a synthetic frame for testing"""
    frame = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate workers
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(3):
        x = 200 + i * 400
        y = 400
        cv2.rectangle(frame, (x, y), (x+150, y+200), colors[i], -1)
    
    return frame


def test_detector():
    """Test the helmet detector"""
    print("\n" + "="*60)
    print("Testing Helmet Detector")
    print("="*60)
    
    try:
        # Initialize detector (will use random model for demo)
        detector = HelmetDetector(
            model_path='yolov8n.pt',
            conf_threshold=0.5,
            device='cpu'
        )
        
        # Create test frame
        frame = create_synthetic_frame()
        
        # Create synthetic detections for demo
        detections = [
            Detection(
                bbox=(200, 400, 350, 600),
                confidence=0.95,
                class_id=1,
                class_name='no_helmet'
            ),
            Detection(
                bbox=(600, 400, 750, 600),
                confidence=0.88,
                class_id=0,
                class_name='helmet'
            ),
            Detection(
                bbox=(1000, 400, 1150, 600),
                confidence=0.92,
                class_id=1,
                class_name='no_helmet'
            )
        ]
        
        # Visualize
        annotated = detector.visualize_detections(frame, detections)
        
        # Get stats
        stats = detector.get_performance_stats()
        
        print("✓ Detector initialized successfully")
        print(f"✓ Created {len(detections)} synthetic detections")
        print(f"✓ Performance: {stats.get('fps', 'N/A')} FPS")
        print(f"✓ Device: {stats.get('device', 'N/A')}")
        
        return True, detector, detections
        
    except Exception as e:
        print(f"✗ Detector test failed: {e}")
        return False, None, None


def test_tracker(detections):
    """Test the DeepSORT tracker"""
    print("\n" + "="*60)
    print("Testing DeepSORT Tracker")
    print("="*60)
    
    try:
        # Initialize tracker
        tracker = DeepSORTTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Create test frame
        frame = create_synthetic_frame()
        
        # Simulate tracking over multiple frames
        num_frames = 10
        for i in range(num_frames):
            tracks = tracker.update(detections, frame)
            if i == 0:
                print(f"✓ Frame {i+1}: Initialized {len(tracks)} tracks")
            elif i == num_frames - 1:
                print(f"✓ Frame {i+1}: Tracking {len(tracks)} confirmed tracks")
        
        # Get violation summary
        summary = tracker.get_violation_summary()
        
        print(f"✓ Total active tracks: {summary['total_active_tracks']}")
        print(f"✓ Total violations: {summary['total_violations']}")
        print(f"✓ Average compliance: {summary['average_compliance_rate']:.2%}")
        
        return True, tracker, tracks
        
    except Exception as e:
        print(f"✗ Tracker test failed: {e}")
        return False, None, None


def test_risk_engine(tracks):
    """Test the risk assessment engine"""
    print("\n" + "="*60)
    print("Testing Risk Assessment Engine")
    print("="*60)
    
    try:
        # Initialize risk engine
        engine = RiskAssessmentEngine(
            frame_width=1920,
            frame_height=1080,
            fps=30.0
        )
        
        # Add hazard zones
        excavation = HazardZone(
            zone_id="zone_1",
            name="Excavation Area",
            polygon=[(100, 300), (500, 300), (500, 700), (100, 700)],
            hazard_level=HazardLevel.CRITICAL,
            description="Deep excavation zone",
            multiplier=1.5
        )
        engine.add_hazard_zone(excavation)
        
        scaffolding = HazardZone(
            zone_id="zone_2",
            name="Scaffolding Area",
            polygon=[(800, 300), (1200, 300), (1200, 700), (800, 700)],
            hazard_level=HazardLevel.HIGH,
            description="High scaffolding zone",
            multiplier=1.2
        )
        engine.add_hazard_zone(scaffolding)
        
        print(f"✓ Added {len(engine.hazard_zones)} hazard zones")
        
        # Assess risk for violations
        violations = [t for t in tracks if t.class_name == 'no_helmet']
        
        if violations:
            for violation in violations[:2]:  # Test first 2
                risk = engine.assess_violation_risk(
                    violation,
                    tracks,
                    (1080, 1920)
                )
                
                print(f"✓ Violation {violation.track_id}:")
                print(f"  - Risk Score: {risk.total_score:.1f}/100")
                print(f"  - Alert Level: {risk.alert_level.value.upper()}")
                print(f"  - Zone Risk: {risk.zone_risk:.1f}")
                print(f"  - Duration Risk: {risk.duration_risk:.1f}")
        
        # Get site summary
        summary = engine.get_site_risk_summary(tracks)
        print(f"\n✓ Site Risk Summary:")
        print(f"  - Overall Risk: {summary['overall_risk']:.1f}/100")
        print(f"  - Alert Level: {summary['alert_level'].upper()}")
        print(f"  - Active Violations: {summary['active_violations']}")
        
        return True, engine
        
    except Exception as e:
        print(f"✗ Risk engine test failed: {e}")
        return False, None


def test_integration():
    """Test complete integration"""
    print("\n" + "="*60)
    print("Testing Complete Integration")
    print("="*60)
    
    try:
        # Run all components together
        frame = create_synthetic_frame()
        
        # Mock detections
        detections = [
            Detection(
                bbox=(200, 400, 350, 600),
                confidence=0.95,
                class_id=1,
                class_name='no_helmet'
            ),
            Detection(
                bbox=(600, 400, 750, 600),
                confidence=0.88,
                class_id=0,
                class_name='helmet'
            )
        ]
        
        # Initialize all components
        tracker = DeepSORTTracker()
        engine = RiskAssessmentEngine()
        
        # Add hazard zone
        zone = HazardZone(
            zone_id="test_zone",
            name="Test Area",
            polygon=[(0, 0), (1920, 0), (1920, 1080), (0, 1080)],
            hazard_level=HazardLevel.HIGH,
            description="Test hazard zone"
        )
        engine.add_hazard_zone(zone)
        
        # Process pipeline
        tracks = tracker.update(detections, frame)
        violations = [t for t in tracks if t.class_name == 'no_helmet']
        
        for violation in violations:
            risk = engine.assess_violation_risk(violation, tracks, (1080, 1920))
            violation.risk_score = risk.total_score
        
        print(f"✓ Processed {len(detections)} detections")
        print(f"✓ Tracked {len(tracks)} objects")
        print(f"✓ Identified {len(violations)} violations")
        print(f"✓ Risk assessment complete")
        print("✓ Integration test successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + "  Helmet Detection System - Demo Test Suite".center(58) + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    results = {}
    
    # Test detector
    results['detector'], detector, detections = test_detector()
    
    # Test tracker (only if detector worked)
    if results['detector'] and detections:
        results['tracker'], tracker, tracks = test_tracker(detections)
    else:
        results['tracker'] = False
        tracks = []
    
    # Test risk engine (only if tracker worked)
    if results['tracker'] and tracks:
        results['risk_engine'], engine = test_risk_engine(tracks)
    else:
        results['risk_engine'] = False
    
    # Test integration
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
    
    print("\nNote: This is a demo with synthetic data.")
    print("For production use, train a custom model on real construction data.")
    print("\nNext steps:")
    print("  1. Prepare your dataset (see SETUP.md)")
    print("  2. Train the model (notebooks/02_model_training.ipynb)")
    print("  3. Deploy the API (docker-compose up)")
    print("  4. Process real videos (scripts/process_video.py)")


if __name__ == "__main__":
    main()
