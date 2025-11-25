"""
Advanced YOLO Detector with Context-Aware Processing
Supports YOLOv8 for both detection and segmentation
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO
import torch
from collections import deque
import time


@dataclass
class Detection:
    """Represents a single detection with all metadata"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    timestamp: float = None
    risk_score: float = 0.0
    zone_id: Optional[str] = None
    
    def center(self) -> Tuple[int, int]:
        """Calculate bounding box center"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def area(self) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class HelmetDetector:
    """
    Advanced helmet detection with multi-model support and optimization
    """
    
    # Class mapping
    CLASSES = {
        0: 'helmet',
        1: 'no_helmet',
        2: 'person'
    }
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'auto',
        use_half_precision: bool = False
    ):
        """
        Initialize the helmet detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ('cpu', 'cuda', 'auto')
            use_half_precision: Use FP16 for faster inference (requires GPU)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing HelmetDetector on {self.device}")
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Enable half precision if requested and available
        if use_half_precision and self.device == 'cuda':
            self.model.model.half()
            print("Half precision (FP16) enabled")
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Warmup
        self._warmup()
        
    def _warmup(self, num_iterations: int = 10):
        """Warmup the model for consistent performance"""
        print("Warming up model...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(num_iterations):
            _ = self.model.predict(
                dummy_img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
        print("Warmup complete")
    
    def detect(
        self,
        frame: np.ndarray,
        return_raw: bool = False
    ) -> List[Detection]:
        """
        Perform helmet detection on a frame
        
        Args:
            frame: Input image (BGR format)
            return_raw: Return raw YOLO results if True
            
        Returns:
            List of Detection objects
        """
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        if return_raw:
            return results
        
        # Parse detections
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detection = Detection(
                    bbox=tuple(box.astype(int)),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.CLASSES.get(int(cls_id), 'unknown'),
                    timestamp=time.time()
                )
                detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Detection]]:
        """
        Batch inference for multiple frames (more efficient)
        
        Args:
            frames: List of input images
            batch_size: Number of frames to process at once
            
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            results = self.model.predict(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            for result in results:
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        detection = Detection(
                            bbox=tuple(box.astype(int)),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=self.CLASSES.get(int(cls_id), 'unknown'),
                            timestamp=time.time()
                        )
                        detections.append(detection)
                
                all_detections.append(detections)
        
        return all_detections
    
    def filter_helmet_violations(
        self,
        detections: List[Detection]
    ) -> List[Detection]:
        """
        Filter detections to only return helmet violations (no_helmet class)
        
        Args:
            detections: List of all detections
            
        Returns:
            List of violation detections only
        """
        return [d for d in detections if d.class_name == 'no_helmet']
    
    def associate_person_with_helmet(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Associate person detections with helmet/no_helmet detections
        Novel feature: Validates helmet detection by checking person presence
        
        Args:
            detections: List of all detections
            iou_threshold: IoU threshold for association
            
        Returns:
            List of person-helmet association dictionaries
        """
        persons = [d for d in detections if d.class_name == 'person']
        helmets = [d for d in detections if d.class_name in ['helmet', 'no_helmet']]
        
        associations = []
        
        for person in persons:
            best_match = None
            best_iou = iou_threshold
            
            for helmet in helmets:
                iou = self._calculate_iou(person.bbox, helmet.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = helmet
            
            association = {
                'person': person,
                'helmet_status': best_match.class_name if best_match else 'unknown',
                'helmet_detection': best_match,
                'confidence': best_match.confidence if best_match else 0.0,
                'iou': best_iou
            }
            associations.append(association)
        
        return associations
    
    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_risk_score: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame with color-coded boxes
        
        Args:
            frame: Input image
            detections: List of detections to visualize
            show_confidence: Display confidence scores
            show_risk_score: Display risk scores
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Color mapping
        colors = {
            'helmet': (0, 255, 0),      # Green
            'no_helmet': (0, 0, 255),   # Red
            'person': (255, 255, 0)     # Yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # Adjust color based on risk score
            if det.risk_score > 0:
                # Blend red based on risk
                risk_factor = min(det.risk_score / 100, 1.0)
                color = (
                    int(color[0] * (1 - risk_factor)),
                    int(color[1] * (1 - risk_factor)),
                    int(color[2] + (255 - color[2]) * risk_factor)
                )
            
            # Draw bounding box
            thickness = 3 if det.class_name == 'no_helmet' else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label_parts = [det.class_name.replace('_', ' ').title()]
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if show_risk_score and det.risk_score > 0:
                label_parts.append(f"Risk:{det.risk_score:.1f}")
            if det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def get_performance_stats(self) -> Dict:
        """Get detector performance statistics"""
        if not self.inference_times:
            return {}
        
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'frames_processed': self.frame_count,
            'device': self.device
        }
    
    def export_onnx(self, output_path: str = 'helmet_detector.onnx'):
        """Export model to ONNX format for deployment"""
        print(f"Exporting model to ONNX: {output_path}")
        self.model.export(format='onnx', imgsz=640)
        print("Export complete")


if __name__ == "__main__":
    # Example usage
    detector = HelmetDetector(
        model_path='yolov8n.pt',
        conf_threshold=0.5,
        device='auto'
    )
    
    # Test on sample image
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_img)
    
    print(f"Found {len(detections)} detections")
    print(f"Performance: {detector.get_performance_stats()}")
