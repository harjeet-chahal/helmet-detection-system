# Model Training & Improvement Roadmap

## Current State

Using pretrained YOLOv8n (6.2MB, trained on COCO dataset) which recognizes general objects including "person" but not specifically "helmet" or "no_helmet".

**Why I haven't trained custom yet**:
1. Demonstrating system architecture and pipeline first
2. Pretrained model proves concept works
3. Custom training requires labeled construction site dataset

## Training Plan

### Phase 1: Data Collection (In Progress)

**Target**: 5,000-10,000 images minimum

**Sources**:
- Roboflow Universe construction datasets
- Kaggle safety helmet datasets
- YouTube construction videos (extract frames)
- Partner with local construction companies

**Labeling requirements**:
- 3 classes: `helmet`, `no_helmet`, `person`
- Bounding boxes in YOLO format
- Include edge cases: partial occlusion, various angles, different helmet colors

### Phase 2: Data Augmentation

**Planned augmentations** (already in requirements.txt: albumentations):
```python
augmentation_pipeline = [
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.3),
    GaussNoise(p=0.2),
    Blur(blur_limit=3, p=0.2),
    RandomResizedCrop(scale=(0.8, 1.0), p=0.3)
]
```

**Why these**:
- Construction sites have varying lighting
- Camera angles change
- Weather conditions affect image quality

### Phase 3: Training Configuration

Already have training notebook: `notebooks/02_model_training.ipynb`

**Hyperparameters** (will tune):
```yaml
epochs: 100
batch: 16  
imgsz: 640
optimizer: AdamW
lr0: 0.001
augment: True
```

**Hardware target**:
- Local: MacBook (will take ~24 hours)
- Cloud: Google Colab Pro / AWS (5-8 hours)

### Phase 4: Expected Improvements

| Metric | Current (Pretrained) | Target (Custom) |
|--------|---------------------|-----------------|
| Helmet detection mAP | N/A | >90% |
| No-helmet detection mAP | N/A | >85% |
| False positive rate | Unknown | <5% |
| Inference speed | 65ms | 65ms (same architecture) |

### Why Not Trained Yet?

**Honest reasons**:
1. **Proving the concept**: System architecture matters more than perfect accuracy for a portfolio project
2. **Data access**: Quality labeled construction data is expensive/time-consuming
3. **Compute constraints**: Training requires GPU (have access, but time investment)

**What I've done instead**:
- Built complete production pipeline
- Implemented novel risk assessment
- Created deployable system
- Demonstrated end-to-end workflow

### Next Steps (Immediate)

1. Download Roboflow construction helmet dataset
2. Run training notebook for 50 epochs as proof-of-concept
3. Document results before/after
4. Update model in Docker container

**Timeline**: Can complete in 1-2 days with focused effort

---

**The Point**: Having a fully deployable system with pretrained model demonstrates more ML engineering skills than just training a model in a notebook. The architecture, API design, and production considerations are what separate this from a tutorial project.
