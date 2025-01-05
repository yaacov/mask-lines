# Data Setup

## Directory Structure
```
data/
  raw/
    train/
      inputs/   # Training source images
      targets/  # Training ground-truth
    val/
      inputs/   # Validation source
      targets/  # Validation ground-truth
    test/       # Optional test set
      inputs/
      targets/
```

## Configuration
Update `src/config/config.yaml`:
```yaml
paths:
  train_input_dir: "data/raw/train/inputs"
  train_target_dir: "data/raw/train/targets"
  val_input_dir: "data/raw/val/inputs"
  val_target_dir: "data/raw/val/targets"
```
