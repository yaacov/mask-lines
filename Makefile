# Makefile for ORTHO line catcher Project

ENV_NAME = deeper_unet_env

.PHONY: help install create_env activate_env train trainresume test inference clean format lint gui

help:
	@echo "Available make targets:"
	@echo "  install        Install Python dependencies via pip (from requirements.txt)"
	@echo "  train          Run the training script (src/training/train.py)"
	@echo "  trainresume    Run the training script with resume (src/training/train.py)"
	@echo "  test           Run unit tests with pytest (tests/ directory)"
	@echo "  inference      Run the inference script (src/training/inference.py)"
	@echo "  format         Format Python code in src/ and tests/ using Black"
	@echo "  lint           Check Python code formatting with Black (no changes applied)"
	@echo "  clean          Remove __pycache__ folders"
	@echo "  gui            Run the mask GUI editor (src/editor/editor.py)"

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python src/training/train.py --config src/config/config.yaml

trainresume:
	PYTHONPATH=. python src/training/train.py --config src/config/config.yaml --resume

test:
	PYTHONPATH=. pytest tests/

inference:
	PYTHONPATH=. python src/training/inference.py --input data/processed/test --output results/ --model models/ortho_lines.pth --patch_size 256 --step 256

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Removed all __pycache__ directories."

format:
	black src/ tests/

lint:
	black --check src/ tests/

gui:
	PYTHONPATH=. python src/editor/editor.py
