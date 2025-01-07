# ORTHO line catcher Project

## Project Overview
This project aims to train a ORTHO (Old Research Text Hebrew OCR) lines catcher
model to perform line tracing tasks on large text images by splitting them into 
smaller patches. The approach allows for higher-capacity models to handle 
arbitrarily large text images without running out of memory.

## Installation
1. (Recommended) Create a new Python virtual environment or conda environment
2. Install dependencies:
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

## Usage

### 1. Training
```bash
make train
```
This runs:
```bash
python src/training/train.py --config src/config/config.yaml
```
Adjust config.yaml if needed (epochs, learning rate, batch size, paths).

### 2. Testing
```bash
make test
```
Runs pytest on the tests/ folder.

### 3. Inference
```bash
make inference
```
This runs:
```bash
python src/training/inference.py --input data/processed/test --output results/ \
    --model models/ortho_lines.pth --patch_size 256 --step 256
```
You can pass different --input / --output paths as needed.

## Code Formatting & Linting
We use Black for code formatting:
- To format code automatically:
  ```bash
  make format
  ```
- To check formatting (no changes applied):
  ```bash
  make lint
  ```

## Notes
- If you need to handle very large images (e.g., 4000x4000), increase patch_size 
  or step accordingly but watch out for GPU memory usage.
- For overlapping patches, consider averaging or blending the overlap regions 
  when reconstructing full images.

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0)

For more details, see the [GNU GPL v3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
