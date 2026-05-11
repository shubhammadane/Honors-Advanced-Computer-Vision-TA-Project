# Background Removal

Advanced background removal using a segmentation pipeline (via `rembg`) with a simple Gradio web UI. The app removes backgrounds from images and returns either a transparent PNG or a white-background JPG-style output.

## Key Features
- High-quality subject separation for common photos
- Transparent or white background output modes
- Lightweight web UI that runs locally
- Works well on laptop hardware (CPU)
- Simple project structure for extension and research

## How It Works
The inference pipeline uses `rembg` to generate an alpha matte from the input image. The UI accepts an image, runs background removal, and returns the cutout. You can choose between transparent or white background output.

## Project Structure
- [app.py](app.py) - Gradio UI entry point
- [src/inference.py](src/inference.py) - Background removal logic
- [scripts/download_model.py](scripts/download_model.py) - Pre-download the model weights
- [requirements.txt](requirements.txt) - Python dependencies
- [assets](assets) - (Optional) sample images
- [outputs](outputs) - (Optional) save results

## Requirements
- Windows 10/11 recommended
- Python 3.10+ recommended
- Internet access for first-time model download

## Setup (Windows)
1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
2. Activate the environment:
   ```powershell
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Run the App
```powershell
python app.py
```
Then open the local Gradio URL shown in the terminal.

## Optional: Pre-download Model Weights
```powershell
python scripts/download_model.py
```
This avoids a first-run delay when you open the web app.

## Usage Tips
- Use clear, well-lit images for the cleanest edges.
- For portraits, choose the transparent output and export as PNG.
- For product shots, white background is often best.

## Troubleshooting
- Missing package error: reinstall with `python -m pip install -r requirements.txt`
- UI does not open: confirm you ran `python app.py` inside the project folder
- First run is slow: run the pre-download script once

## Performance Notes
- Default runtime is CPU. This is fine for most laptop use.
- If you want GPU acceleration later, you can switch to a GPU-enabled runtime.

## Extending the Project
Ideas you can add later:
- Batch processing for multiple images
- Background replacement (custom images)
- Edge refinement or feathering controls
- API endpoint for integration with other apps

## License
This project is provided as-is for learning and research. You can add a license file if you plan to distribute it.
