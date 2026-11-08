# Watershed-Based Image Segmentation

Mini project - Digital Image Processing.

## Files
- `watershed_segmentation.py` - core algorithm (run directly for a command-line demo)
- `app.py` - Streamlit web app
- `requirements.txt` - dependencies

## Run
```bash
pip install -r requirements.txt

# 1) Script only
python watershed_segmentation.py              # synthetic demo
python watershed_segmentation.py my_image.jpg # your image

# 2) Streamlit app
streamlit run app.py
```

## Algorithm (marker-controlled watershed)
1. Gaussian blur -> Otsu threshold
2. Morphological opening (noise removal)
3. Sure background (dilation) and sure foreground (distance transform + threshold)
4. Unknown = background - foreground
5. Connected components -> markers
6. `cv2.watershed` -> boundaries (-1) separating touching objects

## Tips
- Dark objects on a light background -> tick **Invert**.
- Over-segmentation (too many pieces) -> raise the sure-foreground ratio or blur.
- Under-segmentation (objects merged) -> lower the sure-foreground ratio.
