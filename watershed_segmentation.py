"""
Watershed-Based Image Segmentation (marker-controlled)
Digital Image Processing - Mini Project

Pipeline
--------
1. Grayscale + Gaussian blur        -> suppress noise
2. Otsu / manual threshold          -> binary image
3. Morphological opening            -> remove small specks
4. Sure background                  -> dilate the binary image
5. Distance transform + threshold   -> sure foreground (object cores)
6. Unknown region                   -> sure background - sure foreground
7. Connected components             -> markers (one label per object)
8. cv2.watershed                    -> flood from markers, boundaries = -1
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class WatershedResult:
    """Holds every intermediate step so the UI can display them."""
    gray: np.ndarray
    binary: np.ndarray
    opened: np.ndarray
    sure_bg: np.ndarray
    dist_transform: np.ndarray      # normalised to 0-255 for display
    sure_fg: np.ndarray
    unknown: np.ndarray
    markers_before: np.ndarray      # int32 labels before watershed
    markers_after: np.ndarray       # int32 labels after watershed (-1 = boundary)
    boundary_overlay: np.ndarray    # original with red boundaries (RGB)
    colored_segments: np.ndarray    # random colour per region (RGB)
    blended: np.ndarray             # colored segments blended with original (RGB)
    num_objects: int = 0
    areas: list = field(default_factory=list)


def watershed_segment(
    image_rgb: np.ndarray,
    blur_ksize: int = 5,
    threshold_mode: str = "otsu",      # "otsu" | "manual"
    manual_threshold: int = 127,
    invert: bool = False,              # True when objects are darker than background
    morph_ksize: int = 3,
    open_iterations: int = 2,
    bg_dilate_iterations: int = 3,
    fg_ratio: float = 0.6,             # sure-FG threshold = fg_ratio * max(distance)
    min_area: int = 0,                 # ignore objects smaller than this (pixels)
    seed: int = 42,
) -> WatershedResult:
    """Run marker-controlled watershed on an RGB uint8 image."""
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    image_rgb = np.ascontiguousarray(image_rgb[:, :, :3], dtype=np.uint8)

    # 1. Pre-processing
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray.copy()

    # 2. Thresholding
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if threshold_mode == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, manual_threshold, 255, flag)

    # 3. Noise removal
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                              iterations=open_iterations)

    # 4. Sure background
    sure_bg = cv2.dilate(opened, kernel, iterations=bg_dilate_iterations)

    # 5. Sure foreground via distance transform
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 6. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1            # background = 1, objects = 2..N
    markers[unknown == 255] = 0      # unknown = 0 (to be decided by watershed)
    markers_before = markers.copy()

    # 8. Watershed (needs 3-channel 8-bit image)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers_after = cv2.watershed(bgr, markers.astype(np.int32))

    # Optional: drop tiny objects
    labels = [l for l in np.unique(markers_after) if l > 1]
    areas, kept = [], []
    for l in labels:
        a = int((markers_after == l).sum())
        if a >= min_area:
            kept.append(l)
            areas.append(a)
        else:
            markers_after[markers_after == l] = 1   # merge into background

    # Visualisations
    overlay = image_rgb.copy()
    overlay[markers_after == -1] = (255, 0, 0)

    rng = np.random.default_rng(seed)
    colors = rng.integers(40, 255, size=(markers_after.max() + 2, 3), dtype=np.uint8)
    colored = np.zeros_like(image_rgb)
    for l in kept:
        colored[markers_after == l] = colors[l]
    colored[markers_after == -1] = (255, 255, 255)
    blended = cv2.addWeighted(image_rgb, 0.55, colored, 0.45, 0)
    blended[markers_after == -1] = (255, 0, 0)

    return WatershedResult(
        gray=gray, binary=binary, opened=opened, sure_bg=sure_bg,
        dist_transform=dist_vis, sure_fg=sure_fg, unknown=unknown,
        markers_before=markers_before, markers_after=markers_after,
        boundary_overlay=overlay, colored_segments=colored, blended=blended,
        num_objects=len(kept), areas=areas,
    )


def make_demo_image(size: int = 400, seed: int = 0) -> np.ndarray:
    """Synthetic image of touching coins - handy for testing."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 30, np.uint8)
    centers = [(120, 120, 55), (210, 140, 50), (300, 110, 45),
               (140, 250, 60), (250, 270, 55), (330, 330, 40)]
    for x, y, r in centers:
        cv2.circle(img, (x, y), r, (210, 200, 170), -1)
        cv2.circle(img, (x, y), r, (150, 140, 110), 3)
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        img = make_demo_image()
    res = watershed_segment(img)
    print("Objects found:", res.num_objects, "| areas:", res.areas)
    cv2.imwrite("result_blended.png", cv2.cvtColor(res.blended, cv2.COLOR_RGB2BGR))
    print("Saved result_blended.png")
