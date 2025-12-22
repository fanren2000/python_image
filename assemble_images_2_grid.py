import cv2
import os
import math
import numpy as np
import random
import re
from PIL import Image, ImageDraw, ImageFont

import random

def resize_to_fill(img, target_w, target_h, crop_mode="center",
                   weights={"top": 1, "center": 1, "bottom": 1}):
    """
    Resize while preserving aspect ratio, then crop to fill target size.
    crop_mode:
        "top"    → prioritize head
        "center" → balanced
        "bottom" → prioritize feet
        "random" → weighted random choice among top/center/bottom

    weights: dict controlling probability when crop_mode="random"
        e.g., {"top": 3, "center": 1, "bottom": 1}
    """

    # -----------------------------
    # 1. Handle random crop mode
    # -----------------------------
    if crop_mode == "random":
        modes = ["top", "center", "bottom"]
        probs = [weights.get(m, 1) for m in modes]
        crop_mode = random.choices(modes, weights=probs, k=1)[0]

    # -----------------------------
    # 2. Resize to cover target area
    # -----------------------------
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        # Image is wider → height matches, width overflows
        new_h = target_h
        new_w = int(img_ratio * new_h)
    else:
        # Image is taller → width matches, height overflows
        new_w = target_w
        new_h = int(new_w / img_ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # -----------------------------
    # 3. Compute crop offsets
    # -----------------------------
    y_excess = new_h - target_h
    x_excess = new_w - target_w

    # Horizontal crop always centered
    x1 = x_excess // 2

    # Vertical crop depends on crop_mode
    if crop_mode == "top":
        y1 = 0
    elif crop_mode == "bottom":
        y1 = y_excess
    else:  # "center"
        y1 = y_excess // 2

    # -----------------------------
    # 4. Crop and return
    # -----------------------------
    cropped = resized[y1:y1 + target_h, x1:x1 + target_w]
    return cropped

def second_sort_key(path):
    """
    Extracts the numeric seconds from filenames like 'frame_1.99s.jpg'
    and returns it as a float for sorting.
    """
    fname = os.path.basename(path)
    match = re.search(r'_(\d+(\.\d+)?)s', fname)
    if match:
        return float(match.group(1))
    return float('inf')

def resolve_theme(theme):
    """
    Returns (border_color, background_color) based on theme name.
    Colors are BGR tuples.
    """

    theme = theme.lower()

    if theme == "luxury":
        return (30, 215, 255), (120, 40, 0)     # brighter midnight blue, (60, 0, 0)   # gold border, midnight blue bg

    if theme == "clean":
        return (255, 255, 255), (160, 60, 0)    # true blue, (80, 0, 0)  # white border, deep navy bg

    if theme == "neon":
        return (255, 255, 0), (40, 40, 40)      # visible charcoal, (20, 20, 20)  # neon cyan border, charcoal bg

    if theme == "romantic":
        return (80, 120, 255), (40, 0, 60)  # rose-gold border, burgundy bg

    # Default fallback
    return None, (0, 0, 0)  # random border, black background

def apply_background_to_cell(cell_img, mask, border_mask, background_color):
    """
    Preserve:
      - shape interior (mask)
      - outer gradient border (border_mask)
    Replace only the true outside area with background_color.
    """

    bg = np.full_like(cell_img, background_color, dtype=np.uint8)

    # Expand masks to 3 channels
    shape_area  = (mask == 255)[..., None].repeat(3, axis=2)
    border_area = (border_mask == 255)[..., None].repeat(3, axis=2)

    # Start with background everywhere
    result = bg.copy()

    # Paste border
    result[border_area] = cell_img[border_area]

    # Paste shape interior
    result[shape_area] = cell_img[shape_area]

    return result


def apply_shape_with_gradient_border(
    img,
    shape,
    border_thickness=20,
    padding=20,
    border_color=None
):
    """
    OUTER gradient border version (corrected).
    - Border is drawn OUTSIDE the shape.
    - Background compositing will NOT overwrite the border.
    - No spacing issues.
    - No transparency issues.
    """

    h, w = img.shape[:2]

    # -------------------------
    # 1. Create base shape mask
    # -------------------------
    mask = np.zeros((h, w), dtype=np.uint8)

    if shape == "rectangle":
        cv2.rectangle(mask, (padding, padding), (w - padding, h - padding), 255, -1)

    elif shape == "square":
        side = min(h, w) - 2 * padding
        side = max(side, 1)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        cv2.rectangle(mask, (x1, y1), (x1 + side, y1 + side), 255, -1)

    elif shape == "oval":
        center = (w // 2, h // 2)
        axes = (max(w // 2 - padding, 1), max(h // 2 - padding, 1))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    elif shape == "round":
        radius = max(min(h, w) // 2 - padding, 1)
        center = (w // 2, h // 2)
        cv2.circle(mask, center, radius, 255, -1)

    elif shape == "diamond":
        cx, cy = w // 2, h // 2
        pts = np.array([
            [cx, padding],          # top
            [w - padding, cy],      # right
            [cx, h - padding],      # bottom
            [padding, cy]           # left
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    else:
        cv2.rectangle(mask, (padding, padding), (w - padding, h - padding), 255, -1)

    # ----------------------------------------
    # 2. Create OUTER border mask (dilate out)
    # ----------------------------------------
    kernel_size = border_thickness * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    outer_mask = cv2.dilate(mask, kernel)
    border_mask = outer_mask - mask   # OUTER border region

    # ----------------------------------------
    # 3. Determine gradient colors
    # ----------------------------------------
    if border_color is not None:
        color1 = np.array(border_color, dtype=np.uint8)
        color2 = np.array([0, 0, 0], dtype=np.uint8)
    else:
        color1 = np.array([random.randint(0,255) for _ in range(3)], dtype=np.uint8)
        color2 = np.array([random.randint(0,255) for _ in range(3)], dtype=np.uint8)

    # ----------------------------------------
    # 4. Create gradient using distance transform
    # ----------------------------------------
    dist_transform = cv2.distanceTransform(border_mask, cv2.DIST_L2, 3)
    if dist_transform.max() > 0:
        dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    else:
        dist_norm = dist_transform

    gradient = np.zeros_like(img)
    for c in range(3):
        gradient[:, :, c] = (
            color1[c] * (1.0 - dist_norm) +
            color2[c] * dist_norm
        ).astype(np.uint8)

    border_img = cv2.bitwise_and(gradient, gradient, mask=border_mask)

    # ----------------------------------------
    # 5. Apply shape mask to original image
    # ----------------------------------------
    shaped_img = cv2.bitwise_and(img, img, mask=mask)

    # ----------------------------------------
    # 6. Combine shaped image + OUTER gradient border
    # ----------------------------------------
    result = shaped_img.copy()
    result[border_mask == 255] = border_img[border_mask == 255]

    # Return both result and mask (mask is used for background compositing)
    return result, mask, border_mask

def add_text_watermark(
    image,
    text="WATERMARK",
    angle=0,
    font_path="simhei.ttf",
    font_size=475,
    color=(255, 255, 255),
    opacity=0.25
):
    """
    Add a rotated TTF text watermark to the center of the image.
    Uses PIL for text rendering, OpenCV for final output.
    """

    # Convert OpenCV → PIL
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Create transparent layer for watermark
     # Create transparent overlay (same size as base image)
    overlay = Image.new("RGBA", pil_img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
   
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        # fallback to default PIL font
        font = ImageFont.load_default()

    # Measure text
    bbox = draw.textbbox((0,0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position text in center of overlay
    text_x = (overlay.width - text_w) // 2
    text_y = (overlay.height - text_h) // 2
    draw.text((text_x, text_y), text, font=font,
              fill=(color[2], color[1], color[0], int(255*opacity)))

    # Rotate overlay
    rotated_overlay = overlay.rotate(angle, expand=True)

    # Create a new transparent canvas same size as base image
    overlay_canvas = Image.new("RGBA", pil_img.size, (255,255,255,0))

    # Center the rotated overlay on the canvas
    ox = (overlay_canvas.width - rotated_overlay.width) // 2
    oy = (overlay_canvas.height - rotated_overlay.height) // 2
    overlay_canvas.paste(rotated_overlay, (ox, oy), rotated_overlay)
    
    # Composite watermark onto image
    combined = Image.alpha_composite(pil_img, overlay_canvas)

    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)

def images_to_grid_tmp(input_folder, output_image, 
                   theme="luxury",
                   border_thickness=20, 
                   border_padding=20,
                   default_image_path=None, 
                   random_borders=True,
                   watermark_text="My Collage",
                   watermark_opacity=0.2,
                   font_path="simhei.ttf",   # Path to a font that supports Chinese
                   font_size=300,
                   font_color=(200, 200, 200),
                   cell_shape="rectangle",
                   watermark_angle=45):
    """
    Combine all images in a folder into one big artistic grid image with classic frames, watermark, and shape options.
    
    Parameters:
        input_folder (str): Folder containing images.
        output_image (str): Path to save the final grid image.
        border_thickness (int): Thickness of the decorative frame.
        default_image_path (str): Path to default filler image for empty cells.
        random_borders (bool): If True, randomize border colors per image.
        watermark_text (str): Text to overlay as watermark background (supports Chinese).
        watermark_opacity (float): Transparency of watermark (0 to 1).
        font_path (str): Path to a .ttf/.ttc font file that supports Chinese.
        font_size (int): Size of watermark text.
        font_color (tuple): Color of watermark text in BGR.
        cell_shape (str): Shape of each cell image: "rectangle", "square", "oval", "round".
    """
    # Collect all image paths
    image_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=lambda x: float(os.path.basename(x).split('_')[1].replace('s.jpg','')))



    # Read all images
    images = [cv2.imread(img) for img in image_files if cv2.imread(img) is not None]

    if not images and default_image_path is None:
        print("❌ No images found and no default filler provided.")
        return

    # If no images, fill with default
    if not images and default_image_path:
        filler = cv2.imread(default_image_path)
        images = [filler]

    # Calculate average dimensions to use as cell size
    avg_h = int(sum(img.shape[0] for img in images) / len(images))
    avg_w = int(sum(img.shape[1] for img in images) / len(images))
    cell_size = (avg_w, avg_h)

    # Resize all images to same size
    images = [cv2.resize(img, cell_size) for img in images]

    # Determine best grid size (close to square)
    num_images = len(images)
    grid_cols = int(math.ceil(math.sqrt(num_images)))
    grid_rows = int(math.ceil(num_images / grid_cols))

    # Pad with default filler if grid not full
    if default_image_path:
        filler = cv2.resize(cv2.imread(default_image_path), cell_size)
        while len(images) < grid_cols * grid_rows:
            images.append(filler)

    border_color, background_color = resolve_theme(theme)

    # Apply frames and shapes
    # framed_images = [apply_shape(add_classic_frame(img), cell_shape) for img in images]
    framed_images = [
        apply_shape_with_gradient_border(img, cell_shape, border_thickness, border_padding, border_color=border_color)      #, (255,215,0))
        for img in images
    ]


    # Update cell size after framing
    cell_h = framed_images[0].shape[0]
    cell_w = framed_images[0].shape[1]

    # Create blank canvas
    grid_h = cell_h * grid_rows
    grid_w = cell_w * grid_cols
    
    grid_image = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)

    # Paste images into grid
    for idx, img in enumerate(framed_images):
        row = idx // grid_cols
        col = idx % grid_cols
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        grid_image[y1:y2, x1:x2] = img

    # Apply watermark
    grid_image = put_chinese_text(grid_image, watermark_text, font_path, font_size, font_color, watermark_opacity, watermark_angle)

    # Save final image
    cv2.imwrite(output_image, grid_image)
    print(f"✅ Artistic grid image with watermark saved as '{output_image}'")

def images_to_grid(
    input_folder,
    output_image,
    rows,
    cols,
    theme="luxury",
    cell_shape="diamond",
    cell_width=None,
    cell_height=None,
    border_thickness=20,
    padding=20,
    max_images=None,
    default_image_path=None,
    add_watermark=False,
    watermark_text="",
    watermark_angle=0,
    watermark_opacity=0.25,
    font_path="simhei.ttf",
    font_size=475
):
    # 1. Collect and sort images
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_paths = sorted(image_paths, key=second_sort_key)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    # 2. Load default image if provided
    default_img = None
    if default_image_path is not None and os.path.exists(default_image_path):
        default_img = cv2.imread(default_image_path)
        if default_img is None:
            print("Warning: default_image_path could not be read:", default_image_path)
            default_img = None

    # 3. Read images; fallback to default when unreadable
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            if default_img is not None:
                images.append(default_img.copy())
            else:
                continue
        else:
            images.append(img)

    if not images and default_img is None:
        print("No readable images and no valid default image.")
        return

    # 4. Determine cell size
    if cell_width is None or cell_height is None:
        base_img = images[0] if images else default_img
        h0, w0 = base_img.shape[:2]
        cell_width = w0
        cell_height = h0

    # 5. Fill up remaining grid slots with default image if needed
    capacity = rows * cols
    if len(images) < capacity and default_img is not None:
        while len(images) < capacity:
            images.append(default_img.copy())
    else:
        images = images[:capacity]

    if not images:
        print("No images to place into grid.")
        return

    # 6. Resize all images to cell size
    resized_images = [
        resize_to_fill(
            img,
            cell_width,
            cell_height,
            crop_mode="random",
            weights={"top": 6, "center": 3, "bottom": 1}
        )
        for img in images
    ]

    # 7. Resolve theme → border & background
    border_color, background_color = resolve_theme(theme)

    # 8. Create grid canvas with theme background color
    grid_h = rows * cell_height
    grid_w = cols * cell_width
    grid_image = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)

    # 9. Apply shape + border + background to each cell and place into grid
    for idx, img in enumerate(resized_images):
        r = idx // cols
        c = idx % cols

        shaped, mask, border_mask = apply_shape_with_gradient_border(
            img,
            shape=cell_shape,
            border_thickness=border_thickness,
            padding=padding,
            border_color=border_color
        )

        cell_final = apply_background_to_cell(
            shaped,
            mask,
            border_mask,
            background_color=background_color
        )

        y1 = r * cell_height
        x1 = c * cell_width
        grid_image[y1:y1 + cell_height, x1:x1 + cell_width] = cell_final

    # 10. Optional watermark on final grid
    if add_watermark and watermark_text:
        grid_image = add_text_watermark(
            grid_image,
            text=watermark_text,
            angle=watermark_angle,
            font_path=font_path,
            font_size=font_size,
            color=(255, 255, 255),
            opacity=watermark_opacity
        )

    # 11. Save result
    cv2.imwrite(output_image, grid_image)
    print("Saved grid to:", output_image)

# Example usage:
default_image = "Source/sunflower_gift_rz.jpg"      # renzhener_signature_1.jpg"
images_to_grid("frames_output", "grid_chinese_diamond.jpg",
               rows=10,
               cols=8,
               theme="romantic",   # luxury, clean, neon, romantic
               border_thickness=30,
               padding=30,
               default_image_path=default_image,
               add_watermark=True,
               watermark_text="认真儿2025精彩瞬间3",
               watermark_opacity=0.65,
               font_path="simhei.ttf",
               font_size=185,
               cell_shape="round",
               cell_height=300,
               cell_width=300,
               watermark_angle=60)



