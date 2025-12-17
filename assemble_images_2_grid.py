import cv2
import os
import math
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# def apply_shape_with_border(img, shape, border_thickness, border_color, padding=20):
#     h, w = img.shape[:2]

#     # Create mask with padding
#     mask = np.zeros((h, w), dtype=np.uint8)

#     if shape == "rectangle":
#         cv2.rectangle(mask, (padding, padding), (w-padding, h-padding), 255, -1)
#     elif shape == "square":
#         side = min(h, w) - 2*padding
#         y1 = (h - side) // 2
#         x1 = (w - side) // 2
#         cv2.rectangle(mask, (x1, y1), (x1+side, y1+side), 255, -1)
#     elif shape == "oval":
#         center = (w//2, h//2)
#         axes = (w//2 - padding, h//2 - padding)
#         cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
#     elif shape == "round":
#         radius = min(h, w)//2 - padding
#         center = (w//2, h//2)
#         cv2.circle(mask, center, radius, 255, -1)
#     else:
#         cv2.rectangle(mask, (padding, padding), (w-padding, h-padding), 255, -1)

#     # Border mask by dilating
#     kernel_size = border_thickness*2+1
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     border_mask = cv2.dilate(mask, kernel) - mask

#     # Apply mask
#     shaped_img = cv2.bitwise_and(img, img, mask=mask)

#     # Create border
#     border_img = np.zeros_like(img)
#     border_img[border_mask>0] = border_color

#     return cv2.add(shaped_img, border_img)

import cv2
import numpy as np
import random

def apply_shape_with_gradient_border(img, shape, border_thickness=20, padding=20, border_color=None):
    """
    Apply a shape mask with a gradient border to an image.

    Parameters:
        img (numpy.ndarray): Input image.
        shape (str): "rectangle", "square", "oval", "round".
        border_thickness (int): Thickness of the border.
        padding (int): Padding inside the cell to avoid overlap.
        border_color (tuple or None): Optional BGR color for border. 
                                      If None, random gradient colors are used.
    """
    h, w = img.shape[:2]

    # Create base mask for the shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(mask, (padding, padding), (w-padding, h-padding), 255, -1)
    elif shape == "square":
        side = min(h, w) - 2*padding
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        cv2.rectangle(mask, (x1, y1), (x1+side, y1+side), 255, -1)
    elif shape == "oval":
        center = (w//2, h//2)
        axes = (w//2 - padding, h//2 - padding)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    elif shape == "round":
        radius = min(h, w)//2 - padding
        center = (w//2, h//2)
        cv2.circle(mask, center, radius, 255, -1)
    else:
        cv2.rectangle(mask, (padding, padding), (w-padding, h-padding), 255, -1)

    # Dilate mask to create border region
    kernel_size = border_thickness*2+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    outer_mask = cv2.dilate(mask, kernel)
    border_mask = outer_mask - mask

    # Decide border colors
    if border_color is not None:
        # Use provided color with a gradient to black
        color1 = np.array(border_color, dtype=np.uint8)
        color2 = np.array([0, 0, 0], dtype=np.uint8)
    else:
        # Randomize two colors for gradient
        color1 = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.uint8)
        color2 = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.uint8)

    # Create gradient across border thickness
    dist_transform = cv2.distanceTransform(border_mask, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    gradient = np.zeros_like(img)
    for c in range(3):
        gradient[:,:,c] = (color1[c]*(1-dist_norm) + color2[c]*dist_norm).astype(np.uint8)

    # Apply gradient only on border
    border_img = cv2.bitwise_and(gradient, gradient, mask=border_mask)

    # Apply shape mask to image
    shaped_img = cv2.bitwise_and(img, img, mask=mask)

    # Combine image + gradient border
    result = cv2.add(shaped_img, border_img)
    return result

def put_chinese_text(img, text, font_path, font_size, color, opacity, angle):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Create transparent overlay (same size as base image)
    overlay = Image.new("RGBA", img_pil.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    # Load font
    font = ImageFont.truetype(font_path, font_size)

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
    overlay_canvas = Image.new("RGBA", img_pil.size, (255,255,255,0))

    # Center the rotated overlay on the canvas
    ox = (overlay_canvas.width - rotated_overlay.width) // 2
    oy = (overlay_canvas.height - rotated_overlay.height) // 2
    overlay_canvas.paste(rotated_overlay, (ox, oy), rotated_overlay)

    # Composite watermark
    img_pil = Image.alpha_composite(img_pil, overlay_canvas)

    # Convert back to OpenCV image
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def images_to_grid(input_folder, output_image, 
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

    # Function to add classic oil-painting style frame
    def add_classic_frame(img):
        # Outer black frame
        framed = cv2.copyMakeBorder(
            img, border_thickness, border_thickness, border_thickness, border_thickness,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        # Inner decorative layer
        inner_thickness = border_thickness // 2
        color = (255, 215, 0) if not random_borders else (
            random.randint(100,255), random.randint(100,255), random.randint(100,255)
        )
        framed = cv2.copyMakeBorder(
            framed, inner_thickness, inner_thickness, inner_thickness, inner_thickness,
            cv2.BORDER_CONSTANT,
            value=color
        )
        return framed

    # Function to apply shape mask
    def apply_shape(img, shape):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if shape == "rectangle":
            mask[:] = 255
        elif shape == "square":
            side = min(h, w)
            y1 = (h - side) // 2
            x1 = (w - side) // 2
            mask[y1:y1+side, x1:x1+side] = 255
        elif shape == "oval":
            center = (w//2, h//2)
            axes = (w//2, h//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        elif shape == "round":
            radius = min(h, w) // 2
            center = (w//2, h//2)
            cv2.circle(mask, center, radius, 255, -1)
        else:
            mask[:] = 255  # fallback rectangle

        # Apply mask
        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    # Apply frames and shapes
    # framed_images = [apply_shape(add_classic_frame(img), cell_shape) for img in images]
    framed_images = [
        apply_shape_with_gradient_border(img, cell_shape, border_thickness, border_padding)      #, (255,215,0))
        for img in images
    ]


    # Update cell size after framing
    cell_h = framed_images[0].shape[0]
    cell_w = framed_images[0].shape[1]

    # Create blank canvas
    grid_h = cell_h * grid_rows
    grid_w = cell_w * grid_cols
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

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

# Example usage:


# Example usage:
default_image = "Source/renzhener_signature_1.jpg"
images_to_grid("frames_output", "grid_chinese.jpg",
               border_thickness=50,
               border_padding=50,
               default_image_path=default_image,
               random_borders=True,
               watermark_text="认真儿2025精彩瞬间",
               watermark_opacity=0.75,
               font_path="simhei.ttf",
               font_size=475,
               cell_shape="oval",
               watermark_angle=45)



