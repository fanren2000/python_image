

import numpy as np
import cv2 as cv
import random

# Canvas setup
width, height = 900, 600
bg = np.full((height, width, 3), (255, 225, 95), dtype=np.uint8)  # warm ground


bird_colors = [
    (255, 0, 0), (255, 165, 0), (255, 255, 0),
    (0, 128, 255), (128, 0, 255), (0, 255, 255),
    (255, 105, 180)
]
trunk_colors = [(60, 40, 20), (80, 50, 30), (100, 60, 40)]
leaf_colors = [(0, 100, 0), (35, 180, 35), (20, 150, 60)]

def draw_tree(img, x, y, scale):
    # Trunk
    trunk_w = int(10 * scale)
    trunk_h = int(60 * scale)
    trunk_color = random.choice(trunk_colors)
    cv.rectangle(img, (x - trunk_w//2, y), (x + trunk_w//2, y + trunk_h), trunk_color, -1)

    # Branching logic
    branch_count = random.randint(2, 4)
    for i in range(branch_count):
        bx = x + random.randint(-20, 20)
        by = y - int(30 * scale) + i * int(15 * scale)
        angle = random.randint(-30, 30)
        length = random.randint(30, 50)
        x2 = int(bx + length * np.cos(np.radians(angle)))
        y2 = int(by - length * np.sin(np.radians(angle)))
        cv.line(img, (bx, by), (x2, y2), trunk_color, 2)

    # Irregular leaf clusters
    for i in range(random.randint(4, 6)):
        offset_x = x + random.randint(-30, 30)
        offset_y = y - random.randint(40, 80)
        r1 = random.randint(20, 35)
        r2 = random.randint(15, 30)
        angle = random.randint(0, 180)
        leaf_color = random.choice(leaf_colors)
        cv.ellipse(img, (offset_x, offset_y), (r1, r2), angle, 0, 360, leaf_color, -1)

def draw_bird(img, x, y, scale=2.0, flying=False):
    color = random.choice(bird_colors)

    # Body
    cv.ellipse(img, (x, y), (int(14*scale), int(10*scale)), 0, 0, 360, color, -1)

    # Head
    head_x = x + int(12 * scale)
    head_y = y - int(8 * scale)
    cv.circle(img, (head_x, head_y), int(6 * scale), color, -1)

    # Eye with highlight
    cv.circle(img, (head_x, head_y), int(3 * scale), (255, 255, 255), -1)
    cv.circle(img, (head_x, head_y), int(1.5 * scale), (0, 0, 0), -1)

    # Beak (forward-facing)
    beak_length = int(8 * scale)
    beak_width = int(3 * scale)
    beak_tip = (head_x + beak_length, head_y)
    beak_top = (head_x + beak_length - beak_width, head_y - beak_width)
    beak_bottom = (head_x + beak_length - beak_width, head_y + beak_width)
    beak_pts = np.array([beak_tip, beak_top, beak_bottom])
    cv.fillPoly(img, [beak_pts], (0, 165, 255))

    # Tail
    tail_pts = np.array([
        [x - int(14*scale), y],
        [x - int(20*scale), y - int(6*scale)],
        [x - int(20*scale), y + int(6*scale)]
    ])
    cv.fillPoly(img, [tail_pts], color)

    # Wings or legs
    if flying:
        wing_color = tuple(min(c + 40, 255) for c in color)
        cv.ellipse(img, (x, y - int(10*scale)), (int(10*scale), int(6*scale)), -45, 0, 360, wing_color, -1)
        cv.ellipse(img, (x, y + int(10*scale)), (int(10*scale), int(6*scale)), 45, 0, 360, wing_color, -1)
    else:
        cv.line(img, (x - int(3*scale), y + int(10*scale)), (x - int(3*scale), y + int(15*scale)), (0, 0, 0), 2)
        cv.line(img, (x + int(3*scale), y + int(10*scale)), (x + int(3*scale), y + int(15*scale)), (0, 0, 0), 2)


# Tree and bird parameters
n_trees = 20

# Draw trees and perched birds
for _ in range(n_trees):
    x = random.randint(50, width - 50)
    y = random.randint(250, height - 150)
    scale = random.uniform(0.8, 1.4)
    draw_tree(bg, x, y, scale)

    # Add 1–2 perched birds per tree
    for _ in range(random.randint(1, 2)):
        bx = x + random.randint(-25, 25)
        by = y - random.randint(20, 50)
        draw_bird(bg, bx, by, scale=random.uniform(1.8, 2.2), flying=False)

# Add flying birds
for _ in range(6):
    fx = random.randint(100, width - 100)
    fy = random.randint(50, 200)
    draw_bird(bg, fx, fy, scale=random.uniform(1.8, 2.2), flying=True)

# Display
# cv.imshow("Realistic Forest with Graceful Birds", bg)
# cv.waitKey(0)
# cv.destroyAllWindows()

cv.imwrite(f'{n_trees}_tree_bird.png', bg)
