import numpy as np
import cv2 as cv
import random

# Canvas setup
width, height = 900, 600

ground_level = height - 100
bg = np.zeros((height, width, 3), dtype=np.uint8)

# Background
cv.rectangle(bg, (width, 0), (0, ground_level), (255, 225, 95), -1)

# Bird colors
bird_colors = [
    (255, 0, 0), (255, 165, 0), (255, 255, 0),
    (0, 128, 255), (128, 0, 255), (0, 255, 255),
    (255, 105, 180)
]

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

    # Beak (forward-facing triangle)
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

# Original tree class restored
class Tree:
    def __init__(self, image):
        self.img = image
        self.loc = int(np.random.choice(range(width), 1))
        self.ht = int(np.random.choice(range(200, 400), 1))
        self.radius = 50
        self.scale = np.random.choice(np.linspace(0.5, 2, num=8), 1)

    def generate_colours(self):
        green = (0, random.randint(130, 200), 0)
        light_green = (35, random.randint(200, 250), 35)
        brown = random.choice([(2, 30, 85), (5, 55, 120), (0, 70, 140)])
        return green, light_green, brown

    def draw(self):
        small_radius = int(self.radius * self.scale - 20 * self.scale)
        green, light_green, brown = self.generate_colours()

        # Leafs - shadows
        cv.circle(self.img, (self.loc, ground_level - self.ht), int(self.radius * self.scale), green, -1)
        cv.circle(self.img, (self.loc - int(45 * self.scale), ground_level - self.ht + small_radius), small_radius, green, -1)
        cv.circle(self.img, (self.loc + int(45 * self.scale), ground_level - self.ht + small_radius), small_radius, green, -1)

        # Trunk
        cv.line(self.img, (self.loc, ground_level), (self.loc, ground_level - self.ht), brown, int(20 * self.scale))
        cv.line(self.img, (self.loc, ground_level - self.ht + int(75 * self.scale)),
                (self.loc + int(45 * self.scale), ground_level - self.ht + small_radius), brown, int(5 * self.scale))
        cv.line(self.img, (self.loc, ground_level - self.ht + int(75 * self.scale)),
                (self.loc - int(45 * self.scale), ground_level - self.ht + small_radius), brown, int(5 * self.scale))

        # Leafs - highlights
        cv.circle(self.img, (self.loc, ground_level - self.ht), int(self.radius * self.scale - 10 * self.scale), light_green, -1)
        cv.circle(self.img, (self.loc - int(45 * self.scale), ground_level - self.ht + small_radius),
                  small_radius - int(10 * self.scale), light_green, -1)
        cv.circle(self.img, (self.loc + int(45 * self.scale), ground_level - self.ht + small_radius),
                  small_radius - int(10 * self.scale), light_green, -1)

        # Ground overlay
        cv.rectangle(bg, (width, ground_level), (0, height), green, -1)

        # Add perched birds
        if random.random() < 0.4:
            draw_bird(self.img, self.loc, ground_level - self.ht - int(10 * self.scale), scale=2.0, flying=False)
        if random.random() < 0.3:
            draw_bird(self.img, self.loc + int(45 * self.scale), ground_level - self.ht + small_radius - int(10 * self.scale), scale=2.0, flying=False)
        if random.random() < 0.3:
            draw_bird(self.img, self.loc - int(45 * self.scale), ground_level - self.ht + small_radius - int(10 * self.scale), scale=2.0, flying=False)

        return self.img

# Parameters
n_trees = 25

# Draw forest
for i in range(n_trees):
    img = Tree(bg).draw()

# Add flying birds
for _ in range(6):
    fx = random.randint(100, width - 100)
    fy = random.randint(50, 200)
    draw_bird(bg, fx, fy, scale=2.0, flying=True)

# Save
cv.imwrite(f'{n_trees}_tree_bird.png', bg)

# Display
cv.imshow("Restored Forest with Graceful Birds", img)
cv.waitKey(0)
cv.destroyAllWindows()
