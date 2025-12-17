#!/usr/bin/env python
# coding: utf-8

# ## Draw a Forest of Objects with OpenCV
# 
# <br>
# Author: Mariya Sha
# <br>
# YouTube Tutorial: https://youtu.be/-LsuiVGO-88


import numpy as np
import cv2 as cv
import random

# general parameters
width = 900
height = 600
n_trees = 100
ground_level = height-100

# blank image
bg = np.zeros((height, width, 3), dtype=np.uint8)

# draw background
cv.rectangle(bg,(width,0), (0, ground_level), (255,225,95), -1)

# ***********************
class Tree:
    def __init__(self, image):
        self.img = image
        self.loc = int(np.random.choice(range(width), 1))
        self.ht = int(np.random.choice(range(200,400), 1))
        self.radius = 50
        self.scale = np.random.choice(np.linspace(0.5,2, num=8), 1)

    def generate_colours(self):
        green = (0, random.randint(130,200),0)
        light_green = (35, random.randint(200,250),35)
        brown = random.choice([(2,30,85), (5,55,120), (0,70,140)])
        return green, light_green, brown

    def draw(self):
        small_radius = int(self.radius*self.scale-20*self.scale)
        green, light_green, brown = self.generate_colours()

        # leafs - shadows
        cv.circle(self.img, (self.loc,ground_level-self.ht), int(self.radius*self.scale), green, -1)
        cv.circle(self.img, (self.loc-int(45*self.scale),ground_level-self.ht+small_radius), small_radius , green, -1)
        cv.circle(self.img, (self.loc+int(45*self.scale),ground_level-self.ht+small_radius), small_radius, green, -1)    

        # trunk
        cv.line(self.img, (self.loc,ground_level),(self.loc,ground_level-self.ht), brown, int(20*self.scale))
        cv.line(self.img, (self.loc,ground_level-self.ht+int(75*self.scale)),(self.loc+int(45*self.scale),ground_level-self.ht+small_radius), brown, int(5*self.scale))
        cv.line(self.img, (self.loc,ground_level-self.ht+int(75*self.scale)),(self.loc-int(45*self.scale),ground_level-self.ht+small_radius), brown, int(5*self.scale))

        # leafs - highlights
        cv.circle(self.img, (self.loc,ground_level-self.ht), int(self.radius*self.scale-10*self.scale), light_green, -1)
        cv.circle(self.img, (self.loc-int(45*self.scale),ground_level-self.ht+small_radius), small_radius-int(10*self.scale) , light_green, -1)
        cv.circle(self.img, (self.loc+int(45*self.scale),ground_level-self.ht+small_radius), small_radius-int(10*self.scale), light_green, -1)    

        cv.rectangle(bg,(width, ground_level), (0, height), green, -1)
        return self.img
# ***********************

#display image
for i in range(n_trees):
    img = Tree(bg).draw()
cv.imshow('forest of objects', img) 

cv.waitKey(0)
cv.destroyAllWindows()





