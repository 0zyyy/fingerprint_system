import cv2
import numpy as np
from skimage.feature import hog
from config import HOG_CONFIG

class HOGExtractor:
    def __init__(self):
        self.cell_size = HOG_CONFIG['CELL_SIZE']
        self.block_size = HOG_CONFIG['BLOCK_SIZE']
        self.bins = HOG_CONFIG['BINS']
        
    def extract(self, image):
        resized = cv2.resize(image, HOG_CONFIG['RESIZE_DIM'])
        
        features = hog(
            resized,
            orientations=self.bins,
            pixels_per_cell=self.cell_size,
            cells_per_block=self.block_size,
            visualize=False,
            feature_vector=True
        )
        
        return features