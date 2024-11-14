#No usar FFMPEG porque es libreria experimental, hacerlo con import OS y subprocess

from PIL import Image
from typing import Iterator, Tuple, List
from itertools import groupby

#FFMPEG UTILS 
import os
import subprocess
#DCT UTILS
import numpy as np
from scipy.fft import dct, idct
import pywt
import sys
#API


class Exercises:

    #EXERCISE 2
    def RGBtoYUV(self, r, g, b):
        Y = 0.257 * r + 0.504 * g + 0.098 * b + 16
        U = -0.148 * r - 0.291 * g + 0.439 * b + 128
        V = 0.439 * r - 0.368 * g - 0.071 * b + 128
        return Y, U, V

    def YUVtoRGB(self, Y, U, V):
        R = 1.164 * (Y - 16) + 1.596 * (V - 128)
        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
        B = 1.164 * (Y - 16) + 2.018 * (U - 128)
        return R, G, B
    
    #EXERCISE 3
    def resize(self, input, output, w, h):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", f"scale={w}:{h}", output],capture_output=True,text=True)
    
    #EXERCISE 4
    def serpentine(self, input):
        serp_data = []
        
        h = len(input)        #num_rows
        w = len(input[0])     #num_cols
        serp_data = []

        #first column starting diagonals
        for i in range(h):
            row, col = i, 0
            diagonal = []

            while row >= 0 and col < w:
                diagonal.append(input[row][col])
                row -= 1
                col += 1

            # Invert diagonal if is odd
            if i % 2 == 1:
                diagonal.reverse()
            
            serp_data.append(diagonal)

        # last row starting diagonals
        for j in range(1, w): #start at 1 to avoid repeating the main diagonal
            row, col = h - 1, j
            diagonal = []

            while row >= 0 and col < w:
                diagonal.append(input[row][col])
                row -= 1
                col += 1
            
            #Invert diagonal if is odd
            if (h + j - 1) % 2 == 1:
                diagonal.reverse()

            serp_data.append(diagonal)

        return serp_data


    #EXERCISE 5.1
    def bw_converter(self,input,output):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", "format=gray", output], capture_output=True, text=True)

    #EXERCISE 5.2
    def run_length_encode(self,data: List[int]) -> Iterator[Tuple[int, int]]:
        return ((x, sum(1 for _ in y)) for x, y in groupby(data))

class dct_utils:
    
    #EXERCISE 6
    def dct_converter(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    def dct_decoder(self, a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')


class dwt_utils:

    #EXERCISE 7
    def __init__(self, wavelet='haar', level=1):
        self.wavelet = wavelet
        self.level = level
    
    def transform(self, data):
        return pywt.wavedec2(data, self.wavelet, level=self.level)
    
    def inverse_transform(self, coeffs):
        return pywt.waverec2(coeffs, self.wavelet)

