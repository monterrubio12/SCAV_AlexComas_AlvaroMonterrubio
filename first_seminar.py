#No usar FFMPEG porque es libreria experimental, hacerlo con import OS y subprocess

from PIL import Image
#FFMPEG UTILS 
import os
import subprocess
#DCT UTILS
import numpy as np
from scipy.fft import dct, idct
import pywt
import sys

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

        img = Image.open(input)
        w,h = img.size
        file = open(input, 'rb')
        data = file.read()
        serp_data = []
        
        for i in range(h):
            start = i*w #en cada fila avanzo un punto horizontalment par hacer diagonal
            end = start + w
            if(i%2 != 0):
                for j in range(w - 1, -1, -1):
                    serp_data.append(data[i* w + j])
            else:
                for j in range(w):
                    serp_data.append(data[i* w + j])
        
        return bytes(serp_data)


    #EXERCISE 5
    def bw_converter(self,input,output):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", "format=gray", output], capture_output=True, text=True)


class dct_utils:
    
    #EXERCISE 6
    def dct_converter(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    def dft_decoder(self, a):
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

exercise = Exercises()
exercise.resize("mbappe.jpg", "mbappe_resized.jpg", 300, 300)
exercise.bw_converter("mbappe.jpg", "mbappe_bw.jpg")
serp = exercise.serpentine("mbappe.jpg")
print(serp)


#TESTS
exercises = Exercises()

# Probar RGB a YUV
print("Prueba RGB -> YUV:")
r, g, b = 255, 0, 0  # Rojo puro
Y, U, V = exercises.RGBtoYUV(r, g, b)
print(f"RGB({r}, {g}, {b}) -> YUV({Y:.2f}, {U:.2f}, {V:.2f})")

# Valores esperados para YUV
expected_Y, expected_U, expected_V = 81.48, 90.44, 240.57

# Calcular y mostrar desviación para YUV
print("Desviaciones (YUV):")
print(f"Desviación en Y: {Y - expected_Y:.3f}")
print(f"Desviación en U: {U - expected_U:.3f}")
print(f"Desviación en V: {V - expected_V:.3f}\n")

# Probar YUV a RGB
print("Prueba YUV -> RGB:")
r_out, g_out, b_out = exercises.YUVtoRGB(expected_Y, expected_U, expected_V)
print(f"YUV({expected_Y}, {expected_U}, {expected_V}) -> RGB({r_out:.2f}, {g_out:.2f}, {b_out:.2f})")

# Valores esperados para RGB
expected_r, expected_g, expected_b = 255, 0, 0

# Calcular y mostrar desviación para RGB
print("Desviaciones (RGB):")
print(f"Desviación en R: {r_out - expected_r:.3f}")
print(f"Desviación en G: {g_out - expected_g:.3f}")
print(f"Desviación en B: {b_out - expected_b:.3f}")

# Forzar la salida a la terminal de inmediato
sys.stdout.flush()