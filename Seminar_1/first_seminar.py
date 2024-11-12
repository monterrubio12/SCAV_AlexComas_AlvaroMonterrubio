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




#EXERCISE 8, TESTS
exercises = Exercises()

# EX2: Probar RGB a YUV
print()
print("EXERCISE 2 TEST-----------------------------------------")
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

# EX3: Resize imagen
exercises.resize("Seminar_1/mbappe.jpg", "Seminar_1/mbappe_resized.jpg", 300, 300)


# EX4: Serpentine
print()
print("EXERCISE 4 TEST-----------------------------------------")


# EX5.1: Black & White
exercises.bw_converter("Seminar_1/mbappe.jpg", "Seminar_1/mbappe_bw.jpg")


# EX5.2: Run length encoder
print()
print("EXERCISE 5.2 TEST-----------------------------------------")
aux = [1,1,3,3,4,4,5,6]
encoded = list(exercises.run_length_encode(aux))
print("Input Array:", aux)
print("Run Length Encoded Array: ", encoded)


# EX6: DCT encoder-decoder
print()
print("EXERCISE 6 TEST-----------------------------------------")
utils = dct_utils()
input_data = np.array([[1, 2, 3, 4, 5, 6]], dtype=float)
print("Input array:")
print(input_data)

dct_encoded = utils.dct_converter(input_data)
print("\nDCT encoded output:")
print(dct_encoded)

decoded_output = utils.dft_decoder(dct_encoded)
print("\nDecoded output (after applying IDCT):")
print(decoded_output)

# Verify if the decoded result matches with the input data.
if np.allclose(decoded_output, input_data, atol=1e-6):
    print("\nTest passed: Decoded output matches the original input.")
else:
    print("\nTest failed: Decoded output does not match the original input.")


# EX7: DWT encoder-decoder
print()
print("EXERCISE 7 TEST-----------------------------------------")
dwtutils = dwt_utils()
input_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float) 
print("Input array:")
print(input_data)

transformed_data = dwtutils.transform(input_data)
print("\nDWT transformed data:")
print(transformed_data)

reconstructed_data = dwtutils.inverse_transform(transformed_data)
print("\nReconstructed data after applying inverse DWT:")
print(reconstructed_data)

# Verify if the reconstructed data matches with the input data.
if np.allclose(reconstructed_data, input_data, atol=1e-6):
    print("\nTest passed: Decoded output matches the original input.")
else:
    print("\nTest failed: Decoded output does not match the original input.")

# Forzar la salida a la terminal de inmediato
sys.stdout.flush()