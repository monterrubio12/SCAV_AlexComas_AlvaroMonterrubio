#No usar FFMPEG porque es libreria experimental, hacerlo con import OS y subprocess

#FFMPEG UTILS 
import os
import subprocess
#DCT UTILS
import numpy as np
from scipy.fft import dct, idct
import pywt

class Exercises:
    def RGBtoYUV(self, r, g, b):
        Y = 0.257 * r + 0.504 * g + 0.098 * b + 16
        U = -0.148 * r - 0.291 * g + 0.439 * b + 128
        V = 0.439 * r - 0.368 * g - 0.071 * b + 128
        return Y, U, V

    def YUVtoRGB(self, Y, U, V):
        R = 1.164 * (Y - 16) + 2.018 * (U - 128)
        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
        B = 1.164 * (Y - 16) + 1.596 * (V - 128)
        return R, G, B

    def resize(self, input, output, w, h):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", f"scale={w}:{h}", output],capture_output=True,text=True)
    
    def bw_converter(self,input,output,):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", "format=gray", output], capture_output=True, text=True)


class dct_utils:
    def __init__(self, type=2, norm='ortho'):
        self.type = type
        self.norm = norm

    def convert(self, input_data):
        return dct(input_data, type=self.type, norm=self.norm, axis=-1)
    
    def decode(self, transformed_data):
        return idct(transformed_data, type=self.type, norm=self.norm, axis=-1)

class dwt_utils:
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


#TESTS

# Llamamos a RGBtoYUV y verificamos si da algún error
try:
    r, g, b = 255, 0, 0
    Y, U, V = exercise.RGBtoYUV(r, g, b)
    print(f"RGBtoYUV Result: Y={Y}, U={U}, V={V}")
except Exception as e:
    print(f"Error en RGBtoYUV: {e}")

# Llamamos a YUVtoRGB y verificamos si da algún error
try:
    Y, U, V = 76.25, 84.0, 255.0
    r, g, b = exercise.YUVtoRGB(Y, U, V)
    print(f"YUVtoRGB Result: R={r}, G={g}, B={b}")
except Exception as e:
    print(f"Error en YUVtoRGB: {e}")

# Llamamos a resize y verificamos si da algún error
try:
    exercise.resize("input.jpg", "output_resized.jpg", 300, 300)
    print("Resize funcionó correctamente")
except Exception as e:
    print(f"Error en resize: {e}")

# Llamamos a bw_converter y verificamos si da algún error
try:
    exercise.bw_converter("input.jpg", "output_bw.jpg")
    print("BW Converter funcionó correctamente")
except Exception as e:
    print(f"Error en bw_converter: {e}")


# Instanciamos dct_utils y llamamos a su método convert
dct_util = dct_utils()
try:
    input_data = [1, 2, 3, 4, 5]
    transformed_data = dct_util.convert(input_data)
    print(f"DCT Transform Result: {transformed_data}")
except Exception as e:
    print(f"Error en DCT convert: {e}")

# Llamamos a decode en dct_utils
try:
    transformed_data = [1, 2, 3, 4, 5]
    decoded_data = dct_util.decode(transformed_data)
    print(f"DCT Inverse Result: {decoded_data}")
except Exception as e:
    print(f"Error en DCT decode: {e}")


# Instanciamos dwt_utils y llamamos a su método transform
dwt_util = dwt_utils()
try:
    data = [[1, 2], [3, 4]]  # Ejemplo simple de datos
    coeffs = dwt_util.transform(data)
    print(f"DWT Transform Result: {coeffs}")
except Exception as e:
    print(f"Error en DWT transform: {e}")

# Llamamos a inverse_transform en dwt_utils
try:
    coeffs = [np.array([[1, 2], [3, 4]])]  # Ejemplo de coeficientes
    reconstructed_data = dwt_util.inverse_transform(coeffs)
    print(f"DWT Inverse Result: {reconstructed_data}")
except Exception as e:
    print(f"Error en DWT inverse_transform: {e}")