# SCAV_AlexComas_AlvaroMonterrubio
This is the repository where we will be creating a software for SCAV practices at UPF.

## Seminar_1
During this first seminar on video coding in the SCAV course, we will implement a script named `first_seminar` that includes various classes and methods for processing images using the __ffmpeg__ software, along with other libraries for image and data processing such as scipy, numpy, and pillow. In this seminar, we will not only work with image manipulation we also deal with some of the core principles of image and video encoding and compression algorithms, essential tools for media transmission and storage.

### S1 - Exercise 1
In the first exercise of this first seminar, we were asked to compile and install the lastest version of __ffmpeg__ on our laptop/desktop (or any other device) using the command line. We needed to ensure that all necessary recent libraries were installed as part of the process. After the installation, in order to prove that the ffmpeg was succesfully installed, we were asked to run in our terminal the comand `ffmpeg` and upload a screenshot of the result. The screenshot that we obtained is the next one:

![Screenshot Ex1](https://github.com/monterrubio12/SCAV_AlexComas_AlvaroMonterrubio/blob/main/Seminar_1/unnamed-3.png)


### S1 - Exercise 2
After __ffmpeg__ was succesfully installed, in the second exercise, we were asked to start implementing some code. We have created a script named `Seminar_1/first_seminar.py`. Inside this script, we have created a class called `Exercises`, where we are going to include the different subclasses and methods required during this first seminar. For this second exercise, we were tasked with implementing a method which translates 3 values in RGB into the 3 YUV values, and another method to perform the opposite operation. We named this methods `RGBtoYUV()` and `YUVtoRGB()`, respectively. In our implementation, we have choosen to pass the three color values as arguments to our methods. The implementation of our functions is as follows:

```python
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
```
### S1 - Exercise 3
In the third exercise of the seminar, we were asked to extend the class created in the previous exercise by adding a new method. This method was designed to resize an image into a lower quality by means of __ffmpeg__. We have named this method `resize()`. In order to execute this method, we have used the `subprocess` module to run ffmpeg from our local terminal while executing the script from our IDE. The implementation of the method is as follows:

```python
def resize(self, input, output, w, h):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", f"scale={w}:{h}", output],capture_output=True,text=True)
```
To test this method, we have used the image `Seminar_1\mbappe.jpg`. The tests have been implemented in the final exercise of the seminar, so the result after runing this method will be presented and explained in the section __S1 - Exercise 8__.

### S1 - Exercise 4
For this exercise, we were asked to create a method called `serpentine()` that reads the componenets of a matrix in a "serpentine" pattern, where each row of pixels is read alternately from left to right and right to left, creating a zigzag effect. This method transforms the input matrix data into a format that simulates this serpentine reading pattern. To achieve the desired result, we first determine the dimensions of our input matrix (width and height), and after this, we use a combination of diferent loops to distinguish betwen even and odd diagonals to apply the zigzag effect when reading. The implementation of the method is as follows:

```python
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
```
The tests for this exercise have been implemented in the final exercise of the seminar.


### S1 - Exercise 5.1
In the first part of the fifth exercise of the seminar, we were asked to extend again the class created in the Exercise 2 by adding a new method. This method was designed to transform an image to black and white by means of __ffmpeg__. We have named this method `bw_converter()`. In order to execute this method, we have used again the `subprocess` module to run ffmpeg from our local terminal while executing the script from our IDE. The implementation of the method is as follows:

```python
def bw_converter(self,input,output):
        result = subprocess.run(["ffmpeg", "-i", input, "-vf", "format=gray", output], capture_output=True, text=True)
```
To test this method, we have used again the image `Seminar_1\mbappe.jpg`. Remember taht the tests have been implemented in the final exercise of the seminar, so the result after runing this method on our input image will be presented and explained in the section __S1 - Exercise 8__.

### S1 - Exercise 5.2
In the second part of the fifth exercise of the seminar, we were asked to extend again the class created in the Exercise 2 by adding a new method. In this case the method was designed to perform run-lenght encoding on a given series of bytes. We have named this method `run_length_encode()`. For this method we stop using `subproces` module, cause we don't need __ffmpeg__. Instead, we perform the encoding by means of the Python's grupby to compress the byte series and return the encoded list. The implementation of the method is as follows:

```python
def run_length_encode(self,data: List[int]) -> Iterator[Tuple[int, int]]:
        return ((x, sum(1 for _ in y)) for x, y in groupby(data))
```
In this case the tests have also been implemented in the final exercise of the seminar.

### S1 - Exercise 6
In the sixth exercise of the seminar, we were asked to create a new class named `dct_utils`, which would implement the necessary methods to perform an encoding and decoding of a signal using Discrete Cosine Transform (DCT) operators. Inside this class, we have implemented the method `dct_converter()`to encode the signal, and the method `dct_decoder()`, to decode our encoded signals back to their original form. The implementation of the method is as follows:

```python
class dct_utils:
    
    #EXERCISE 6
    def dct_converter(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    def dct_decoder(self, a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')

```
Again, the tests for this exercise have been implemented in the final exercise of the seminar.

### S1 - Exercise 7
In the seventh exercise of the seminar, we were asked to perform a task similar to the previous one. We had to create a new class named `dwt_utils`, which implements the necessary methods for encoding and decoding a signal using Discrete Wavelet Transform (DWT). To do it we have used the `pywt` library. Inside this class, we have implemented the method ` __init__()` that is the constructor used to initialize the wavelet type and the decomposition level. Also we have implemented the `transform()` method, to encode our signal using DWT, and the `inverse_transform()` method to decode the encoded signal back to its original form. The implementation of the method is as follows:

```python
class dwt_utils:

    #EXERCISE 7
    def __init__(self, wavelet='haar', level=1):
        self.wavelet = wavelet
        self.level = level
    
    def transform(self, data):
        return pywt.wavedec2(data, self.wavelet, level=self.level)
    
    def inverse_transform(self, coeffs):
        return pywt.waverec2(coeffs, self.wavelet)
```
The tests for this exercise have been also implemented in the final exercise of the seminar.

### S1 - Exercise 8
Finaly, in the last exercise we were asked to cerate unit tests to our code, for each method and class.

For the __Exercise 2__ we have performed a first test consisting on converting values from RGB to YUV, and also a second test consisting on converting values from YUV to RGB. The implementation of the test is as follows:

```python
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
```
The obtained output after runing the previous test using the methods implemented in Exercise 2 is the next one

```
Prueba RGB -> YUV:
RGB(255, 0, 0) -> YUV(81.53, 90.26, 239.94)
Desviaciones (YUV):
Desviación en Y: 0.055
Desviación en U: -0.180
Desviación en V: -0.625

Prueba YUV -> RGB:
YUV(81.48, 90.44, 240.57) -> RGB(255.88, -0.61, 0.42)
Desviaciones (RGB):
Desviación en R: 0.880
Desviación en G: -0.615
Desviación en B: 0.423
```

For the __Exercise 3__ we have performed a test consisting on resizing the image `Seminar_1\mbappe.jpg` , the result after performing this test is stored as `Seminar_1\mbappe_resized.jpg` and is the next one:

![TEST ex 3, resized image](https://github.com/monterrubio12/SCAV_AlexComas_AlvaroMonterrubio/blob/75a6a225929ff313138788b3db8fa41af26ca6cd/Seminar_1/mbappe_resized.jpg)

For the __Exercise 4__ we have performed a test to read the following matrix in a serpentine pattern.
```
[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]
```  
To achive this we have used the implemented `serpentine()` method. The result after performing this test is `[[1], [2, 5], [9, 6, 3], [4, 7, 10, 13], [14, 11, 8], [12, 15], [16]]`, where we can observe that the matrix has been read following this zigzag pattern, alternating the direction of each diagonal. In order to perform this test we have implmented the next code:

```python
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

print("Input Matrix:")
for i in matrix:
    print(i)

results = exercises.serpentine(matrix)

print("\nSerpentine reading output:")
print(results)
```

For the __Exercise 5.1__ we have performed a test consisting on transforming the image `Seminar_1\mbappe.jpg` to black and white, the result after performing this test is stored as `Seminar_1\mbappe_bw.jpg` and is the next one:

![TEST ex 5.1, black and white image](https://github.com/monterrubio12/SCAV_AlexComas_AlvaroMonterrubio/blob/2748a33599088e10138ca5b1a59571b14633d13f/Seminar_1/mbappe_bw.jpg)

For the __Exercise 5.2__ we have performed a test consisting on encoding an array `[1, 1, 3, 3, 4, 4, 5, 6]` using the Run Length Encoder, the result after performing this test is `[(1, 2), (3, 2), (4, 2), (5, 1), (6, 1)]` where the first component of each of the elements in the encoded array is each of the values of the input array, and the second component of each of the elements is the number of times that the value apears in the input array. In order to perform this test we have implmented the next code:

```python
aux = [1,1,3,3,4,4,5,6]
encoded = list(exercises.run_length_encode(aux))
print("Input Array:", aux)
print("Run Length Encoded Array: ", encoded)
```

For the __Exercise 6__ we have performed a test consisting on encoding an array `[[[1. 2. 3. 4. 5. 6.]]` using the DCT encoder, and after this decoding the encoded arary to compare if the obtained result is equal to the input array. The result after performing the encoding is `[[ 8.57321410e+00, -4.16256180e+00, -4.44089210e-16, -4.08248290e-01, -2.56395025e-16, -8.00788912e-02]]`. And the result after decoding the encoded signal is `[[1. 2. 3. 4. 5. 6.]]` that is exactly the input array. In order to perform this test we have implmented the next code:

```python
utils = dct_utils()
input_data = np.array([[1, 2, 3, 4, 5, 6]], dtype=float)
print("Input array:")
print(input_data)

dct_encoded = utils.dct_converter(input_data)
print("\nDCT encoded output:")
print(dct_encoded)

decoded_output = utils.dct_decoder(dct_encoded)
print("\nDecoded output (after applying IDCT):")
print(decoded_output)

# Verify if the decoded result matches with the input data.
if np.allclose(decoded_output, input_data, atol=1e-6):
    print("\nTest passed: Decoded output matches the original input.")
else:
    print("\nTest failed: Decoded output does not match the original input.")

```

Finaly for the __Exercise 7__ we have performed a test consisting on encoding an array `[[1. 2. 3. 4.],[5. 6. 7. 8.]]` using the DWT encoder, and after this decoding the encoded arary to compare if the obtained result is equal to the input array. The result after performing the encoding is `[array([[ 7., 11.]]), (array([[-4., -4.]]), array([[-1., -1.]]), array([[0.0000000e+00, 4.4408921e-16]]))]`. And the result after decoding the encoded signal is `[[1. 2. 3. 4.],[5. 6. 7. 8.]]` that is exactly the input array. In order to perform this test we have implmented the next code:

```python
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
```

## Practice_1
During this first practice on video coding in the SCAV course, we are going to create an API and containerizing it using Docker. For this task, we will use FastAPI that is a modern web framework for building APIs with Python. We will build a API serving for several purposes, and we will implement some endpoinds that will be used to process some actions of the first seminar. These endpoints will be designed to process video data, perform encoding or decoding tasks, and interact with other components introduced during the first seminar.

By the end of this practice, we will have a Dockerized environment ensuring that the application runs smoothly across different platforms. This will allow us to deploy the API in a consistent and isolated environment, making it easier to manage dependencies and facilitate collaboration.

### P1 - Exercise 1
In the first exercise of this practice we had to crate the API and put it inside a Docker. To do this, we first of all create a 

### P1 - Exercise 2

### P1 - Exercise 3
In this third exercise, we were asked to integrate all of our previous work into the new API. To adapt our unit tests for this new setup, we were allowed to use the help of any AI tools. As part of the integration process, we created two main files inside the `Practice_1 directory`.

The first file is `first_practice.py`. This file contains all the library imports and class definitions from the first seminar, which we will use to run our unit tests. These classes and methods were originally implemented to perform various tasks such as color space conversion (RGB to YUV and vice versa), image resizing, matrix transformations, and other utilities. The `first_practice.py` file serves to integrate this existing functionality into our new API, allowing us to call these functions using HTTP requests.

The implementation of this first file is as follows:

```python
from PIL import Image
from typing import Iterator, Tuple, List
from itertools import groupby
import os
import subprocess
import numpy as np
from scipy.fft import dct, idct
import pywt

class Exercises:
    # Conversion RGB <-> YUV
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

    # Redimensionar un video usando FFMPEG
    def resize(self, input, output, w, h):
        result = subprocess.run(
            ["ffmpeg", "-i", input, "-vf", f"scale={w}:{h}", output],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG error: {result.stderr}")

    # Lectura en serpentina de matrices
    def serpentine(self, input):
        serp_data = []
        h = len(input)
        w = len(input[0])

        for i in range(h):
            row, col = i, 0
            diagonal = []
            while row >= 0 and col < w:
                diagonal.append(input[row][col])
                row -= 1
                col += 1
            if i % 2 == 1:
                diagonal.reverse()
            serp_data.append(diagonal)

        for j in range(1, w):
            row, col = h - 1, j
            diagonal = []
            while row >= 0 and col < w:
                diagonal.append(input[row][col])
                row -= 1
                col += 1
            if (h + j - 1) % 2 == 1:
                diagonal.reverse()
            serp_data.append(diagonal)

        return serp_data

    # Conversión a blanco y negro usando FFMPEG
    def bw_converter(self, input, output):
        result = subprocess.run(
            ["ffmpeg", "-i", input, "-vf", "format=gray", output],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG error: {result.stderr}")

    # Codificación Run-Length
    def run_length_encode(self, data: List[int]) -> Iterator[Tuple[int, int]]:
        return ((x, sum(1 for _ in y)) for x, y in groupby(data))

class dct_utils:
    def dct_converter(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    def dct_decoder(self, a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')

class dwt_utils:
    def __init__(self, wavelet='haar', level=1):
        self.wavelet = wavelet
        self.level = level
    
    def transform(self, data):
        return pywt.wavedec2(data, self.wavelet, level=self.level)
    
    def inverse_transform(self, coeffs):
        return pywt.waverec2(coeffs, self.wavelet)

```
The second file is `test_main.py`. In this file, we first of all import the classes and functions defined in the `first_practice.py` file. This allows us to reuse the existing code from our previous work in the new API. After the necessary imports, we initialize the FastAPI application, which acts as the foundation for our API.

In this file, we also define the various endpoints required to run the different unit tests. Each endpoint is associated with a specific function from the `first_practice.py` file, which we test through HTTP requests. This design allows us to interact with the methods already implemented (RGB to YUV conversion, image resizing, matrix transformations, etc) and verify that they work correctly through the API.

By running these tests through the API, we ensure that each function performs as expected when called remotely. This also facilitates easy testing and allows us to have a more organized and manageable code.

The code we have implemented is as follows:

```python
rom fastapi import FastAPI
from first_practice import Exercises, dct_utils, dwt_utils
import numpy as np

app = FastAPI()
exercise = Exercises()
dct_utils_instance = dct_utils()
dwt_utils_instance = dwt_utils()

@app.get("/")
async def root():
    return {"message": "Testing FastAPI Endpoints"}

@app.get("/test_rgb_to_yuv/")
async def test_rgb_to_yuv():
    r, g, b = 255, 0, 0  # Red
    Y, U, V = exercise.RGBtoYUV(r, g, b)
    expected_Y, expected_U, expected_V = 81.48, 90.44, 240.57
    
    deviations = {
        "Y_deviation": Y - expected_Y,
        "U_deviation": U - expected_U,
        "V_deviation": V - expected_V
    }
    
    return {
        "input_rgb": {"r": r, "g": g, "b": b},
        "output_yuv": {"Y": round(Y, 2), "U": round(U, 2), "V": round(V, 2)},
        "deviations": deviations
    }

@app.get("/test_yuv_to_rgb/")
async def test_yuv_to_rgb():
    Y, U, V = 81.48, 90.44, 240.57  # Expected YUV for Red
    r, g, b = exercise.YUVtoRGB(Y, U, V)
    expected_r, expected_g, expected_b = 255, 0, 0
    
    deviations = {
        "R_deviation": r - expected_r,
        "G_deviation": g - expected_g,
        "B_deviation": b - expected_b
    }

    return {
        "input_yuv": {"Y": Y, "U": U, "V": V},
        "output_rgb": {"r": round(r), "g": round(g), "b": round(b)},
        "deviations": deviations
    }

@app.get("/test_serpentine/")
async def test_serpentine():
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    result = exercise.serpentine(matrix)
    return {
        "input_matrix": matrix,
        "serpentine_output": result
    }

@app.get("/test_resize/")
async def test_resize():
    try:
        exercise.resize("../Practice_1/mbappe.jpg", "../Practice_1/mbappe_resized.jpg", 300, 300)
        return {"message": "Resize test passed."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test_bw_converter/")
async def test_bw_converter():
    try:
        exercise.bw_converter("../Practice_1/mbappe.jpg", "../Practice_1/mbappe_bw.jpg")
        return {"message": "Black & White conversion test passed."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test_run_length_encoding/")
async def test_run_length_encoding():
    aux = [1, 1, 3, 3, 4, 4, 5, 6]
    encoded = list(exercise.run_length_encode(aux))
    return {
        "input_array": aux,
        "encoded_array": encoded
    }

@app.get("/test_dct_encoding/")
async def test_dct_encoding():
    input_data = np.array([[1, 2, 3, 4, 5, 6]], dtype=float)
    dct_encoded = dct_utils_instance.dct_converter(input_data)
    decoded_output = dct_utils_instance.dct_decoder(dct_encoded)
    
    passed = np.allclose(decoded_output, input_data, atol=1e-6)
    
    return {
        "input_array": input_data.tolist(),
        "dct_encoded_output": dct_encoded.tolist(),
        "decoded_output": decoded_output.tolist(),
        "test_passed": passed
    }

@app.get("/test_dwt_encoding/")
async def test_dwt_encoding():
    input_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
    transformed_data = dwt_utils_instance.transform(input_data)
    reconstructed_data = dwt_utils_instance.inverse_transform(transformed_data)
    
    passed = np.allclose(reconstructed_data, input_data, atol=1e-6)
    
    return {
        "input_array": input_data.tolist(),
        "transformed_data": transformed_data,
        "reconstructed_data": reconstructed_data.tolist(),
        "test_passed": passed
    }

```
### P1 - Exercise 4
In the fourth exercise of this practice, we were asked to create at least two endpoints which will process some actions from the previous S1. As we have already explained, in exrecise 3, we have implmented the required endpoints to run the unit tests for our API. These tests were defined in the `test_main.py`` file. In order to ensure that the tests works we need to link this file to the `main.py` file. This simplify all the structure of the application.

To achieve this, we imported the `test_main.py` file into `main.py`. Then, we created the FastAPI application and mounted the test functions to the API. This allows the API to use requests for running the unit tests and ensures that everything is properly connected. This linking between the two files, ensure a smooth conection from the API to the test functions, making it easier to execute and monitor the results of the tests.

The implementation of the main.py` to link it with `test_main.py` and run the endpoints is as follows:

```python
from fastapi import FastAPI
from test_main import app as test_app

app = FastAPI()
app.mount("/test", test_app)

@app.get("/")
async def root():
    return {"message": "Practice 1 by Alex Comas & Alvaro Monterrubio"}
```

Also, in the `main.py` file, we have included some instrucctions defining the endpoints that need to be tested and how to execute them. These instructions define the necessary routes to run the different unit tests that have been defined in the API. Each test corresponds to a specific functionality from the previous exercises and is accessible through a dedicated endpoint.

The following instructions are the routes defining the endpoints and the corresponding tests you can run:

```
#HACER TEST CON ENDPOINTS:

#Test RGB to YUV: /test/test_rgb_to_yuv/
##Test YUV to RGB: /test/test_yuv_to_rgb/
#Test Serpentine: /test/test_serpentine/
#Test Resize: /test/test_resize/
#Test Black & White Conversion: /test/test_bw_converter/
#Test Run-Length Encoding: /test/test_run_length_encoding/
#Test DCT Encoding: /test/test_dct_encoding/
#Test DWT Encoding: /test/test_dwt_encoding/
```

In order to answer in more detail we are going to include two of the endpoints implemnetations done in `test_main.py` file with some explanationto understan better how they work. 


