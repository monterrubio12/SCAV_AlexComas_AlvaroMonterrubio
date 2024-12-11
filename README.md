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
In the first exercise of this practice we had to crate the API and put it inside a Docker. To do this, we first of all create a main.py, which is the main core of the api, where we are going to define the calls and endpoints. We are using the FastAPI as API for this practice after seeing that could be the easiest option for us without knowldege of APIS.
Inside the main, we are importing the fastapi and Exercises, the class which contain all the exercises or methods from the previous seminar.
Then, from here, we create the "app", which is usually used inside the main file as the instance of FastAPI. After defining the app, we created an instance of the Exercises class, named exercise, which contains all the methods implemented in the previous seminar.

```from fastapi import FastAPI
from first_practice import Exercises

app = FastAPI()
exercise = Exercises()

@app.get("/")
async def root():
    return {"message": "Practice 1 by Alex Comas & Alvaro Monterrubio"}
```

Once the setup was ready, we started by defining the root endpoint ("/") using the @app.get decorator. This is a simple endpoint that returns a welcome message and provides basic information about the practice. Then, in the next exercises, we will define some more endpoints.
Then, we have created the requirements.txt and Dockerfile. Inside the requirements file, we list all the necessary Python dependencies for our API, and inside the coker file, we containerize the application, with the python version, directory, the requeirements previously mentioned, FFMPEG...

```FROM python:3.13

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### P1 - Exercise 2
As it's possible to observe in the previous lines of code, we have added some comands to run FFMPEG. With the RUN apt-get update, we ensure that we have the latest version of FFPMEG, and with the apt-get install -y, the docker process the action of installing the FFMPEG automatically.


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

In order to run the docker and API, we are going to use the following comands in the terminal:

```
docker run -d -p 8000:8000 practice1-app
uvicorn main:app --reload
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

## Seminar 2
For this second seminar, we will continue working with APIs and FFMPEG calls, focusing on the Big Buck Bunny video. Most (if not all) of the commands used in this seminar are based on the FFMPEG documentation.

We have created a new directory in our project named `Seminar_2`. Inside this directory, we have included the necessary Python files to run our API and the required endpoints. The file `second_seminar.py` contains the implementation of the functions for each of the exercises tasked in the seminar. The file `integration_tests.py` includes the integration tests using the Big Buck Bunny video to validate the functionality of the exercises. The file `main.py` specifies the routes for the new endpoints and instantiates `integration_tests.py`. Finally, in this delivery was not mandatory, but we decided to implement the `unit_tests.py` file, which contains unit tests for each of the exercises.

To run the integration tests, the project folder must be organized as follows: Create two subfolders inside `Seminar_2`: one named `input_file`, which should contain the Big Buck Bunny video in two formats: `bbb.mp4` and `bbb.mov`. The mp4 format is required for the mp4_reader function, while the mov format is used by the other functions. The second folder is named `output_file`, and it will store the output files generated by the FFMPEG calls.

This structure ensures that the input files are correctly organized and accessible for all tests and functions.

In order to run the unit tests you should run the next command from the terminal inside the `Seminar_2` folder:
```
python -m unittest unit_tests.py
```

The instructions to  run the docker and API, are the same ones as in the previous delivery but we include it here again:
```
docker run -d -p 8000:8000 practice1-app
uvicorn main:app --reload
```

### Endpoint routes
As we have already explained in the `main.py`file we include the routes to acces the different endpoints of our API. We put them here to eases the process if you want to run the code while reading the instructions.

```
#NUEVOS TESTS CON ENDPOINTS:

#Test resolution adaptor: /test/test_resolution_adaptor/?width=1280&height=720"
#Test chroma subsampling: /test/test_chroma_subsampling/?pix_fmt=yuv420p"
#Test get metadata: /test/test_get_metadata/"
#Test bbb_editor: /test/test_bbb_editor/"
#Test mp4 reader: /test/test_mp4_reader/"
#Test macro blocks: /test/test_video_macroblocks/"
#Test yuv histogram: /test/test_yuv_histogram/"
```


### S2 - Exercise 1
In this first exercise we were asked to download the Big Buck Bunny video and implement a new endpoint / feature which let us modify the resolution of the video. To do this we have used a FFMPEG comand:

```python
def resolution_adaptor(self, input_file, width, height, output_file):
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-vf", f"scale={width}:{height}", output_file],
            check=True
        )
        return output_file
```
Basically, given a new width and height, and an output and input file path, we are using a FFMPEG comand to resize the video.

We also include here the implementation of the integration test and the unit test for this first exercise:

```python
#Integration test resolution adaptor
@app.get("/test_resolution_adaptor/")
async def test_resolution_adaptor(width: int, height: int):
    input_path = f"../Seminar_2/input_file/bbb.mov"
    output_file = f"bbb_{width}x{height}.mp4"
    output_path = f"../Seminar_2/output_file/{output_file}"

    try:
        result = ffmpeg_utils.resolution_adaptor(input_path, width, height, output_path)
        return {"message": "Resolution adaptation successful", "output_file": result}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Error during resolution adaptation: {e}")


#Unit test resolution adaptor
@patch('subprocess.run')
def test_resolution_adaptor(self, mock_run):
        # Llama al método
        output = ffmpeg_utils_comas_alvaro.resolution_adaptor(self,"input.mp4", 1920, 1080, "output.mp4")
        
        # Verifica que subprocess.run fue llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            ["ffmpeg", "-i", "input.mp4", "-vf", "scale=1920:1080", "output.mp4"],
            check=True
        )
        
        # Verifica el valor devuelto
        self.assertEqual(output, "output.mp4")
```


### S2 - Exercise 2
For the second exercise, we are asked to modify the chroma subsampling. As in the previous exercise, given some inputs (input and ouput file path, and the new format), we return the new file with the chroma subsampling:

```python
def chroma_subsampling(self, input_file, output_file, pix_fmt):
        subprocess.run(
            #Pix format debe ser tipo yuv420, yuv422...
            ["ffmpeg", "-i", input_file, "-c:v", "libx264", "-pix_fmt", pix_fmt, output_file],
            check=True

        )
```
Again we include here the implementation of the integration test and the unit test for this second exercise:

```python
#Integration test chroma subsampling
@app.get("/test_chroma_subsampling/")
async def test_chroma_subsampling(pix_fmt: str):
    input_path = f"../Seminar_2/input_file/bbb.mov"
    output_file = f"bbb_{pix_fmt}.mp4"
    output_path = f"../Seminar_2/output_file/{output_file}"

    try:
        result = ffmpeg_utils.chroma_subsampling(input_path, output_path, pix_fmt)
        return {"message": "Chroma subsampling successful", "output_file": result}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Error during chroma subsampling: {e}")


#Unit test chroma subsampling
@patch('subprocess.run')
def test_chroma_subsampling(self, mock_run):
        # Llama al método
        output = ffmpeg_utils_comas_alvaro.chroma_subsampling(self, "input.mp4", "output.mp4", "yuv420p")
        
        # Verifica que subprocess.run fue llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            ["ffmpeg", "-i", "input.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "output.mp4"],
            check=True
        )
        
        # Verifica el valor devuelto
        self.assertEqual(output, "output.mp4")
```

### S2 - Exercise 3
For the third exercise, we are asked to create a new endpoint / feature which let us read the video info and print at least 5 relevant data from the video. We decided to use an FFMPEG command to generate a metadata.txt file containing this information. As in the previous exercise, given some inputs (input and ouput file path), we return the new txt. file with the metadata. The implementation of this exercise is as follows:

```python
def get_metadata(self, input_file, metadata_file):
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-f", "ffmetadata", metadata_file],
            check=True
        )
```
Again we include here the implementation of the integration test and the unit test for this third exercise:

```python
#Integration test get metadata
@app.get("/test_get_metadata/")
async def test_get_metadata():
    input_path = f"../Seminar_2/input_file/bbb.mov"
    metadata_file = "bbb_metadata.txt"
    metadata_path = f"../Seminar_2/output_file/{metadata_file}"

    try:
        ffmpeg_utils.get_metadata(input_path, metadata_path)
        return {"message": "Metadata extraction successful", "metadata_file": metadata_path}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Error during metadata extraction: {e}")


#Unit test get metadata
@patch('subprocess.run')
def test_get_metadata(self, mock_run):
        # Llama al método
        ffmpeg_utils_comas_alvaro.get_metadata(self,"input.mp4", "metadata.txt")
        
        # Verifica que subprocess.run fue llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            ["ffmpeg", "-i", "input.mp4", "-f", "ffmetadata", "metadata.txt"],
            check=True
        )
```

### S2 - Exercise 4
For this fourth exercise, we are going to cut and export the bbb video in some different formats. First of all, the function asks for a input and output file paths. Then, we start by defining the name of the files that we are going to create later and we start to run some commands for the asked questions:

1. Cut the 20 seconds clip: We use a FFMPEG comand to to this and store it as mp4 file
```python
subprocess.run(["ffmpeg", "-i", input_file, "-ss", "00:00:00", "-t", "20", "-c:v", "copy", "-c:a", "copy", video_20s], check=True)
```

2. We export the audio tracks in some different formats:
```python
            subprocess.run(["ffmpeg", "-i", video_20s, "-ac", "1", "-c:a", "aac", audio_aac], check=True)
            subprocess.run(["ffmpeg", "-i", video_20s, "-ac", "2", "-c:a", "libmp3lame", "-b:a", "128k", audio_mp3], check=True)
            subprocess.run(["ffmpeg", "-i", video_20s, "-c:a", "ac3", audio_ac3], check=True)
```

3. We package the previous information inside a MP4 container, were we combine the 20-second video and all the generated audio tracks into a final .mp4 container (bbb_final_container.mp4), maintaining the original video quality.
```python
subprocess.run(
                [
                    "ffmpeg", "-i", video_20s, "-i", audio_aac, "-i", audio_mp3, "-i", audio_ac3,
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
                    "-c:v", "copy", "-c:a", "copy", final_output
                ],
                check=True
            )
```

Finally, we do a return statement that creates a dictionary with key-value pairs, where the keys are descriptive labels and the values are the paths to the generated output files:

```python
return {
    "video_20s": video_20s,        # Path to the 20-second video clip
    "audio_aac": audio_aac,        # Path to the AAC mono audio file
    "audio_mp3": audio_mp3,        # Path to the MP3 stereo audio file
    "audio_ac3": audio_ac3,        # Path to the AC3 audio file
    "final_container": final_output # Path to the final .mp4 container with video and all audio tracks
}
```
Again we include here the implementation of the integration test and the unit test for this fourth exercise:

```python
#Integration test bbb_editor
@app.get("/test_bbb_editor/")
async def test_bbb_editor():
    input_path = "../Seminar_2/input_file/bbb.mov"  
    output_dir = "../Seminar_2/output_file/"       

    try:
        # Llamar a la función `bbb_editor` con los argumentos requeridos
        result = ffmpeg_utils.bbb_editor(input_path, output_dir)
        return {"message": "BBB container created successfully", "output_files": result}
    except Exception as e:
        return {"error": str(e)}


#Unit test bbb_editor
@patch('subprocess.run')
def test_bbb_editor(self, mock_run):
        # Mock del entorno y de los paths
        mock_run.return_value = None  # Simular que todos los comandos se ejecutan correctamente
        output_dir = "output_directory"
        input_file = "input.mp4"
        
        # Crear instancia y llamar al método
        utils = ffmpeg_utils_comas_alvaro()
        result = utils.bbb_editor(input_file, output_dir)
        
        # Verificar múltiples llamadas de subprocess.run
        expected_calls = [
            # Primer comando (20 segundos de video)
            [
                "ffmpeg", "-i", input_file, "-ss", "00:00:00", "-t", "20", "-c:v", "copy", "-c:a", "copy", 
                f"{output_dir}/bbb_20s.mp4"
            ],
            # Segundo comando (AAC audio)
            [
                "ffmpeg", "-i", f"{output_dir}/bbb_20s.mp4", "-ac", "1", "-c:a", "aac", 
                f"{output_dir}/bbb_20s_aac.m4a"
            ],
            # Tercer comando (MP3 audio)
            [
                "ffmpeg", "-i", f"{output_dir}/bbb_20s.mp4", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "128k", 
                f"{output_dir}/bbb_20s_mp3.mp3"
            ],
            # Cuarto comando (AC3 audio)
            [
                "ffmpeg", "-i", f"{output_dir}/bbb_20s.mp4", "-c:a", "ac3", 
                f"{output_dir}/bbb_20s_ac3.ac3"
            ],
            # Quinto comando (empaquetar todos)
            [
                "ffmpeg", 
                "-i", f"{output_dir}/bbb_20s.mp4", 
                "-i", f"{output_dir}/bbb_20s_aac.m4a", 
                "-i", f"{output_dir}/bbb_20s_mp3.mp3", 
                "-i", f"{output_dir}/bbb_20s_ac3.ac3",
                "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
                "-c:v", "copy", "-c:a", "copy", 
                f"{output_dir}/bbb_final_container.mp4"
            ]
        ]

        for call_args in expected_calls:
            mock_run.assert_any_call(call_args, check=True)
        
        # Comprobar los resultados retornados
        self.assertEqual(result, {
            "video_20s": f"{output_dir}/bbb_20s.mp4",
            "audio_aac": f"{output_dir}/bbb_20s_aac.m4a",
            "audio_mp3": f"{output_dir}/bbb_20s_mp3.mp3",
            "audio_ac3": f"{output_dir}/bbb_20s_ac3.ac3",
            "final_container": f"{output_dir}/bbb_final_container.mp4"
        })
```


### S2 - Exercise 5
In this fifth exercise, we were asked to create a new endpoint / feature which reads the tracks from an MP4 container, and it’s able to say how many tracks does the container contains. In this exercise given an inputs (input file path), we print the number of tracks contained. The implementation of this exrecise is as follows:

```python
def mp4_reader(self,input_file):
        result = subprocess.run(
            ["ffprobe", "-i", input_file, "-show_streams", "-v", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        track_count = result.stdout.count("index=")
        return track_count
```
Again we include here the implementation of the integration test and the unit test for this fifth exercise:

```python
#Integration test mp4 reader (notice that input must be mp4 big buck bunny)
@app.get("/test_mp4_reader/")
async def test_mp4_reader():
    input_path = f"../Seminar_2/input_file/bbb.mp4"

    try:
        track_count = ffmpeg_utils.mp4_reader(input_path)
        return {"message": "MP4 reader test successful", "track_count": track_count}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Error during MP4 reading: {e}")


#Unit test mp4 reader
@patch('subprocess.run')
def test_mp4_reader_success(self, mock_run):
        # Crear un objeto simulado (mock) para la salida de subprocess.run
        mock_process = MagicMock()

        # Simulamos una salida de stdout que contiene información sobre dos flujos (video y audio)
        mock_process.stdout = "index=0\nindex=1\n"
        mock_process.stderr = ""

        # Configuramos el mock para que subprocess.run devuelva esta salida simulada
        mock_run.return_value = mock_process

        # Llamar a la función con un archivo de prueba
        result = ffmpeg_utils_comas_alvaro.mp4_reader(self, 'input.mp4')

        # Verificar que subprocess.run fue llamado con los parámetros correctos
        mock_run.assert_called_once_with(
            ["ffprobe", "-i", 'input.mp4', "-show_streams", "-v", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Verificar que el número de streams (tracks) es 2
        self.assertEqual(result, 2)
```

### S2 - Exercise 6
For this exercise, we were asked to extract macroblocks and motion vectors from a video using FFMPEG. The function takes an input file (the video to process) and an output file path (where the processed video will be saved). As in the previous exercises, We start by defining the input video file (input_file) and the output file (output_file) where the processed video will be saved. These paths are provided as arguments when the function is called.

Then, by testing with different bbb videos, we noticed that H.264 codec requires the video dimensions (width and height) to be divisible by 2. If the video dimensions are odd, FFMPEG could throw an error or not process the video correctly. To address this, the function scales the video using the following FFMPEG filter:

```python
"scale=trunc(iw/2)*2:trunc(ih/2)*2"
```

With this, we can run the FFMPEG comand with `subprocess.run`. This command processes the video and saves the output with the motion vectors and macroblocks shown:

```python
subprocess.run(
    [
        "ffmpeg", "-flags2", "+export_mvs", "-i", input_file, 
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,codecview=mv=pf+bf+bb",  
        output_file
    ],
    check=True
)
```
Again we include here the implementation of the integration test and the unit test for this sixth exercise:

```python
#Integration test video macroblocks
@app.get("/test_video_macroblocks/")
async def test_video_macroblocks():
    input_path = f"../Seminar_2/input_file/bbb.mov"
    output_file = "bbb_with_macroblocks.mp4"
    output_path = f"../Seminar_2/output_file/{output_file}"

    try:
        ffmpeg_utils.video_macroblocks(input_path, output_path)
        return {"message": "Video macroblocks extraction successful", "output_file": output_path}
    except Exception as e:
        return {"error": str(e)}


#Unit test video macroblocks
@patch('subprocess.run')
def test_video_macroblocks(self, mock_run):
        # Llamada al método
        ffmpeg_utils_comas_alvaro.video_macroblocks(self,"input.mp4", "output.mp4")
        
        # Verificar que subprocess.run se haya llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            [
                "ffmpeg", "-flags2", "+export_mvs", "-i", "input.mp4", 
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,codecview=mv=pf+bf+bb", 
                "output.mp4"
            ],
            check=True
        )
```

### S2 - Exercise 7
For the last exercise of this seminar we were asked to create a new endpoint / feature which output a video showing the YUV histogram. As in the previous exercises, given some inputs (input and ouput file path), we return the new file with the yuv histogra. The implementation of the function for this exercise is as follows:

```python
def yuv_histogram(self,input_file, output_file):
        subprocess.run(
            [
                "ffmpeg", 
                "-i", input_file, 
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay", 
                output_file
            ],
        check=True
        )
```
Again we include here the implementation of the integration test and the unit test for this last exercise:

```python
#Integration test yuv histogram
@app.get("/test_yuv_histogram/")
async def test_yuv_histogram():
    input_path = f"../Seminar_2/input_file/bbb.mov"
    output_file = "bbb_with_yuv.mp4"
    output_path = f"../Seminar_2/output_file/{output_file}"

    try:
        ffmpeg_utils.yuv_histogram(input_path,output_path)
        return {"message": "YUV histogram applied and video played successfully", "input_file": input_path}
    except Exception as e:
        return {"error": str(e)}


#Unit test yuv histogram
@patch('subprocess.run')
def test_yuv_histogram(self, mock_run):
        # Llamada al método
        ffmpeg_utils_comas_alvaro.yuv_histogram(self,"input.mp4","output.mp4")
        
        # Verificar que subprocess.run se haya llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            [
                "ffmpeg", 
                "-i", "input.mp4", 
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay", 
                "output.mp4"
            ],
            check=True
        )
```

## Practice_2
For this second practice, we will expand the implemened API by including additional FFMPEG functionalities. In this case, we ara tasked with adding two new endpoints/features in our API. The first feature consists on convert any input video into four different codec formats: VP8, VP9, h265 & AV1. And the second feature involves the implemenation of an encoding ladder. All the commands used in this practice to implment these two new endpoints are based on the FFMPEG documentation. 

Additionally, we are required to create a graphical user interface (GUI) as the final work of our API. The GUI will provide functionality for selecting a video file, select an output directory, and choosing one of the four codecs (VP8, VP9, H.265, or AV1) to apply the conversion. Further details about the GUI design and implementation will be explained in the subsequent sections.

For this delivery. we have created a new directory in our project named `Practice_2`. Inside this directory, we have included all the necessary Python files to run our API, operate our GUI, and implement the required endpoints:

- `practice_2.py`: This file contains the implementation of the new functions for the two endpoints specified in this practice. Some of the methods use functions implemented in previous deliveries like Seminar 2, so we have imported the necessary classes from the `second_seminar.py` file at the top of this file.

- `integration_tests.py`: This file also imports the classes from `second_seminar.py` and includes the integration tests. It uses the Big Buck Bunny video from previous exercises and adds a new integration test for the encoding ladder to validate its functionality.

- `main.py`: This file specifies the routes for the new endpoints, instantiates the tests from `integration_tests.py`, and call the function implemented to convert the codec type using the parameters defined in our GUI.

- `unit_tests.py`: This file includes unit tests from previous exercises, along with five new unit tests for this practice. Four of the new tests validate each of the codec formats (VP8, VP9, H.265, and AV1), and the fifth test proves the functionality of the encoding ladder.

- `gui.py`: This file contains the design and implementation of our GUI, which allows users to interact with the API.

To run the integration tests, we have to do exactly the same as in the previous delivery. The project folder must be organized as follows: Create two subfolders inside `Practice_2`: one named `input_file`, which should contain the Big Buck Bunny video in two formats: `bbb.mp4` and `bbb.mov`. The mp4 format is required for the mp4_reader function, while the mov format is used by the other functions. The second folder is named `output_file`, and it will store the output files generated by the FFMPEG calls.

This structure ensures that the input files are correctly organized and accessible for all tests and functions.

In order to run the unit tests you should run the next command from the terminal inside the `Practice_2` folder:
```
python -m unittest unit_tests.py
```

The instructions to  run the docker and API, are the same ones as in the previous delivery but we include it here again:
```
docker run -d -p 8000:8000 practice1-app
uvicorn main:app --reload
```

Finally in order to run our GUI, the steps that you need to follow are first of all run the API using the previous specified commands, and open a new terminal to run the GUI  using the next comand inside the `Practice_2` folder:

```
python gui.py
```

### Endpoint routes 
As we have explained in the `main.py`file we include a new routes to acces the new endpoints of our API. We put here to eases the process if you want to run the code while reading the instructions. We only add the endpoint to run the encoding ladder cause the route of the endpoint for changing the codec type is automatically defined in the GUI.

```
#NUEVO TEST CON ENDPOINTS:
#Test encode lader: /test/test_encode_ladder/"
```

### P2 - Exercise 1
For this first exercise of the last practice of the course, we have to create a new endpoint/feature to convert any input video into VP8, VP9, h265 & AV1.
In order to do this, we have created a new class inside the py file `practice_2.py`. Here, we have created the class `transcoding_utils_comas_alvaro`, where we are creting the new functions.
The first function is `convert_to_multiple_formats`. As an input, we ask for a input and outut path, and the format type, which should be one of the asked in the statement. 
First of all, we define all the possible codecs and how they will be exported:

```python
# Tipos de codec con los que trabvajaremos y como se exportaran
        output_files = {
            "H265": os.path.join(output_dir, "output_h265.mp4"),
            "VP9": os.path.join(output_dir, "output_vp9.webm"),
            "AV1": os.path.join(output_dir, "output_av1.mkv"),
            "VP8": os.path.join(output_dir, "output_vp8.webm"),
        }
```

Then, by using IF clauses, depending on the type asked by the user, we will execute different subprocess to have an optimized algorithm.
Most of the comands used are from FFMPEG website. With the VP9 one, we tried to do the 2 steps algorithm, but we obtained some errors and due to the lack of time, we decided to use the simpliest way to encode to VP9.

```python
#En función de la petición, decidimos que proceso ejecutamos con condiciones if
        if type == "H265":
            subprocess.run(
                [
                    "ffmpeg", "-i", input_file, 
                    "-c:v", "libx265", "-crf", "26", "-preset", "fast", 
                    "-c:a", "aac", "-b:a", "128k", output_files["H265"]
                ],
                check=True
            )
        #Intentamos hacer el de dos pasos pero peta, lo realizamos solo con uno
        elif type == "VP9":
            subprocess.run(
                [
                    "ffmpeg", "-i", input_file, 
                    "-c:v", "libvpx-vp9", "-b:v", "2M", output_files["VP9"]
                ],
                check=True
            )
        elif type == "AV1":
            subprocess.run(
                ["ffmpeg", "-i", input_file, "-c:v", "libaom-av1", "-crf", "30", output_files["AV1"]],
                check=True
            )
        elif type == "VP8":
            subprocess.run(
                ["ffmpeg", "-i", input_file, "-c:v", "libvpx", "-b:v", "1M", "-c:a", "libvorbis", output_files["VP8"]],
                check=True
            )
```

### P2 - Exercise 2
For this second exercise of the final practice in the course, we were tasked with creating a new endpoint/feature to generate an encoding ladder for input videos. This involves producing multiple output videos at different resolutions and formats, optimizing the backend by reusing previously implemented methods and applying inheritance or modular function design where appropriate.

To implement this functionality, we extended the `transcoding_utils_comas_alvaro` class in the `practice_2.py` file, by adding the `encode_ladder` function, which is responsible for generating an encoding ladder. This method processes the input video to produce multiple outputs at different resolutions: 1080p, 720p, 480p, and 360p. Each version is saved with a suffix indicating the resolution. Internally, this method utilizes the `resolution_adaptor` function from the `ffmpeg_utils_comas_alvaro` module to adapt the input video to each resolution.

```python
def encode_ladder(self, input_file, output_dir):

        #Todas las resoluciones con las que trabajaremos y su sufijo
        resolutions = [
            ("1920x1080", "1080p"),
            ("1280x720", "720p"),
            ("854x480", "480p"),
            ("640x360", "360p")
        ]


        # Generar cada versión del video
        for resolution, suffix in resolutions:
            output_file = os.path.join(output_dir, f"{suffix}.mp4")
            width, height = map(int, resolution.split('x'))
            ffmpeg_utils_comas_alvaro.resolution_adaptor(self, input_file, width, height, output_file)
```

As we have explained in the introduction we have implemented an integration test for this new functionality inside the `integration_tests.py`. The implementation of this test is as follows:

```python
#Integration test encode lader.
@app.get("/test_encode_ladder/")
async def test_encode_ladder():
    input_path = "../Practice_2/input_file/bbb.mov"  
    output_dir = "../Practice_2/output_file/"        

    try:
        # Crear instancia de transcoding utils
        transcoding_utils = transcoding_utils_comas_alvaro()
        
        ## Llamar a la función `encode_ladder` con los argumentos requeridos
        transcoding_utils.encode_ladder(input_path, output_dir)
        
        # Listar los archivos generados
        generated_files = [
            os.path.join(output_dir, f"{suffix}.mp4")
            for suffix in ["1080p", "720p", "480p", "360p"]
        ]

        #Verificar la creacion de los archivos
        missing_files = [file for file in generated_files if not os.path.exists(file)]
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Missing expected output files: {missing_files}"
            )

        return {
            "message": "Encode ladder test successful",
            "output_files": generated_files
        }
    except Exception as e:
        return {"error": str(e)}
```

### P2 - Exercise 3
For this third task, and one of the most complicated of the course, we have to design a GUI to use the created APIs in a intuitive way. We have previously worked with PyQt5 but not with APIs, so it's a challange for us. The initial idea was to implement most of the generated functions, but as we didn't have enough time, we have implemented the convert_to_multiple_formats, as we can use the input, output file and type. The GUI has been implemented in the file `gui.py`.

We first of all inicialize a class, and the gui settings:

```python
class monsterConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monster Converter") #Hemos llamado así a nuestra super GUI
        self.setGeometry(100, 100, 600, 400)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
```  
Then, we set some variables and titles of the gui:

```python
# Variables para almacenar rutas y formato + layout
        self.input_file_path = ""
        self.output_directory = ""
        self.selected_format = ""  
        layout = QVBoxLayout(self.central_widget)

        # Título principal
        title_label = QLabel("Monster Converter")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label, alignment=Qt.AlignTop)

```
Also, we wanted to have the codec type as a combo format, the input and output paths as a button to can surf on the local searcher and a execute button who runs and calls the API, so we inicialize the combo and the buttons with their respective subtittles

```python
# ComboBox con los formatos
        subtitle_label = QLabel("Select Codec to Convert:")
        subtitle_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(subtitle_label)
        self.activity_combo = QComboBox()
        self.activity_combo.addItems(["H265", "VP9", "AV1", "VP8"])
        self.activity_combo.currentTextChanged.connect(self.on_activity_selected)
        layout.addWidget(self.activity_combo)

        # Botón para cargar archivo de entrada
        load_button = QPushButton("Cargar Archivo (Input)")
        load_button.clicked.connect(lambda: self.browse_path("input_file_path"))
        layout.addWidget(load_button)

        self.file_input = QLineEdit()
        layout.addWidget(self.file_input)

        # Botón para seleccionar carpeta de salida
        output_button = QPushButton("Seleccionar Carpeta de Destino")
        output_button.clicked.connect(lambda: self.browse_path("output_directory"))
        layout.addWidget(output_button)

        self.output_input = QLineEdit()
        layout.addWidget(self.output_input)

        # Botón para ejecutar la acción
        execute_button = QPushButton("Ejecutar")
        execute_button.clicked.connect(self.execution)  # Conecta el botón a la función
        layout.addWidget(execute_button)
```

Finally, in order to do more interactive the buttons, we have created some extra functions, which allows to store on the variables the paths, calls some external functions for searching or to execute:

```python
   def browse_path(self, variable_name): #Función para poder acceder al explorador de manera amigable a seleccionar archivos y rutas

        if variable_name == "input_file_path":
            file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo")
            if file_path:
                self.file_input.setText(file_path)
                self.input_file_path = file_path  # Asigna a la variable global
        elif variable_name == "output_directory":
            output_directory = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Destino")
            if output_directory:
                self.output_input.setText(output_directory)
                self.output_directory = output_directory  # Asigna a la variable global

    def execution(self):
 
        input_file = self.input_file_path
        output_dir = self.output_directory
        format_type = self.selected_format

        if not input_file or not output_dir or not format_type: #Nos aseguramos de que todos los campos no esten vacios para poder ejecutar bien la llamada a la API
            print("Todos los campos son obligatorios")
            return

        # Llamada a la API para la conversión
        url = "http://127.0.0.1:8000/convert_video/" # URL del endpoint de la API para realizar la conversión.
        payload = {"input_file": input_file, "format_type": format_type, "output_dir": output_dir} # Creamos un diccionario con los datos que se enviarán en la solicitud
        response = requests.post(url, json=payload) # Envía una solicitud POST a la API con los datos en formato JSON
        print(response.json())
```

Finally, this block initializes the PyQt5 application. It creates an instance of QApplication, displays the main window of the monsterConverter class, and starts the event loop to handle user interactions. The sys.exit(app.exec()) ensures a clean exit when the application is closed.

```python
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = monsterConverter()
    window.show()
    sys.exit(app.exec())
```

To run the GUI, we have first to run the API, and then in the terminal the next command

```
python gui.py
```

### P2 - Exercise 4
In this last exercise we were asked to use AI to try to improve and reduce lines of our code. In our case, in order to improve our code using the AI we have implemented the fifth new unit tests  as we have comented in the introduction. The tests are included in the `unit_tests.py` file. And to run them, as we have already explained, we have tu use the next command from the terminal inside the `Practice_2` folder:

```
python -m unittest unit_tests.py
```

The new unit tests we have included allow us to verify that the conversion to the four codec types (VP8, VP9, H.265, and AV1) functions correctly and that the encoding ladder functionality operates as expected. The implementation of these tests is as follows:

```python
#FIRST TEST
@patch('subprocess.run')
def test_convert_to_multiple_formats_h265(self, mock_run):
        # Set up
        input_file = "input.mp4"
        output_dir = "output_directory"
        type = "H265"

        # Call the function
        transcoding_utils_comas_alvaro.convert_to_multiple_formats(self, input_file, output_dir, type)

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once_with(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libx265", "-crf", "26", "-preset", "fast", 
                "-c:a", "aac", "-b:a", "128k", os.path.join(output_dir, "output_h265.mp4")
            ],
            check=True
        )

#SECOND TEST        
@patch('subprocess.run')
def test_convert_to_multiple_formats_vp9(self, mock_run):
        # Set up
        input_file = "input.mp4"
        output_dir = "output_directory"
        type = "VP9"

        # Call the function
        transcoding_utils_comas_alvaro.convert_to_multiple_formats(self, input_file, output_dir, type)

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once_with(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libvpx-vp9", "-b:v", "2M", os.path.join(output_dir, "output_vp9.webm")
            ],
            check=True
        )

#THIRD TEST
@patch('subprocess.run')
def test_convert_to_multiple_formats_av1(self, mock_run):
        # Set up
        input_file = "input.mp4"
        output_dir = "output_directory"
        type = "AV1"

        # Call the function
        transcoding_utils_comas_alvaro.convert_to_multiple_formats(self, input_file, output_dir, type)

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once_with(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libaom-av1", "-crf", "30", os.path.join(output_dir, "output_av1.mkv")
            ],
            check=True
        )

#FOURTH TEST
@patch('subprocess.run')
def test_convert_to_multiple_formats_vp8(self, mock_run):
        # Set up
        input_file = "input.mp4"
        output_dir = "output_directory"
        type = "VP8"

        # Call the function
        transcoding_utils_comas_alvaro.convert_to_multiple_formats(self, input_file, output_dir, type)

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once_with(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libvpx", "-b:v", "1M", "-c:a", "libvorbis", os.path.join(output_dir, "output_vp8.webm")
            ],
            check=True
        )

#FIFTH TEST
@patch('second_seminar.ffmpeg_utils_comas_alvaro.resolution_adaptor')
def test_encode_ladder(self, mock_resolution_adaptor):
        # Set up
        input_file = "input.mp4"
        output_dir = "output_directory"

        # Call the function
        transcoding_utils_comas_alvaro.encode_ladder(self, input_file, output_dir)

        # Define the expected calls to resolution_adaptor
        expected_calls = [
            (1920, 1080, "1080p"),
            (1280, 720, "720p"),
            (854, 480, "480p"),
            (640, 360, "360p"),
        ]

        # Verify resolution_adaptor was called for each resolution
        for width, height, suffix in expected_calls:
            output_file = os.path.join(output_dir, f"{suffix}.mp4")
            mock_resolution_adaptor.assert_any_call(self, input_file, width, height, output_file)

        # Verify the number of calls matches
        self.assertEqual(mock_resolution_adaptor.call_count, len(expected_calls))

```


