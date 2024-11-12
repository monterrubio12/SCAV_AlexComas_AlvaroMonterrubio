# SCAV_AlexComas_AlvaroMonterrubio
This is the repository where we will be creating a software for SCAV practices at UPF.

## Seminar_1
During this first seminar on video coding in the SCAV course, we will implement a script named `first_seminar` that includes various classes and methods for processing images using the __ffmpeg__ software, along with other libraries for image and data processing such as scipy, numpy, and pillow. In this seminar, we will not only work with image manipulation we also deal with some of the core principles of image and video encoding and compression algorithms, essential tools for media transmission and storage.

### S1 - Exercise 1
In the first exercise of this first seminar, we were asked to compile and install the lastest version of __ffmpeg__ on our laptop/desktop (or any other device) using the command line. We needed to ensure that all necessary recent libraries were installed as part of the process. After the installation, in order to prove that the ffmpeg was succesfully installed, we were asked to run in our terminal the comand `ffmpeg` and upload a screenshot of the result. The screenshot that we obtained is the next one:

'ADD SCREENHOT'

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

