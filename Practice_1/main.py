from fastapi import FastAPI
from first_practice import Exercises

app = FastAPI()
exercise = Exercises()

@app.get("/")
async def root():
    return {"message": "Practice 1 by Alex Comas & Alvaro Monterrubio"}

@app.get("/rgb_to_yuv/")
async def rgb_to_yuv(r: int, g: int, b: int):


    Y, U, V = exercise.RGBtoYUV(r, g, b)
    return {
        "input_rgb": {"r": r, "g": g, "b": b},
        "output_yuv": {"Y": round(Y, 2), "U": round(U, 2), "V": round(V, 2)},
    }

@app.get("/yuv_to_rgb/")
async def yuv_to_rgb(Y: float, U: float, V: float):

    R, G, B = exercise.YUVtoRGB(Y, U, V)
    R, G, B = round(R), round(G), round(B)

    return {
        "input_yuv": {"Y": Y, "U": U, "V": V},
        "output_rgb": {"r": R, "g": G, "b": B},
    }


#HACER TEST CON ENDPOINTS:
#/rgb_to_yuv/?r=255&g=0&b=0
#/yuv_to_rgb/?Y=100&U=128&V=128
