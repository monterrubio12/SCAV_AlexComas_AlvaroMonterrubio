from fastapi import FastAPI
from second_seminar import Exercises, dct_utils, dwt_utils
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
