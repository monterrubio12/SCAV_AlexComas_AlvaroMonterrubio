from fastapi import FastAPI
from test_main import app as test_app

app = FastAPI()
app.mount("/test", test_app)

@app.get("/")
async def root():
    return {"message": "Seminar 2 by Alex Comas & Alvaro Monterrubio"}


#HACER TEST CON ENDPOINTS:

#Test RGB to YUV: /test/test_rgb_to_yuv/
##Test YUV to RGB: /test/test_yuv_to_rgb/
#Test Serpentine: /test/test_serpentine/
#Test Resize: /test/test_resize/
#Test Black & White Conversion: /test/test_bw_converter/
#Test Run-Length Encoding: /test/test_run_length_encoding/
#Test DCT Encoding: /test/test_dct_encoding/
#Test DWT Encoding: /test/test_dwt_encoding/
