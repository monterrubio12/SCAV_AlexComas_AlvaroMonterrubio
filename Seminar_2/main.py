from fastapi import FastAPI
from integration_tests import app as test_app

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


#NUEVOS TESTS CON ENDPOINTS:

#Test resolution adaptor: /test/test_resolution_adaptor/?width=1280&height=720"
#Test chroma subsampling: /test/test_chroma_subsampling/?pix_fmt=yuv420p"
#Test get metadata: /test/test_get_metadata/"
#Test bbb_editor: /test/test_bbb_editor/"
#Test mp4 reader: /test/test_mp4_reader/"
#Test macro blocks: /test/test_video_macroblocks/"
#Test yuv histogram: /test/test_yuv_histogram/"