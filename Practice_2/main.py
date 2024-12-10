from fastapi import FastAPI
from pydantic import BaseModel  
from integration_tests import app as test_app
from practice_2 import transcoding_utils_comas_alvaro

app = FastAPI()
app.mount("/test", test_app)

# Definir la clase ConversionRequest
class ConversionRequest(BaseModel):
    input_file: str
    format_type: str
    output_dir: str


@app.get("/")
async def root():
    return {"message": "Practice2 by Alex Comas & Alvaro Monterrubio"}

@app.post("/convert_video/")
async def convert_video(request: ConversionRequest):
    input_file = request.input_file
    format_type = request.format_type
    output_dir = request.output_dir
    
    try:
        # Instanciar el transcoder
        transcoder = transcoding_utils_comas_alvaro()
        
        # Llamamos a la función de transcoding
        transcoder.convert_to_multiple_formats(input_file, output_dir, format_type)
        
        return {"message": f"Video convertido a {format_type} exitosamente", "input_file": input_file}
    
    except Exception as e:
        return {"error": str(e)}



#HACER TEST CON ENDPOINTS:

#Test RGB to YUV: /test/test_rgb_to_yuv/
##Test YUV to RGB: /test/test_yuv_to_rgb/
#Test Serpentine: /test/test_serpentine/
#Test Resize: /test/test_resize/
#Test Black & White Conversion: /test/test_bw_converter/
#Test Run-Length Encoding: /test/test_run_length_encoding/
#Test DCT Encoding: /test/test_dct_encoding/
#Test DWT Encoding: /test/test_dwt_encoding/


#TESTS CON ENDPOINTS S2:

#Test resolution adaptor: /test/test_resolution_adaptor/?width=1280&height=720"
#Test chroma subsampling: /test/test_chroma_subsampling/?pix_fmt=yuv420p"
#Test get metadata: /test/test_get_metadata/"
#Test bbb_editor: /test/test_bbb_editor/"
#Test mp4 reader: /test/test_mp4_reader/"
#Test macro blocks: /test/test_video_macroblocks/"
#Test yuv histogram: /test/test_yuv_histogram/"

#NUEVO TEST CON ENDPOINTS:
#Test encode lader: /test/test_encode_ladder/"
