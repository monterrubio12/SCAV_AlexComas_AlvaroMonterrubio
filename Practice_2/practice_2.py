import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Seminar_2')))
from second_seminar import  Exercises, dct_utils, dwt_utils, ffmpeg_utils_comas_alvaro
from PIL import Image
from typing import Iterator, Tuple, List
from itertools import groupby
import os
import subprocess
import numpy as np
from scipy.fft import dct, idct
import pywt

class transcoding_utils_comas_alvaro: 
    def convert_to_multiple_formats(self, input_file, type):
        output_dir = os.path.dirname(input_file)

        # Tipos de codec con los que trabvajaremos y como se exportaran
        output_files = {
            "H265": os.path.join(output_dir, "output_h265.mp4"),
            "VP9": os.path.join(output_dir, "output_vp9.webm"),
            "AV1": os.path.join(output_dir, "output_av1.mkv"),
            "VP8": os.path.join(output_dir, "output_vp8.webm"),
        }

        # En función de la petición, decidimos que proceso ejecutamos con condiciones if
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
            ffmpeg_utils_comas_alvaro.resolution_adaptor(input_file, resolution, output_file)