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

class ffmpeg_utils_comas_alvaro: 
    def resolution_adaptor(self, input_file, width, height, output_file):
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-vf", f"scale={width}:{height}", output_file],
            check=True
        )
        return output_file

    def chroma_subsampling(self, input_file, output_file, pix_fmt):
        subprocess.run(
            #Pix format debe ser tipo yuv420, yuv422...
            ["ffmpeg", "-i", input_file, "-c:v", "libx264", "-pix_fmt", pix_fmt, output_file],
            check=True

        )
        return output_file

    def get_metadata(self, input_file, metadata_file):
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-f", "ffmetadata", metadata_file],
            check=True
        )

    def bbb_editor(self,input_file, output_dir):
        import os
        import subprocess

        os.makedirs(output_dir, exist_ok=True)

        # Definimos los paths que usaremos para exportar los diferentes formatos
        video_20s = os.path.join(output_dir, "bbb_20s.mp4")
        audio_aac = os.path.join(output_dir, "bbb_20s_aac.m4a")
        audio_mp3 = os.path.join(output_dir, "bbb_20s_mp3.mp3")
        audio_ac3 = os.path.join(output_dir, "bbb_20s_ac3.ac3")
        final_output = os.path.join(output_dir, "bbb_final_container.mp4")

        # Comandos FFMPEG, en el último empaquetamos todo
        subprocess.run(["ffmpeg", "-i", input_file, "-ss", "00:00:00", "-t", "20", "-c:v", "copy", "-c:a", "copy", video_20s], check=True)
        subprocess.run(["ffmpeg", "-i", video_20s, "-ac", "1", "-c:a", "aac", audio_aac], check=True)
        subprocess.run(["ffmpeg", "-i", video_20s, "-ac", "2", "-c:a", "libmp3lame", "-b:a", "128k", audio_mp3], check=True)
        subprocess.run(["ffmpeg", "-i", video_20s, "-c:a", "ac3", audio_ac3], check=True)
        subprocess.run(
            [
                "ffmpeg", "-i", video_20s, "-i", audio_aac, "-i", audio_mp3, "-i", audio_ac3,
                "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
                "-c:v", "copy", "-c:a", "copy", final_output
            ],
            check=True
        )

        return {
            "video_20s": video_20s,
            "audio_aac": audio_aac,
            "audio_mp3": audio_mp3,
            "audio_ac3": audio_ac3,
            "final_container": final_output
        }
        
    
    def mp4_reader(self,input_file):
        result = subprocess.run(
            ["ffprobe", "-i", input_file, "-show_streams", "-select_streams", "v,a", "-v", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        track_count = result.stdout.count("Stream #")
        return track_count

    def video_macroblocks(self,input_file,output_file):
        subprocess.run(
        [
            "ffmpeg", "-flags2", "+export_mvs", "-i", input_file, 
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,codecview=mv=pf+bf+bb",  # Show motion vectors and macroblocks. we have used CHATGPT to rescale in orther to work with all video sizes (problems with H264)
            output_file
        ],
        check=True
        )

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
