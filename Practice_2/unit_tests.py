import unittest
import subprocess
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Seminar_2')))
from second_seminar import ffmpeg_utils_comas_alvaro  # Reemplaza con el nombre real de tu módulo
from practice_2 import transcoding_utils_comas_alvaro

class TestFFmpegUtils(unittest.TestCase):
    
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

    @patch('subprocess.run')
    def test_get_metadata(self, mock_run):
        # Llama al método
        ffmpeg_utils_comas_alvaro.get_metadata(self,"input.mp4", "metadata.txt")
        
        # Verifica que subprocess.run fue llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            ["ffmpeg", "-i", "input.mp4", "-f", "ffmetadata", "metadata.txt"],
            check=True
        )


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



# Ejecuta los tests
if __name__ == '__main__':
    unittest.main()