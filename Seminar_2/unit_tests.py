import unittest
import subprocess
from unittest.mock import patch, MagicMock
from second_seminar import ffmpeg_utils_comas_alvaro  # Reemplaza con el nombre real de tu módulo

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
    def test_video_macroblocks(self, mock_run):
        # Llamada al método
        ffmpeg_utils_comas_alvaro.video_macroblocks(self,"input.mp4", "output.mp4")
        
        # Verificar que subprocess.run se haya llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            [
                "ffmpeg", "-flags2", "+export_mvs", "-i", "input.mp4", 
                "-vf", "codecview=mv=pf+bf+bb", "output.mp4"
            ],
            check=True
        )

    @patch('subprocess.run')
    def test_yuv_histogram(self, mock_run):
        # Llamada al método
        ffmpeg_utils_comas_alvaro.yuv_histogram(self,"input.mp4")
        
        # Verificar que subprocess.run se haya llamado con los argumentos correctos
        mock_run.assert_called_once_with(
            [
                "ffplay", "input.mp4", 
                "-vf", "split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay"
            ],
            check=True
        )

    @patch('subprocess.run')
    def test_mp4_reader(self, mock_run):
        # Crear un objeto simulado (mock) para la salida de subprocess.run
        mock_process = MagicMock()
        
        # Simulamos una salida de stdout que contiene información sobre dos flujos (video y audio)
        mock_process.stdout = "Stream #0:0(und): Video: h264, yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 1500 kb/s, 25 fps, 25 tbr, 25 tbn, 50 tbc\nStream #0:1(und): Audio: aac, 44100 Hz, stereo, fltp, 128 kb/s\n"
        mock_process.stderr = ""
        
        # Configuramos el mock para que subprocess.run devuelva esta salida simulada
        mock_run.return_value = mock_process

        # Llamar a la función con un archivo de prueba
        result = ffmpeg_utils_comas_alvaro.mp4_reader(self,'input.mp4')

        # Verificar que subprocess.run fue llamado con los parámetros correctos
        mock_run.assert_called_once_with(
            ["ffprobe", "-i", 'input.mp4', "-show_streams", "-select_streams", "v,a", "-v", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Verificar que el número de streams (tracks) es 2
        self.assertEqual(result, 2)

    @patch('subprocess.run')
    def test_mp4_reader_no_tracks(self, mock_run):
        # Simulamos que la salida de stdout no contiene flujos
        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_process.stderr = ""
        
        # Configuramos el mock para que subprocess.run devuelva esta salida simulada
        mock_run.return_value = mock_process

        # Llamar a la función con un archivo de prueba
        result = ffmpeg_utils_comas_alvaro.mp4_reader(self,'input.mp4')

        # Verificar que el número de streams (tracks) es 0
        self.assertEqual(result, 0)

    @patch('subprocess.run')
    def test_mp4_reader_error(self, mock_run):
        # Simulamos un error en subprocess.run
        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_process.stderr = "Error: File not found."
        
        # Configuramos el mock para que subprocess.run devuelva esta salida simulada
        mock_run.return_value = mock_process

        # Verificar que el número de streams (tracks) es 0, ya que no se encontraron flujos
        result = ffmpeg_utils_comas_alvaro.mp4_reader(self,'input.mp4')

        # Verificar que no haya tracks
        self.assertEqual(result, 0)


# Ejecuta los tests
if __name__ == '__main__':
    unittest.main()
