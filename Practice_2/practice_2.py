class transcoding_utils_comas_alvaro: 
    def convert_to_multiple_formats(self, input_file):
 

        # Convertimos el video a H265
        subprocess.run(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libx265", "-crf", "26", "-preset", "fast", 
                "-c:a", "aac", "-b:a", "128k", 
                "output_h265.mp4"
            ],
            check=True
        )

        # Convertimos a VP9 en dos pasos, dentro de FFMPEG hemos visto que es lo recomendado para mantener algunos ajustes de calidad
        #Primer paso (Pass1) -> Usa el códec libvpx-vp9 para analizar el archivo de entrada

        subprocess.run(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libvpx-vp9", "-b:v", "2M", "-pass", "1", 
                "-an", "-f", "null", "/dev/null"
            ],
            check=True
        )

        # Segundo paso (Pass 2) -> Usa las estadísticas generadas en el primer paso para realizar la codificación optimizada
        subprocess.run(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libvpx-vp9", "-b:v", "2M", "-pass", "2", 
                "-c:a", "libopus", "output_vp9.webm"
            ],
            check=True
        )

        # AV1 Conversion
        subprocess.run(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libaom-av1", "-crf", "30", 
                "output_av1.mkv"
            ],
            check=True
        )

        # VP8 Conversion
        subprocess.run(
            [
                "ffmpeg", "-i", input_file, 
                "-c:v", "libvpx", "-b:v", "1M", 
                "-c:a", "libvorbis", 
                "output_vp8.webm"
            ],
            check=True
        )
    
    def encode_ladder(self, input_file, output_dir):


        resolutions = [
            ("1920x1080", "1080p"),
            ("1280x720", "720p"),
            ("854x480", "480p"),
            ("640x360", "360p")
        ]


        # Generar cada versión del video
        for resolution, suffix in resolutions:
            output_file = os.path.join(output_dir, f"{suffix}.mp4")
            self.resolution_adaptor(input_file, resolution, output_file)