import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QLineEdit, QComboBox
)
from PyQt5.QtCore import Qt
import requests

class monsterConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monster Converter") #Hemos llamado así a nuestra super GUI
        self.setGeometry(100, 100, 600, 400)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Variables para almacenar rutas y formato + layout
        self.input_file_path = ""
        self.output_directory = ""
        self.selected_format = ""  
        layout = QVBoxLayout(self.central_widget)

        # Título principal
        title_label = QLabel("Monster Converter")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label, alignment=Qt.AlignTop)

        # ComboBox con los formatos
        subtitle_label = QLabel("Select Codec to Convert:")
        subtitle_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(subtitle_label)
        self.activity_combo = QComboBox()
        self.activity_combo.addItems(["H265", "VP9", "AV1", "VP8"])
        self.activity_combo.currentTextChanged.connect(self.on_activity_selected)
        layout.addWidget(self.activity_combo)

        # Botón para cargar archivo de entrada
        load_button = QPushButton("Cargar Archivo (Input)")
        load_button.clicked.connect(lambda: self.browse_path("input_file_path"))
        layout.addWidget(load_button)

        self.file_input = QLineEdit()
        layout.addWidget(self.file_input)

        # Botón para seleccionar carpeta de salida
        output_button = QPushButton("Seleccionar Carpeta de Destino")
        output_button.clicked.connect(lambda: self.browse_path("output_directory"))
        layout.addWidget(output_button)

        self.output_input = QLineEdit()
        layout.addWidget(self.output_input)

        # Botón para ejecutar la acción
        execute_button = QPushButton("Ejecutar")
        execute_button.clicked.connect(self.execution)  # Conecta el botón a la función
        layout.addWidget(execute_button)

    def on_activity_selected(self, selected_activity):
        # Actualizamos la variable con el formato seleccionado
        self.selected_format = selected_activity

    def browse_path(self, variable_name): #Función para poder acceder al explorador de manera amigable a seleccionar archivos y rutas

        if variable_name == "input_file_path":
            file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo")
            if file_path:
                self.file_input.setText(file_path)
                self.input_file_path = file_path  # Asigna a la variable global
        elif variable_name == "output_directory":
            output_directory = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Destino")
            if output_directory:
                self.output_input.setText(output_directory)
                self.output_directory = output_directory  # Asigna a la variable global

    def execution(self):
 
        input_file = self.input_file_path
        output_dir = self.output_directory
        format_type = self.selected_format

        if not input_file or not output_dir or not format_type: #Nos aseguramos de que todos los campos no esten vacios para poder ejecutar bien la llamada a la API
            print("Todos los campos son obligatorios")
            return

        # Llamada a la API para la conversión
        url = "http://127.0.0.1:8000/convert_video/" # URL del endpoint de la API para realizar la conversión.
        payload = {"input_file": input_file, "format_type": format_type, "output_dir": output_dir} # Creamos un diccionario con los datos que se enviarán en la solicitud
        response = requests.post(url, json=payload) # Envía una solicitud POST a la API con los datos en formato JSON
        print(response.json())
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = monsterConverter()
    window.show()
    sys.exit(app.exec())


