import sys
#pip install PyQt5, pip install requests

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QProgressBar, QLineEdit, QComboBox
)
from PyQt5.QtCore import Qt

import requests

class monsterConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monster Converter")
        self.setGeometry(100, 100, 600, 400)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.input_file_path = ""
        self.output_directory = ""
        self.selected_format = ""  

        # Layout principal
        layout = QVBoxLayout(self.central_widget)

        # Título principal en la parte superior
        title_label = QLabel("Monster Converter")
        title_label.setAlignment(Qt.AlignCenter)  # Centrar horizontalmente
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label, alignment=Qt.AlignTop)  # Alinear en la parte superior

        # ComboBox con los formatos, sin texto adicional
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

        # Botón para cargar directorio de salida
        output_button = QPushButton("Seleccionar Carpeta de Destino")
        output_button.clicked.connect(lambda: self.browse_path("output_directory"))
        layout.addWidget(output_button)

        self.output_input = QLineEdit()
        layout.addWidget(self.output_input)

        # Botón para ejecutar la acción
        execute_button = QPushButton("Ejecutar")
        layout.addWidget(execute_button)

    def on_activity_selected(self, selected_activity):
        # Actualizamos la variable con el formato seleccionado
        self.selected_format = selected_activity

    def browse_path(self, variable_name):
        # Dependiendo del nombre de la variable, seleccionamos archivo o directorio
        if variable_name == "input_file_path":
            # Seleccionamos un archivo de entrada
            file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo")
            if file_path:
                self.file_input.setText(file_path)
                setattr(self, variable_name, file_path)  # Asignamos a la variable indicada
        elif variable_name == "output_directory":
            # Seleccionamos una carpeta de salida
            output_directory = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Destino")
            if output_directory:
                self.output_input.setText(output_directory)
                setattr(self, variable_name, output_directory)  # Asignamos a la variable indicada
    
    def execution(self):
        input_file = self.input_file_path
        output_dir = self.output_directory
        format_type = self.selected_format
        # Realizamos la llamada a la API para la conversión
        url = "http://127.0.0.1:8000/convert_video/"
        payload = {"input_file": input_file, "format_type": format_type}
        response = requests.post(url, json=payload)







if __name__ == "__main__":
    app = QApplication(sys.argv)  
    window = monsterConverter()   
    window.show()                 
    sys.exit(app.exec_())         
