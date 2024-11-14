FROM python:3.13


WORKDIR /usr/src/app

# Instalar las dependencias del requirements 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el codigo al contenedor
COPY . .

# Usar puerto 8000 para la API
EXPOSE 8000

# Comando para iniciar la aplicación usando uvicorn
CMD ["uvicorn", "main:app"]
