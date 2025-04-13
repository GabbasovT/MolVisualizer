FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-glx \
    libgtk2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
