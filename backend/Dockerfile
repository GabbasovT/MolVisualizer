FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y git && apt-get clean

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
