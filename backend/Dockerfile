FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
