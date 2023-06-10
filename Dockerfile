FROM python:3.10

WORKDIR /app
COPY . /app/

RUN pip install numpy opencv-python tensorflow-cpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["python", "inference.py"]

