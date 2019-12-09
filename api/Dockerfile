FROM python:3.6.7
RUN mkdir -p /home/api
RUN mkdir -p /home/ml_models
WORKDIR /home/api
COPY . /home/api/
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install --no-cache-dir -r requirements.txt
CMD ["gunicorn","-t 100", "-w2", "-b 0.0.0.0:8000", "app:app"]
