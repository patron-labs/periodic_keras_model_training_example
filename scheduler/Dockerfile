FROM python:3.6.7
RUN mkdir -p /home/scheduler
RUN mkdir -p /home/ml_models
WORKDIR /home/scheduler
COPY . /home/scheduler/
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]
