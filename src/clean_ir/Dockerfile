FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY pipeline.py /app/pipeline.py
COPY evaluate.py /app/evaluate.py
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python3", "/app/pipeline.py"]
