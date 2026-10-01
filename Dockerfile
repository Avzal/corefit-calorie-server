FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# CPU-only torch wheel — much smaller than the default CUDA build, and this
# model is light enough (ViT-B/32) that CPU inference is fine for one request
# at a time.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
EXPOSE 8787
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8787"]
