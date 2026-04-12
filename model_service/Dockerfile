FROM python:3.10-slim
WORKDIR /app
COPY requirements/training.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /app
ENV PYTHONPATH=/app
CMD ["uvicorn", "model_service.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
