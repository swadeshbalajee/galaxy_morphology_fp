FROM python:3.10-slim
WORKDIR /app
COPY requirements/api.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir "psycopg[binary]==3.2.13"
COPY . /app
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
