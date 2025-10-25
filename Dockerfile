FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
