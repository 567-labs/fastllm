FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
