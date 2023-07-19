# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose the port that the FastAPI app listens on
EXPOSE 8000

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
