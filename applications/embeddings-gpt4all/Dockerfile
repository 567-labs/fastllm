# Stage 1: Build Stage
FROM python:3.9-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Install the app dependencies in a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM python:3.9-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Add the virtual environment to the PATH
ENV PATH="/venv/bin:$PATH"

COPY . .

# Expose the port that the FastAPI app listens on
EXPOSE 8000

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
