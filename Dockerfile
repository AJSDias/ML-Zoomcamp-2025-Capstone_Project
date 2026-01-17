FROM python:3.12-slim

WORKDIR /app

# Copy dependency file
COPY pyproject.toml ./

# Install uv
RUN pip install uv

# Install dependencies from pyproject.toml
RUN uv sync

# Install uvicorn explicitly
RUN uv pip install uvicorn

# Copy all project files
COPY . .

# Copy the trained model file
COPY model.pth ./

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start FastAPI
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
