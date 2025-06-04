FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better caching
COPY requirements*.txt ./
COPY pyproject.toml ./
COPY src ./src

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install -r requirements-test.txt

# Copy test files
COPY tests ./tests

# Run tests
CMD ["pytest", "tests/", "-v"]