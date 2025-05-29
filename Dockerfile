FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install pytest pytest-cov matplotlib psutil tensorpac mngs

# Run tests
CMD ["pytest", "tests/", "-v"]