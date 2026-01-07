FROM python:3.11-slim

# 1. Install system dependencies first (Top of file = Better Caching)
# We combine update and install into one RUN to keep the image small
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]