FROM python:3.11-slim

WORKDIR /app

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir .

# Run the server
CMD ["tos-bridge"]
