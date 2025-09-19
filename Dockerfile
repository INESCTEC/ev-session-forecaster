# Use an official Python runtime as a parent image
FROM python:3.12-bookworm

# File Author / Maintainer
LABEL maintainer="jose.j.dias@inesctec.pt"

# Set the working directory to /app
WORKDIR /app

# Install base software:
RUN apt-get update && apt-get -y install nano htop tree && apt-get clean all

# Copying the requirements for installation to take
# advantage of the caching.
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy necessary project files:
COPY . /app

EXPOSE 80

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
