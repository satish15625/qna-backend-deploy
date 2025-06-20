FROM python:3.10-slim

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000
CMD ["bash", "start.sh"]
