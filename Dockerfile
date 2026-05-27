FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    swig \
    libopenblas-dev \
    libmagic1 \
    file

COPY requirements-linux.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements-linux.txt

RUN python -m spacy download en_core_web_sm

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.11
# CMD ["bash"]