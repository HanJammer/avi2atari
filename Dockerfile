# AVI2ATARI AVF Suite - web service
FROM python:3.12-slim

# ffmpeg is the converter's decode/scale backend
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# scratch on tmpfs (compose mounts /dev/shm); nothing is persisted
ENV SCRATCH_DIR=/dev/shm \
    HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
