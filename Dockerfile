FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# ffmpeg re-encodes the result video so browsers can play it;
# libgl1 and libglib2.0-0 are needed by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# run the app as an ordinary user, not as administrator
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# CPU-only PyTorch keeps the image small and fast to build (the server has no GPU).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# copy app.py, best.pt and templates/ into the image
COPY --chown=user . .

# Railway tells the app which port to use through the PORT variable (8080 is the fallback when run locally)
EXPOSE 8080
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 600 app:app
