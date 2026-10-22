# Meat type detector

CES117V capstone project. A YOLOv8n model, fine-tuned on a labelled dataset of raw chicken, beef,
pork and lamb, detects those types of meat in uploaded videos and photos. The model is served by a
Flask app, packaged with Docker and deployed on Railway from this GitHub repository.

## Files

| File | What it does |
|---|---|
| `app.py` | Flask backend. Loads `best.pt` and handles `/`, `/health`, `/predict` and `/outputs/<name>`. |
| `best.pt` | The fine-tuned YOLOv8n weights. |
| `templates/index.html` | The web page for uploading a video or photo and seeing the result. |
| `requirements.txt` | Python packages (PyTorch CPU is installed in the Dockerfile). |
| `Dockerfile` | Build and run instructions used by Railway. |

## Run it locally

```
pip install flask ultralytics
python app.py
```

Then open http://127.0.0.1:8080 in a browser. (ffmpeg must be installed for the result
video to play in every browser.)

## Deploy

Push this folder to a GitHub repository, then in Railway choose **New Project, Deploy from
GitHub repo**, pick the repository and generate a public domain under
**Settings, Networking**. Railway finds the Dockerfile and sets the `PORT` variable itself.
