# Deployment Guide

This guide covers deploying IMPULATOR to various platforms.

## Hugging Face Spaces (Recommended for Free Hosting)

### Prerequisites
- Hugging Face account (free)
- Git installed locally

### Deploy Steps

1. **Create a new Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select **Docker** as the SDK
   - Choose a name for your space

2. **Clone and push**
   ```bash
   # Clone your new space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME

   # Copy all files from this project
   cp -r /path/to/impulator/* .

   # Push to HF Spaces
   git add .
   git commit -m "Initial deployment"
   git push
   ```

3. **Configure Secrets (Optional but recommended)**
   - Go to Space Settings → Repository secrets
   - Add `AZURE_CONNECTION_STRING` for persistent storage
   - Add `AZURE_CONTAINER` (default: `impulator`)

4. **Wait for build**
   - Build takes ~5-10 minutes
   - Your app will be available at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### HF Spaces Limitations (Free Tier)
- 2 vCPU, 16GB RAM
- Container sleeps after 48h inactivity
- No persistent storage without Azure/external DB

---

## Docker (Local or VPS)

### Quick Start
```bash
# Build and run
docker-compose up -d

# Or without compose
docker build -t impulator .
docker run -p 7860:7860 -v ./data:/app/data impulator
```

### Access
- Frontend: http://localhost:7860
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run both backend and frontend
./start.sh

# Or run separately
python -m uvicorn backend.main:app --port 8000 &
streamlit run frontend/app.py --server.port 7860
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | Backend host |
| `API_PORT` | No | `8000` | Backend port |
| `FRONTEND_PORT` | No | `7860` | Frontend port |
| `AZURE_CONNECTION_STRING` | No | - | Azure Blob connection string |
| `AZURE_CONTAINER` | No | `impulator` | Azure container name |
| `MAX_WORKERS` | No | `2` | Concurrent job limit |
| `DEBUG` | No | `false` | Enable debug mode |

---

## Architecture

```
┌─────────────────────────────────────┐
│         Single Container            │
│                                     │
│  ┌─────────────┐  ┌──────────────┐  │
│  │  Streamlit  │  │   FastAPI    │  │
│  │   (7860)    │──│   (8000)     │  │
│  └─────────────┘  └──────────────┘  │
│                         │           │
│                    ┌────▼────┐      │
│                    │ SQLite  │      │
│                    └─────────┘      │
└─────────────────────────────────────┘
              │
              ▼ (optional)
        ┌───────────┐
        │Azure Blob │
        └───────────┘
```

---

## Troubleshooting

### Container won't start
- Check logs: `docker logs impulator-app`
- Ensure ports 7860 and 8000 are free

### Backend not responding
- Backend needs ~30-60s to initialize
- Check health: `curl http://localhost:8000/api/v1/health`

### Data not persisting (HF Spaces)
- Add Azure credentials in Space secrets
- Without Azure, data resets on container restart
