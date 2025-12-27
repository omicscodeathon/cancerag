# Docker Deployment Guide for CancerAg Inference App

This guide explains how to deploy the CancerAg inference application using Docker.

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose 2.0+ (optional, for easier deployment)
- Trained models and preprocessing artifacts from the main pipeline

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Ensure models are trained**:
   ```bash
   # From project root, run the pipeline to generate models
   python src/cancerag/main.py
   ```

2. **Build and run with Docker Compose**:
   ```bash
   cd inference_app
   docker-compose up --build
   ```

3. **Access the app**:
   Open your browser to `http://localhost:7860`

### Option 2: Using Docker Directly

1. **Build the image**:
   ```bash
   # From project root
   docker build -f inference_app/Dockerfile -t cancerag-inference:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name cancerag-inference \
     -p 7860:7860 \
     -v $(pwd)/results/models:/app/results/models:ro \
     -v $(pwd)/data/processed/ml_preprocessed:/app/data/processed/ml_preprocessed:ro \
     -v $(pwd)/data/processed/receptors:/app/data/processed/receptors:ro \
     cancerag-inference:latest
   ```

3. **Access the app**:
   Open your browser to `http://localhost:7860`

## Required Files

Before deploying, ensure these files exist:

### Model Files
- `results/models/random_forest.pkl` (or other trained model)
- `results/models/` directory with at least one `.pkl` model file

### Preprocessing Artifacts
- `data/processed/ml_preprocessed/scaler.pkl`
- `data/processed/ml_preprocessed/preprocessing_metadata.json`
- `data/processed/ml_preprocessed/imputer.pkl` (optional)

### Receptor Structures (Optional, for docking)
- `data/processed/receptors/` directory with receptor PDBQT files
- `data/processed/receptors/binding_sites.json` (if using docking)

## Docker Image Details

### Base Image
- **Python 3.12-slim** - Lightweight Python runtime

### Included Tools
- **AutoDock Vina 1.2.5** - Pre-installed and available in PATH
- **OpenBabel** - Pre-installed for molecular format conversion
- **RDKit** - Python package for molecular processing

### Ports
- **7860** - Gradio web interface (exposed)

### Volumes
The following directories are mounted as read-only volumes:
- `results/models/` - Trained ML models
- `data/processed/ml_preprocessed/` - Preprocessing artifacts
- `data/processed/receptors/` - Receptor structures (optional)

## Environment Variables

You can customize the deployment with environment variables:

```bash
docker run -e GRADIO_SERVER_PORT=8080 \
           -e GRADIO_SERVER_NAME=0.0.0.0 \
           cancerag-inference:latest
```

Available variables:
- `GRADIO_SERVER_PORT` - Port for Gradio server (default: 7860)
- `GRADIO_SERVER_NAME` - Hostname to bind (default: 0.0.0.0)
- `PYTHONPATH` - Python path (default: /app)

## Production Deployment

### Using Docker Compose

For production, use a production-ready `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  inference-app:
    build:
      context: ..
      dockerfile: inference_app/Dockerfile
    ports:
      - "7860:7860"
    volumes:
      - ../results/models:/app/results/models:ro
      - ../data/processed/ml_preprocessed:/app/data/processed/ml_preprocessed:ro
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:7860')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Using a Reverse Proxy

For production, use nginx or traefik as a reverse proxy:

```nginx
# nginx.conf example
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Troubleshooting

### Container won't start

1. **Check logs**:
   ```bash
   docker logs cancerag-inference
   ```

2. **Verify model files exist**:
   ```bash
   ls -la results/models/*.pkl
   ls -la data/processed/ml_preprocessed/
   ```

3. **Check file permissions**:
   ```bash
   docker exec cancerag-inference ls -la /app/results/models/
   ```

### Model not found error

Ensure model files are mounted correctly:
```bash
docker run --rm -v $(pwd)/results/models:/app/results/models:ro \
  cancerag-inference:latest ls -la /app/results/models/
```

### Port already in use

Change the host port:
```bash
docker run -p 8080:7860 cancerag-inference:latest
```

Then access at `http://localhost:8080`

### RDKit import errors

RDKit is installed via `rdkit-pypi`. If issues persist, rebuild the image:
```bash
docker build --no-cache -f inference_app/Dockerfile -t cancerag-inference:latest .
```

## Building for Different Platforms

### Build for ARM64 (Apple Silicon, Raspberry Pi)

```bash
docker buildx build --platform linux/arm64 \
  -f inference_app/Dockerfile \
  -t cancerag-inference:arm64 .
```

### Build for AMD64

```bash
docker buildx build --platform linux/amd64 \
  -f inference_app/Dockerfile \
  -t cancerag-inference:amd64 .
```

## Updating the Application

1. **Rebuild the image**:
   ```bash
   docker-compose build --no-cache
   ```

2. **Restart the container**:
   ```bash
   docker-compose up -d
   ```

## Monitoring

### View logs
```bash
docker logs -f cancerag-inference
```

### Check health status
```bash
docker inspect --format='{{.State.Health.Status}}' cancerag-inference
```

### Resource usage
```bash
docker stats cancerag-inference
```

## Security Considerations

1. **Read-only volumes**: Model files are mounted as read-only
2. **Non-root user**: Consider running as non-root user in production
3. **Network isolation**: Use Docker networks to isolate the container
4. **Secrets management**: Use Docker secrets for sensitive data

## Performance Optimization

1. **Resource limits**: Set CPU and memory limits
2. **Caching**: Use Docker layer caching for faster rebuilds
3. **Multi-stage builds**: Already implemented for smaller image size
4. **Health checks**: Monitor container health automatically

## Support

For issues or questions:
1. Check the main README.md
2. Review inference_app/README.md
3. Check Docker logs: `docker logs cancerag-inference`
4. Verify all required files are present

---

**Status**: ✅ Docker deployment is production-ready and works well with the pipeline.

