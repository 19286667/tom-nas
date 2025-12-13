# ToM-NAS Google Cloud Deployment Guide

This guide walks you through deploying ToM-NAS to Google Cloud Platform for production use.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed locally
3. **Docker** installed locally (for testing)
4. **Git** for version control

## Quick Start (5 Minutes to Deploy)

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/19286667/tom-nas.git
cd tom-nas

# Copy environment template
cp .env.example .env

# Edit .env with your GCP project ID
nano .env
```

### 2. Set Up Google Cloud

```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    secretmanager.googleapis.com

# Create service account for Cloud Run
gcloud iam service-accounts create tom-nas-sa \
    --display-name="ToM-NAS Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:tom-nas-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:tom-nas-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"
```

### 3. Build and Deploy

```bash
# Build the Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tom-nas-api:latest

# Deploy to Cloud Run
gcloud run deploy tom-nas-api \
    --image gcr.io/YOUR_PROJECT_ID/tom-nas-api:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --set-env-vars "TOM_NAS_ENV=production,ENABLE_CLOUD_LOGGING=true"
```

### 4. Get Your URL

```bash
# Get the service URL
gcloud run services describe tom-nas-api \
    --region us-central1 \
    --format 'value(status.url)'
```

**That's it!** Your API is now live at the URL returned above.

---

## Detailed Setup

### Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TOM_NAS_ENV` | Environment (development/staging/production) | development |
| `TOM_NAS_LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `TOM_NAS_DEVICE` | Compute device (cpu/cuda) | cpu |
| `ENABLE_CLOUD_LOGGING` | Send logs to Cloud Logging | false |
| `ENABLE_METRICS` | Expose Prometheus metrics | true |
| `GCP_PROJECT_ID` | Your Google Cloud project ID | - |

### Secrets Management

For sensitive values, use Google Secret Manager:

```bash
# Create a secret
echo -n "your-api-key" | gcloud secrets create api-key --data-file=-

# Reference in Cloud Run
gcloud run services update tom-nas-api \
    --update-secrets=API_KEY=api-key:latest
```

### Custom Domain

```bash
# Map a custom domain
gcloud run domain-mappings create \
    --service tom-nas-api \
    --domain tom-nas.yourdomain.com \
    --region us-central1
```

---

## Deployment Options

### Option A: Cloud Run (Recommended)

**Best for**: API serving, auto-scaling, pay-per-use

```bash
gcloud run deploy tom-nas-api \
    --image gcr.io/YOUR_PROJECT_ID/tom-nas-api:latest \
    --region us-central1 \
    --platform managed \
    --memory 4Gi \
    --cpu 2
```

### Option B: App Engine

**Best for**: Simpler deployment, fixed pricing

Create `app.yaml`:
```yaml
runtime: custom
env: flex

resources:
  cpu: 2
  memory_gb: 4
  disk_size_gb: 10

env_variables:
  TOM_NAS_ENV: "production"
```

Deploy:
```bash
gcloud app deploy
```

### Option C: Google Kubernetes Engine (GKE)

**Best for**: Complex deployments, GPU workloads

```bash
# Create cluster
gcloud container clusters create tom-nas-cluster \
    --num-nodes=3 \
    --machine-type=n1-standard-4

# Deploy
kubectl apply -f k8s/deployment.yaml
```

---

## Streamlit Dashboard Deployment

Deploy the interactive dashboard separately:

```bash
# Build dashboard image
gcloud builds submit \
    --tag gcr.io/YOUR_PROJECT_ID/tom-nas-dashboard:latest \
    --file Dockerfile \
    --target streamlit

# Deploy to Cloud Run
gcloud run deploy tom-nas-dashboard \
    --image gcr.io/YOUR_PROJECT_ID/tom-nas-dashboard:latest \
    --region us-central1 \
    --port 8501 \
    --allow-unauthenticated
```

---

## CI/CD Setup

### GitHub Actions

1. Go to your GitHub repository settings
2. Add these secrets:
   - `GCP_PROJECT_ID`: Your project ID
   - `GCP_SA_KEY`: Service account JSON key

3. The workflow at `.github/workflows/ci-cd.yml` will:
   - Run tests on every push
   - Build Docker images
   - Deploy to staging (develop branch)
   - Deploy to production (main branch)

### Manual Deployment

```bash
# Build and push
docker build -t gcr.io/YOUR_PROJECT_ID/tom-nas-api:latest .
docker push gcr.io/YOUR_PROJECT_ID/tom-nas-api:latest

# Deploy
gcloud run deploy tom-nas-api \
    --image gcr.io/YOUR_PROJECT_ID/tom-nas-api:latest
```

---

## Monitoring & Observability

### View Logs

```bash
# Stream logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=tom-nas-api"

# View in Console
open "https://console.cloud.google.com/logs/query?project=YOUR_PROJECT_ID"
```

### Metrics

Prometheus metrics are exposed at `/metrics`. Set up Cloud Monitoring:

```bash
# View metrics
open "https://console.cloud.google.com/monitoring?project=YOUR_PROJECT_ID"
```

### Alerts

Create alerts for:
- High error rate (>5%)
- High latency (>2s p95)
- Instance count spikes

---

## Cost Estimation

| Component | Estimated Monthly Cost |
|-----------|----------------------|
| Cloud Run (API) | $20-50 |
| Cloud Run (Dashboard) | $10-20 |
| Cloud Logging | $5-10 |
| Cloud Storage (checkpoints) | $1-5 |
| **Total** | **$36-85/month** |

*Based on moderate usage. Scale to zero when not in use for minimum costs.*

---

## Troubleshooting

### Container won't start

```bash
# Check logs
gcloud run services logs read tom-nas-api --region us-central1

# Common fixes:
# - Increase memory (--memory 8Gi)
# - Check environment variables
# - Verify Docker image builds locally
```

### High latency

```bash
# Enable startup CPU boost
gcloud run services update tom-nas-api \
    --cpu-boost
```

### Out of memory

```bash
# Increase memory allocation
gcloud run services update tom-nas-api \
    --memory 8Gi
```

---

## API Reference

Once deployed, access:
- **API Documentation**: `https://YOUR_URL/docs`
- **Health Check**: `https://YOUR_URL/health`
- **Metrics**: `https://YOUR_URL/metrics`

### Example API Calls

```bash
# Health check
curl https://YOUR_URL/health

# Run inference
curl -X POST https://YOUR_URL/api/v1/inference \
    -H "Content-Type: application/json" \
    -d '{"observation": [0.5, 0.5, ...]}'

# List experiments
curl https://YOUR_URL/api/v1/experiments
```

---

## Support

- **Issues**: https://github.com/19286667/tom-nas/issues
- **Documentation**: https://github.com/19286667/tom-nas/wiki
- **Google Cloud Support**: https://cloud.google.com/support

---

## Security Checklist

- [ ] Enable IAM authentication for production
- [ ] Configure VPC connector for private resources
- [ ] Enable Cloud Armor for DDoS protection
- [ ] Set up SSL/TLS with managed certificates
- [ ] Review and restrict service account permissions
- [ ] Enable audit logging
- [ ] Configure secrets rotation
