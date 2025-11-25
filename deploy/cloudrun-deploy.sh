#!/bin/bash
# =============================================================================
# ToM-NAS Cloud Run Deployment (Simpler, CPU-only but still real)
# Good for demos and small experiments without GPU costs
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "=============================================="
echo "  ToM-NAS Cloud Run Deployment"
echo "  Quick & easy serverless deployment"
echo "=============================================="
echo -e "${NC}"

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="tom-nas"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No GCP project configured.${NC}"
    echo "Run: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

echo -e "${GREEN}Project: $PROJECT_ID${NC}"
echo -e "${GREEN}Region: $REGION${NC}"
echo ""

# Enable APIs
echo -e "${BLUE}Enabling APIs...${NC}"
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID

# Build and deploy in one step
echo -e "${BLUE}Building and deploying to Cloud Run...${NC}"
echo "This will take 5-10 minutes..."
echo ""

gcloud run deploy $SERVICE_NAME \
    --project=$PROJECT_ID \
    --region=$REGION \
    --source=. \
    --platform=managed \
    --allow-unauthenticated \
    --memory=4Gi \
    --cpu=2 \
    --timeout=600 \
    --concurrency=10 \
    --min-instances=0 \
    --max-instances=3 \
    --set-env-vars="PYTHONPATH=/app"

# Get the URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --project=$PROJECT_ID \
    --region=$REGION \
    --format='value(status.url)')

echo ""
echo -e "${GREEN}=============================================="
echo "  Deployment Complete!"
echo "==============================================${NC}"
echo ""
echo -e "  ${BLUE}Application URL:${NC} $SERVICE_URL"
echo ""
echo -e "  ${YELLOW}Note: Cloud Run scales to zero when idle (free!)"
echo "  First request after idle may take 30-60 seconds.${NC}"
echo ""
echo "  To view logs:"
echo "    gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo ""
echo "  To delete:"
echo "    gcloud run services delete $SERVICE_NAME --region=$REGION"
echo ""
