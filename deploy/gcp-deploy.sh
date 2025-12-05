#!/bin/bash
# =============================================================================
# ToM-NAS Google Cloud Deployment Script
# Deploys the web application with GPU support for real neural evolution
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================="
echo "  ToM-NAS Google Cloud Deployment"
echo "  Evolving AI that understands minds"
echo "=============================================="
echo -e "${NC}"

# Configuration - EDIT THESE
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="tom-nas-gpu"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"    # Cost-effective GPU
GPU_COUNT=1

# Check for project ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}GCP_PROJECT_ID not set. Attempting to detect...${NC}"
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: No GCP project configured.${NC}"
        echo "Please run: export GCP_PROJECT_ID=your-project-id"
        echo "Or: gcloud config set project your-project-id"
        exit 1
    fi
fi

echo -e "${GREEN}Using project: $PROJECT_ID${NC}"
echo -e "${GREEN}Region: $REGION, Zone: $ZONE${NC}"
echo ""

# Function to check if gcloud is installed
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}Error: gcloud CLI not installed.${NC}"
        echo "Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

# Function to enable required APIs
enable_apis() {
    echo -e "${BLUE}Enabling required APIs...${NC}"
    gcloud services enable compute.googleapis.com --project=$PROJECT_ID
    gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID
    gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
    echo -e "${GREEN}APIs enabled.${NC}"
}

# Function to build and push Docker image
build_image() {
    echo -e "${BLUE}Building Docker image with GPU support...${NC}"

    IMAGE_URI="gcr.io/$PROJECT_ID/tom-nas-gpu:latest"

    # Build using Cloud Build (faster, no local Docker needed)
    gcloud builds submit \
        --project=$PROJECT_ID \
        --tag=$IMAGE_URI \
        --timeout=30m \
        --machine-type=e2-highcpu-8 \
        -f Dockerfile.gpu \
        .

    echo -e "${GREEN}Image built: $IMAGE_URI${NC}"
}

# Function to create GPU VM instance
create_instance() {
    echo -e "${BLUE}Creating GPU-enabled VM instance...${NC}"

    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
        echo -e "${YELLOW}Instance $INSTANCE_NAME already exists.${NC}"
        read -p "Delete and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --quiet
        else
            echo "Keeping existing instance."
            return
        fi
    fi

    # Create the instance with GPU
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --maintenance-policy=TERMINATE \
        --image-family=cos-stable \
        --image-project=cos-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --metadata=startup-script='#!/bin/bash
# Install NVIDIA drivers
cos-extensions install gpu
# Pull and run the container
docker-credential-gcr configure-docker
docker pull gcr.io/'$PROJECT_ID'/tom-nas-gpu:latest
docker run -d \
    --name tom-nas \
    --gpus all \
    -p 80:8000 \
    --restart unless-stopped \
    gcr.io/'$PROJECT_ID'/tom-nas-gpu:latest
' \
        --tags=http-server,https-server \
        --scopes=https://www.googleapis.com/auth/cloud-platform

    echo -e "${GREEN}Instance created: $INSTANCE_NAME${NC}"
}

# Function to create firewall rule
create_firewall() {
    echo -e "${BLUE}Creating firewall rule for HTTP access...${NC}"

    # Check if rule exists
    if gcloud compute firewall-rules describe allow-tom-nas-http --project=$PROJECT_ID &>/dev/null; then
        echo "Firewall rule already exists."
    else
        gcloud compute firewall-rules create allow-tom-nas-http \
            --project=$PROJECT_ID \
            --allow=tcp:80,tcp:8000 \
            --target-tags=http-server \
            --description="Allow HTTP access to ToM-NAS"
    fi

    echo -e "${GREEN}Firewall configured.${NC}"
}

# Function to get instance IP
get_instance_ip() {
    echo -e "${BLUE}Getting instance external IP...${NC}"

    # Wait for instance to be running
    echo "Waiting for instance to start..."
    sleep 10

    EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

    echo -e "${GREEN}Instance IP: $EXTERNAL_IP${NC}"
}

# Function to wait for application
wait_for_app() {
    echo -e "${BLUE}Waiting for application to start (this may take 2-3 minutes)...${NC}"

    for i in {1..30}; do
        if curl -s "http://$EXTERNAL_IP/api/status" &>/dev/null; then
            echo -e "${GREEN}Application is ready!${NC}"
            return 0
        fi
        echo "  Waiting... ($i/30)"
        sleep 10
    done

    echo -e "${YELLOW}Application may still be starting. Check in a few minutes.${NC}"
}

# Function to display final info
show_info() {
    echo ""
    echo -e "${GREEN}=============================================="
    echo "  Deployment Complete!"
    echo "==============================================${NC}"
    echo ""
    echo -e "  ${BLUE}Application URL:${NC} http://$EXTERNAL_IP"
    echo ""
    echo "  To check logs:"
    echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='docker logs tom-nas'"
    echo ""
    echo "  To stop the instance (save costs):"
    echo "    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo "  To delete everything:"
    echo "    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo -e "${YELLOW}  Note: GPU instances cost ~\$0.35/hour. Stop when not in use!${NC}"
    echo ""
}

# Main deployment flow
main() {
    check_gcloud

    echo -e "${YELLOW}This will deploy ToM-NAS to Google Cloud with GPU support.${NC}"
    echo -e "${YELLOW}Estimated cost: ~\$0.35/hour while running.${NC}"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    enable_apis
    build_image
    create_firewall
    create_instance
    get_instance_ip
    wait_for_app
    show_info
}

# Parse arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    build)
        check_gcloud
        build_image
        ;;
    status)
        check_gcloud
        gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
        ;;
    stop)
        check_gcloud
        gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
        echo "Instance stopped."
        ;;
    start)
        check_gcloud
        gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
        get_instance_ip
        echo "Instance started. URL: http://$EXTERNAL_IP"
        ;;
    delete)
        check_gcloud
        read -p "Delete instance $INSTANCE_NAME? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --quiet
            echo "Instance deleted."
        fi
        ;;
    logs)
        check_gcloud
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='docker logs -f tom-nas'
        ;;
    ssh)
        check_gcloud
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
        ;;
    *)
        echo "Usage: $0 {deploy|build|status|stop|start|delete|logs|ssh}"
        exit 1
        ;;
esac
