#!/bin/bash

# CancerAg Inference App - Google Cloud Run Deployment Script
# This script automates the deployment process

set -e  # Exit on error

echo "======================================"
echo "  CancerAg Cloud Run Deployment"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found!${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo -e "${GREEN}✅ gcloud CLI found${NC}"

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged in to Google Cloud${NC}"
    echo "Running: gcloud auth login"
    gcloud auth login
fi

echo -e "${GREEN}✅ Authenticated${NC}"

# Get or create project
echo ""
echo "Select or create a Google Cloud project:"
echo ""
gcloud projects list 2>/dev/null || echo "No projects found"
echo ""
read -p "Enter project ID (or press Enter to create new): " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    # Generate random project ID
    RANDOM_ID=$(openssl rand -hex 4)
    PROJECT_ID="cancerag-${RANDOM_ID}"
    echo -e "${YELLOW}Creating new project: $PROJECT_ID${NC}"
    gcloud projects create "$PROJECT_ID" --name="CancerAg Inference" || {
        echo -e "${RED}Failed to create project. Try a different ID.${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ Project created: $PROJECT_ID${NC}"
fi

# Set project
gcloud config set project "$PROJECT_ID"
echo -e "${GREEN}✅ Using project: $PROJECT_ID${NC}"

# Enable required APIs
echo ""
echo -e "${YELLOW}Enabling required APIs (this may take 1-2 minutes)...${NC}"
gcloud services enable run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

echo -e "${GREEN}✅ APIs enabled${NC}"

# Select region
echo ""
echo "Select a region (closer to your users = better performance):"
echo "  1) us-central1 (Iowa, USA)"
echo "  2) us-east1 (South Carolina, USA)"
echo "  3) europe-west1 (Belgium)"
echo "  4) asia-east1 (Taiwan)"
echo "  5) Other (manual entry)"
read -p "Enter choice [1-5]: " REGION_CHOICE

case $REGION_CHOICE in
    1) REGION="us-central1" ;;
    2) REGION="us-east1" ;;
    3) REGION="europe-west1" ;;
    4) REGION="asia-east1" ;;
    5)
        read -p "Enter region (e.g., us-west1): " REGION
        ;;
    *) REGION="us-central1" ;;
esac

gcloud config set run/region "$REGION"
echo -e "${GREEN}✅ Region set to: $REGION${NC}"

# Service configuration
SERVICE_NAME="cancerag-inference"
MEMORY="4Gi"
CPU="2"
MAX_INSTANCES="10"
MIN_INSTANCES="0"

echo ""
echo "Deployment Configuration:"
echo "  Service Name: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Memory: $MEMORY"
echo "  CPU: $CPU vCPUs"
echo "  Max Instances: $MAX_INSTANCES"
echo "  Min Instances: $MIN_INSTANCES (scales to zero)"
echo ""

read -p "Proceed with deployment? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Deploy to Cloud Run
echo ""
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"
echo "This will take 5-15 minutes (building Docker image with models and receptors)"
echo ""

# First, build and push the image using Cloud Build
echo "Step 1: Building Docker image with Cloud Build..."
gcloud builds submit \
    --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME" \
    --dockerfile inference_app/Dockerfile \
    --timeout=1800s \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

echo ""
echo "Step 2: Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/$PROJECT_ID/$SERVICE_NAME" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --timeout 3600 \
    --max-instances "$MAX_INSTANCES" \
    --min-instances "$MIN_INSTANCES" \
    --port 8080 \
    --quiet

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo -e "  ✅ Deployment Successful!"
    echo -e "======================================${NC}"
    echo ""

    # Get service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format='value(status.url)')

    echo -e "${GREEN}🌐 Your app is live at:${NC}"
    echo -e "${GREEN}   $SERVICE_URL${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Visit the URL to test your app"
    echo "  2. Check logs: gcloud run services logs tail $SERVICE_NAME --region $REGION"
    echo "  3. Add custom domain (see GOOGLE_CLOUD_RUN_DEPLOYMENT.md)"
    echo ""
    echo "Cost monitoring:"
    echo "  - View usage: https://console.cloud.google.com/run"
    echo "  - Estimated cost: \$5-40/month (depends on usage)"
    echo ""
else
    echo -e "${RED}❌ Deployment failed!${NC}"
    echo "Check logs with: gcloud builds log \$(gcloud builds list --limit=1 --format='value(ID)')"
    exit 1
fi
