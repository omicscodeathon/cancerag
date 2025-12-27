# Render.com Deployment Guide for CancerAg Inference App

This guide provides step-by-step instructions for deploying the CancerAg GPCR ligand bias prediction application to Render.com.

## Overview

The application has been configured to bundle all necessary models, preprocessing artifacts, and receptor structures directly into the Docker image. This eliminates the need for volume mounts and makes deployment straightforward on cloud platforms.

**What's Included:**
- ✅ 4 trained models (random_forest, xgboost, lightgbm, logistic_regression)
- ✅ Preprocessing artifacts (scaler, metadata, label encoder)
- ✅ All GPCR receptor structures (~214MB for full docking features)
- ✅ Gradio web interface on port 7860

**Expected Docker Image Size:** ~700-900MB

**Expected Startup Time:** 60-120 seconds (loading models + receptors)

---

## Prerequisites

### 1. Render.com Account
- Sign up at [render.com](https://render.com/)
- Free trial available, but production deployment requires paid plan

### 2. GitHub Repository
- Push your code to GitHub (or GitLab/Bitbucket)
- Ensure all required files are committed:
  ```bash
  git status
  # Should show clean working tree after committing
  ```

### 3. Required Files Checklist

Verify these files exist and are tracked in git:

**Models** (in `results/models/`):
- [ ] `random_forest.pkl` (4.8MB)
- [ ] `xgboost.pkl` (2.3MB)
- [ ] `lightgbm.pkl` (5.5MB)
- [ ] `logistic_regression.pkl` (16KB)

**Preprocessing** (in `data/processed/ml_preprocessed/`):
- [ ] `scaler.pkl`
- [ ] `preprocessing_metadata.json`
- [ ] `label_encoder.pkl`

**Receptors** (in `data/processed/`):
- [ ] `receptors/` directory with .pdb and .pdbqt files
- [ ] `binding_sites.json`

**Code** (in `inference_app/`):
- [ ] `app.py`
- [ ] `Dockerfile`
- [ ] `requirements.txt`
- [ ] `src/` directory with all modules

**Configuration**:
- [ ] `render.yaml` (in repository root)

---

## Step 1: Verify Git Tracking

First, ensure essential files are no longer ignored:

```bash
# Check that models and data are tracked
git status

# You should see these files as tracked (not ignored):
git ls-files | grep -E "(random_forest.pkl|scaler.pkl|receptors/)"

# If files are still ignored, force add them:
git add -f results/models/*.pkl
git add -f data/processed/ml_preprocessed/*.pkl
git add -f data/processed/ml_preprocessed/*.json
git add -f data/processed/receptors/
git add -f data/processed/binding_sites.json
```

---

## Step 2: Commit and Push Changes

```bash
# Add all deployment configuration files
git add .gitignore
git add inference_app/Dockerfile
git add render.yaml
git add inference_app/RENDER_DEPLOYMENT.md

# Add the Render deployment docs
git add inference_app/RENDER_DEPLOYMENT.md

# Commit everything
git commit -m "Configure Docker bundling for Render deployment

- Update .gitignore to allow essential models and data
- Modify Dockerfile to copy models/receptors into image
- Add render.yaml for automated deployment
- Increase health check start-period to 120s for model loading"

# Push to GitHub
git push origin main
```

---

## Step 3: Deploy to Render

### Option A: Using Blueprint (Recommended - Automated)

1. **Navigate to Render Dashboard:**
   - Go to [dashboard.render.com](https://dashboard.render.com/)
   - Click "New +" → "Blueprint"

2. **Connect Repository:**
   - Select "Connect a repository"
   - Authorize GitHub access
   - Choose your `cancerag` repository

3. **Render Detects Configuration:**
   - Render will automatically detect `render.yaml`
   - Review the configuration:
     - Service name: `cancerag-inference`
     - Plan: **Standard** ($25/month recommended)
     - Region: Oregon (or closest to your users)

4. **Deploy:**
   - Click "Apply"
   - Render will begin building the Docker image
   - Watch the logs for progress

### Option B: Manual Service Creation

1. **Create Web Service:**
   - Click "New +" → "Web Service"
   - Connect your repository

2. **Configure Service:**
   - **Name:** `cancerag-inference`
   - **Environment:** Docker
   - **Dockerfile Path:** `./inference_app/Dockerfile`
   - **Docker Context:** `.` (repository root)
   - **Plan:** Standard ($25/month)
   - **Region:** Oregon

3. **Environment Variables:**
   Add these variables:
   ```
   GRADIO_SERVER_NAME=0.0.0.0
   GRADIO_SERVER_PORT=7860
   BASE_PATH=/app
   GRADIO_ANALYTICS_ENABLED=false
   PYTHONUNBUFFERED=1
   ```

4. **Advanced Settings:**
   - **Health Check Path:** `/`
   - **Auto-Deploy:** Yes
   - **Min Instances:** 1
   - **Max Instances:** 3

5. **Create Service:**
   - Click "Create Web Service"
   - Render begins building

---

## Step 4: Monitor Deployment

### Build Phase (5-10 minutes)

Watch the logs for these stages:

```
1. Cloning repository...
2. Building Docker image...
   - Installing system dependencies (openbabel, wget)
   - Downloading AutoDock Vina
   - Installing Python packages
   - Copying models (this will take time due to 214MB receptors)
   - Copying receptors
3. Pushing image to registry...
4. Deploying...
```

### Startup Phase (60-120 seconds)

Look for these log messages:

```
INFO - Initializing app components...
INFO - Loading model: random_forest
INFO - Receptor manager initialized with 50+ receptors
INFO - App initialized successfully
Running on local URL:  http://0.0.0.0:7860
```

### Health Check

Render will check `http://your-app.onrender.com/` every 30 seconds. The service is healthy when:
- HTTP 200 response received
- Gradio interface loads

---

## Step 5: Access Your Application

Once deployed, Render provides a public URL:

```
https://cancerag-inference.onrender.com
```

**First Visit:**
- May take a few seconds to load (models already loaded globally)
- You'll see the Gradio interface with two tabs:
  - 🔬 Interactive Prediction
  - ℹ️ About

**Test Prediction:**
1. Select a receptor (e.g., "ADRB2 - Beta-2 Adrenergic Receptor")
2. Enter a SMILES string (e.g., `CC(C)NCC(O)c1ccc(O)c(O)c1` for isoprenaline)
3. Optional: Enable "Run Docking Analysis"
4. Click "Run Prediction"

Expected response time: 1-2 seconds (or 10-30 seconds with docking)

---

## Pricing & Resource Planning

### Recommended Plan: Standard ($25/month)

**Specifications:**
- 2GB RAM
- 1 CPU
- Always-on (1 min instance configured)
- No cold starts
- Suitable for production traffic

**Why Standard over Starter?**
- Starter (512MB RAM) may run out of memory with:
  - All receptors loaded (~214MB)
  - 4 models in memory (~15MB)
  - Multiple concurrent predictions
- Standard provides comfortable headroom

### Cost Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| Base plan | $25/month | Standard plan |
| Persistent disk | $0.25/month | 1GB for logs |
| Bandwidth | Free | Generous allowance |
| **Total** | **~$25/month** | Fixed cost, no usage charges |

### Monitoring Resource Usage

After deployment, check metrics:

1. **Memory Usage:**
   - Go to Service → Metrics
   - Watch "Memory" graph
   - Should stay below 1.5GB for Standard plan
   - If consistently above 1.8GB, upgrade to Pro

2. **CPU Usage:**
   - Should be low when idle
   - Spikes during predictions are normal

3. **Response Time:**
   - Target: <3 seconds per prediction (without docking)
   - With docking: 10-30 seconds

---

## Scaling & Performance Optimization

### Current Configuration

```yaml
scaling:
  minInstances: 1  # One instance always running
  maxInstances: 3  # Auto-scale to 3 under load
```

**How it works:**
- Base load: 1 instance handles requests
- Traffic spike: Render spins up to 3 instances
- Low traffic: Scales back to 1 instance

### When to Adjust

**Increase minInstances to 2-3 if:**
- You have >100 users/day
- Peak traffic exceeds 10 concurrent users
- Response times degrade during busy periods

**Decrease to 0 if:**
- You're okay with 60-120 second cold starts
- Traffic is very sporadic (<10 predictions/day)
- Budget is tight (saves ~$15/month on Starter plan)

---

## Troubleshooting

### Issue: Build fails with "COPY failed"

**Cause:** Models/data files not tracked in git

**Solution:**
```bash
# Check if files are ignored
git status

# Force add if needed
git add -f results/models/*.pkl
git add -f data/processed/receptors/

# Commit and push
git commit -m "Add model and receptor files"
git push
```

### Issue: Health check failing, service keeps restarting

**Cause:** Application taking too long to start

**Check logs for:**
- Model loading errors
- Missing files
- Python errors

**Solutions:**
1. Verify all required files are in the image
2. Check Dockerfile health check `start-period` (should be 120s)
3. Review Render logs for specific error messages

### Issue: Out of memory errors

**Symptoms:**
```
MemoryError: Unable to allocate array
Killed (signal 9)
```

**Solutions:**
1. **Upgrade to Pro plan** (4GB RAM, 2 CPU) - $85/month
2. **Reduce bundled models:**
   - Remove heavy models (gradient_boosting, stacking_ensemble)
   - Keep only random_forest and xgboost
3. **Disable docking by default:**
   - Edit `app.py` line 74: `enable_docking=False`

### Issue: Slow response times (>5 seconds)

**Causes:**
- Too many concurrent requests
- Insufficient CPU
- Docking enabled on large receptors

**Solutions:**
1. Upgrade to Pro plan (2 CPUs)
2. Increase max instances to 5-10
3. Disable docking or limit to small receptors
4. Add caching for repeated SMILES

### Issue: Build takes >20 minutes

**Cause:** Large receptor files (214MB) being copied

**Expected:**
- First build: 10-15 minutes (normal for large files)
- Subsequent: 5-10 minutes (Docker layer caching)

**Not a problem unless build timeout errors occur**

---

## Post-Deployment Checklist

After successful deployment:

- [ ] Access the public URL and verify Gradio loads
- [ ] Test prediction with a known SMILES
- [ ] Test receptor selection (existing, PDB ID, upload)
- [ ] Verify docking works (if enabled)
- [ ] Check memory usage in Render metrics
- [ ] Set up monitoring/alerts (optional)
- [ ] Configure custom domain (optional)
- [ ] Enable HTTPS (automatic with Render)

---

## Custom Domain (Optional)

To use your own domain (e.g., `inference.yourdomain.com`):

1. **In Render Dashboard:**
   - Go to Service → Settings → Custom Domains
   - Click "Add Custom Domain"
   - Enter your domain

2. **In Your DNS Provider:**
   - Add CNAME record:
     ```
     inference.yourdomain.com → cancerag-inference.onrender.com
     ```

3. **SSL Certificate:**
   - Render automatically provisions Let's Encrypt SSL
   - HTTPS enabled within minutes

---

## Updating the Application

### Code Changes

```bash
# Make changes to app.py or other code
git add inference_app/app.py
git commit -m "Update prediction logic"
git push

# Render auto-deploys (if enabled)
# Build time: ~5 minutes (uses cached layers)
```

### Model Changes

```bash
# Replace model file
cp new_model.pkl results/models/random_forest.pkl

# Commit and push
git add results/models/random_forest.pkl
git commit -m "Update random forest model"
git push

# Render rebuilds with new model
# Build time: ~8 minutes (model layer invalidated)
```

---

## Alternative Deployment: Render Web Services CLI

For advanced users, deploy via CLI:

```bash
# Install Render CLI
brew install render  # macOS
# or download from https://render.com/docs/cli

# Login
render login

# Deploy
render blueprint launch

# Monitor
render logs -f cancerag-inference
```

---

## Cost Optimization Tips

1. **Use Starter Plan for Testing:**
   - Start with Starter ($7/month)
   - Remove receptors if memory constrained
   - Upgrade to Standard when ready for production

2. **Pause During Development:**
   - Suspend service when not actively using
   - No charges while suspended
   - Resume instantly when needed

3. **Monitor Usage:**
   - Check Render metrics weekly
   - Downgrade if consistently under-utilized

4. **Free Tier for Staging:**
   - Create a second "staging" service on free tier
   - Remove receptors and heavy models
   - Use for testing changes before production deploy

---

## Next Steps

**After successful Render deployment**, consider:

1. **Set up Google Cloud Run** (for comparison):
   - See `DOCKER_DEPLOYMENT.md` section on Cloud Run
   - May offer better pricing for sporadic traffic

2. **Implement monitoring:**
   - Add logging to track prediction counts
   - Monitor model performance over time
   - Track response times

3. **Add features:**
   - User authentication (Gradio's `auth` parameter)
   - Batch prediction upload (CSV input)
   - Result export (download predictions)
   - API endpoint (for programmatic access)

4. **Optimize performance:**
   - Cache frequently predicted molecules
   - Add database for prediction history
   - Implement async processing for docking

---

## Support Resources

- **Render Documentation:** https://render.com/docs
- **Render Community:** https://community.render.com
- **Gradio Documentation:** https://gradio.app/docs
- **This Repository:** [Create an issue](https://github.com/yourusername/cancerag/issues)

---

## Summary

**Deployment Status:** Ready ✅

**Key Configuration:**
- Platform: Render.com
- Plan: Standard ($25/month)
- Models: Bundled in Docker image
- Receptors: All included (214MB)
- Expected uptime: 99.9%
- Expected latency: 1-2 seconds (without docking)

**URL:** `https://cancerag-inference.onrender.com` (after deployment)

**Next Action:** Push to GitHub and connect repository in Render dashboard!
