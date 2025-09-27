# Server Configuration Guide

This document explains how to configure the server endpoints that were previously hardcoded in the EditGRPO codebase.

## Server Placeholders

The following placeholders have been used to replace hardcoded IP addresses:

### 1. REWARD_SERVER_HOST
- **Purpose**: Main reward computation server
- **Default Port**: 5000
- **Endpoint**: `/predict`
- **Used in**: Configuration files, Python reward functions
- **Example**: `http://REWARD_SERVER_HOST:5000/predict`

### 2. EXTRACTOR_SERVER_HOST  
- **Purpose**: RaTEScore extractor server for post-rollout edits
- **Default Port**: 5001
- **Endpoint**: `/analyze`
- **Used in**: Rollout configurations, batch scripts
- **Example**: `http://EXTRACTOR_SERVER_HOST:5001/analyze`

### 3. GREEN_SERVER_HOST
- **Purpose**: GREEN score computation server
- **Default Port**: 5002
- **Endpoint**: `/predict`
- **Used in**: GREEN scoring functions
- **Example**: `http://GREEN_SERVER_HOST:5002/predict`

### 4. RATESCORE_SERVER_HOST
- **Purpose**: Alternative RaTEScore server
- **Default Port**: 5001
- **Endpoint**: `/analyze`
- **Used in**: RaTEScore extractor utilities
- **Example**: `http://RATESCORE_SERVER_HOST:5001/analyze`

## Configuration Methods

### Method 1: Environment Variables
Set environment variables before running the training:

```bash
export REWARD_SERVER_HOST="your-reward-server.example.com"
export EXTRACTOR_SERVER_HOST="your-extractor-server.example.com"
export GREEN_SERVER_HOST="your-green-server.example.com"
export RATESCORE_SERVER_HOST="your-ratescore-server.example.com"
```

### Method 2: Direct Replacement
Replace the placeholders directly in the configuration files with your actual server addresses:

```yaml
# In examples/*.yaml files
reward_server: 'http://your-actual-server.com:5000/predict'
```

### Method 3: Command Line Override
Override server addresses when running training scripts:

```bash
python3 -m verl.trainer.main \
    config=examples/editgrpo_mimic_cxr_2_5.yaml \
    worker.reward.reward_server='http://your-server.com:5000/predict' \
    worker.rollout.extractor_server='http://your-extractor.com:5001/analyze'
```

## Local Development

For local development, you can use localhost:

```bash
export REWARD_SERVER_HOST="localhost"
export EXTRACTOR_SERVER_HOST="localhost"
export GREEN_SERVER_HOST="localhost"
export RATESCORE_SERVER_HOST="localhost"
```

## Server Setup

Make sure your servers are running and accessible:

1. **Reward Server**: Should respond to POST requests at `/predict`
2. **Extractor Server**: Should respond to POST requests at `/analyze`
3. **GREEN Server**: Should respond to POST requests at `/predict`
4. **RaTEScore Server**: Should respond to POST requests at `/analyze`

Refer to the `flask_server/` directory for server implementation examples.
