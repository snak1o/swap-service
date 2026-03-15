# RunPod Serverless Deployment Guide

To deploy the custom `bodyswap-gpu` image to RunPod:

1.  **Go to RunPod -> Serverless -> Templates**
    *   Click "New Template"
    *   Name: `BodySwap-GPU`
    *   Container Image: `snak1o/bodyswap-gpu:latest` (or your Docker Hub username)
    *   Container Disk: `30 GB` (models are large)
    *   Save Template

2.  **Go to RunPod -> Serverless -> Endpoints**
    *   Click "New Endpoint"
    *   Name: `swap-pipeline`
    *   Select Template: `BodySwap-GPU` (the one you just created)
    *   Active GPUs: A40 or A100 (A40 is good for MVP cost/performance)
    *   Workers: Min `0` (saves money), Max `3` (or as needed)
    *   Idle Timeout: `60` seconds
    *   Deploy!

3.  **Get Credentials**
    *   Copy the `Endpoint ID` from your new endpoint's page.
    *   Go to RunPod Settings -> API Keys and create a new key.
    *   Add these to your local `.env` file:
        ```env
        RUNPOD_API_KEY=your_key_here
        RUNPOD_ENDPOINT_ID=your_endpoint_id_here
        ```
