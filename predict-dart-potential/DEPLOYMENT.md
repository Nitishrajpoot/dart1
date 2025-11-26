# Deployment Guide

This guide covers deploying the DART Toxicity Prediction System to production.

## Local Deployment

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone Repository**
   \`\`\`bash
   git clone https://github.com/yourusername/dart-toxicity-prediction.git
   cd dart-toxicity-prediction
   \`\`\`

2. **Create Virtual Environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. **Install Dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Prepare Data and Train Model**
   \`\`\`bash
   # Fetch data
   python scripts/fetch_toxcast_data.py
   python scripts/fetch_dart_labels.py
   
   # Generate features
   python scripts/generate_features.py
   
   # Train model
   python scripts/train_models.py
   \`\`\`

5. **Run Application**
   \`\`\`bash
   streamlit run app/streamlit_app.py
   \`\`\`

6. **Access Application**
   - Open browser to `http://localhost:8501`

## Cloud Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**
   \`\`\`bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   \`\`\`

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app/streamlit_app.py`
   - Click "Deploy"

3. **Important Notes**
   - Pre-train your model locally
   - Commit the `data/models/` directory with trained models
   - Streamlit Cloud has memory limits (~1GB)
   - For large models, consider optimization or alternative hosting

### Heroku Deployment

1. **Create Procfile**
   \`\`\`
   web: streamlit run app/streamlit_app.py --server.port=$PORT
   \`\`\`

2. **Create runtime.txt**
   \`\`\`
   python-3.9.16
   \`\`\`

3. **Deploy to Heroku**
   \`\`\`bash
   heroku create dart-toxicity-app
   git push heroku main
   heroku open
   \`\`\`

### Docker Deployment

1. **Create Dockerfile**
   \`\`\`dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app/streamlit_app.py"]
   \`\`\`

2. **Build and Run**
   \`\`\`bash
   docker build -t dart-predictor .
   docker run -p 8501:8501 dart-predictor
   \`\`\`

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   - Ubuntu Server 20.04 LTS
   - t2.medium or larger (for model inference)
   - Open port 8501 in security group

2. **SSH and Setup**
   \`\`\`bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and dependencies
   sudo apt install python3-pip python3-venv -y
   
   # Clone and setup application
   git clone <your-repo>
   cd dart-toxicity-prediction
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   \`\`\`

3. **Run with systemd (production)**
   Create `/etc/systemd/system/dart-app.service`:
   \`\`\`ini
   [Unit]
   Description=DART Toxicity Predictor
   After=network.target
   
   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/dart-toxicity-prediction
   Environment="PATH=/home/ubuntu/dart-toxicity-prediction/venv/bin"
   ExecStart=/home/ubuntu/dart-toxicity-prediction/venv/bin/streamlit run app/streamlit_app.py
   
   [Install]
   WantedBy=multi-user.target
   \`\`\`
   
   Enable and start:
   \`\`\`bash
   sudo systemctl enable dart-app
   sudo systemctl start dart-app
   \`\`\`

### Google Cloud Run

1. **Create Dockerfile** (as above)

2. **Build and Deploy**
   \`\`\`bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/dart-predictor
   gcloud run deploy dart-predictor \
     --image gcr.io/PROJECT-ID/dart-predictor \
     --platform managed \
     --memory 2Gi
   \`\`\`

## Production Considerations

### Performance Optimization

1. **Model Optimization**
   - Use model quantization for smaller size
   - Implement model caching
   - Consider using ONNX runtime

2. **Caching**
   \`\`\`python
   @st.cache_data
   def expensive_computation(input_data):
       # Your computation
       pass
   \`\`\`

3. **Load Balancing**
   - Use NGINX for reverse proxy
   - Implement horizontal scaling

### Security

1. **HTTPS**
   - Use Let's Encrypt for SSL certificates
   - Configure reverse proxy (NGINX/Caddy)

2. **Rate Limiting**
   - Implement API rate limits
   - Use authentication for batch processing

3. **Input Validation**
   - Sanitize all user inputs
   - Validate SMILES strings
   - Limit file upload sizes

### Monitoring

1. **Logging**
   \`\`\`python
   import logging
   logging.basicConfig(level=logging.INFO)
   \`\`\`

2. **Analytics**
   - Track prediction counts
   - Monitor model performance drift
   - Log errors and exceptions

3. **Health Checks**
   - Implement `/health` endpoint
   - Monitor uptime and latency

### Maintenance

1. **Model Updates**
   - Version control for models
   - A/B testing new models
   - Automated retraining pipeline

2. **Data Updates**
   - Regular ToxCast data refreshes
   - Update training data quarterly
   - Track new chemicals

3. **Dependency Management**
   - Regular security updates
   - Pin critical package versions
   - Test updates in staging

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Increase instance memory
   - Optimize model size
   - Use batch processing for large inputs

2. **Slow Predictions**
   - Enable model caching
   - Use GPU acceleration if available
   - Optimize feature generation

3. **Import Errors**
   - Check Python version compatibility
   - Verify all dependencies installed
   - Use virtual environment

## Support

For deployment issues:
- Open GitHub issue
- Check documentation
- Contact maintainers

## License

See LICENSE file for deployment restrictions and terms.
