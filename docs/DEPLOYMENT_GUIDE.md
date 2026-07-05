# TripAI — Production Deployment Guide

This guide describes how to deploy the TripAI Travel Intelligence platform on various web services.

---

## ☁️ 1. Streamlit Community Cloud (Recommended & Fastest)

Streamlit Community Cloud is a free hosting platform for Streamlit projects.

### Step-by-Step Deployment:
1. **GitHub Setup**: Ensure all code, models, data, and configs are pushed to your GitHub repository.
2. **Account Creation**: Sign up/Log in to [Streamlit Share](https://share.streamlit.io/) using your GitHub account.
3. **App Deployment**:
   - Click **New app**.
   - Select your repository, branch (`main`), and main file path (`app.py`).
   - Click **Deploy**.
4. **Environment Variables**:
   - Click **Settings** in the Streamlit share panel.
   - Go to **Secrets** and add your Google Maps API key (optional):
     ```toml
     GOOGLE_MAPS_API_KEY = "your_google_maps_api_key_here"
     ```

---

## 🐳 2. Docker Deployment

Deploying with Docker containerizes the application, ensuring it runs identically on any environment.

### 1. Create a `Dockerfile` at the project root:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run:
```bash
# Build image
docker build -t tripai:latest .

# Run container
docker run -p 8501:8501 --env GOOGLE_MAPS_API_KEY="your_key" tripai:latest
```

---

## 🌐 3. AWS EC2 (Virtual Machine)

For full control over deployment scaling.

### Step-by-Step:
1. **Provision EC2 Instance**: Launch an Ubuntu instance (t2.micro is sufficient). Expose port `8501` in your security group.
2. **Clone & Setup**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-dev git
   git clone https://github.com/Srujanaaddanki/TravelTripBudgetPrediction.git
   cd TravelTripBudgetPrediction
   pip3 install -r requirements.txt
   ```
3. **Run in Background** (using screen or systemd):
   ```bash
   # Run screen session
   screen -S tripai
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   # Detach: Ctrl + A, then D
   ```

---

## 🎨 4. Render Deployment

Render is a modern cloud platform that builds apps directly from GitHub.

### Step-by-Step:
1. Log in to [Render](https://render.com/).
2. Create a **New Web Service** and link your GitHub repo.
3. Configure settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Add environment variables in Render's settings tab if required.
