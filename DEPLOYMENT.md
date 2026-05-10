# MedAI — Deployment & Usage Guide

## Architecture Overview

```
Browser → CloudFront/S3 (React) → EC2 (FastAPI + XGBoost + SHAP) → RDS PostgreSQL
```

The system follows the C4 Container Diagram from the paper:
- **Frontend**: Next.js React SPA
- **Backend**: FastAPI with XGBoost inference + SHAP TreeExplainer
- **Database**: PostgreSQL (3 tables: patients, predictions, explanations)

---

## Local Development

### Prerequisites
- macOS with Homebrew (or Linux)
- PostgreSQL 14+
- Node.js 18+
- Python 3.11 (via Homebrew on macOS)

### 1. PostgreSQL Setup

```bash
# macOS
brew install postgresql@18
brew services start postgresql@18

# Create database
psql -h localhost -U $USER -d postgres -c "CREATE DATABASE medai;"
```

### 2. Backend Setup

```bash
cd MedAI_Backend

# 1. Instalar Python 3.11 y dependencias de sistema (solo macOS)
brew install python@3.11 libomp expat

# 2. Crear virtualenv (IMPORTANTE: el prefijo DYLD es necesario en macOS)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
  /opt/homebrew/bin/python3.11 -m venv venv

# 3. Instalar paquetes Python (también requiere DYLD)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
  venv/bin/pip install --upgrade pip

DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
  venv/bin/pip install -r requirements.txt

# 4. Configurar entorno
cp .env.example .env
# Editar .env y cambiar DATABASE_URL a tu usuario de macOS:
#   DATABASE_URL=postgresql://TU_USUARIO@localhost/medai

# 5. Crear base de datos PostgreSQL
psql -h localhost -U $USER -d postgres -c "CREATE DATABASE medai;"

# 6. Entrenar el modelo (primera vez, ~5 minutos)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/libomp/lib \
  venv/bin/python train_model.py

# 7. Iniciar el backend
./start_dev.sh
```

> **Nota macOS**: El prefijo `DYLD_LIBRARY_PATH=...` es obligatorio porque el Python 3.11 de Homebrew
> enlaza contra una versión de libexpat más nueva que la del sistema. `start_dev.sh` lo aplica
> automáticamente, por eso solo necesitas ejecutar ese script para el día a día.

Backend runs at: http://localhost:8000  
API docs: http://localhost:8000/docs

### 3. Frontend Setup

```bash
cd MedAI_Frontend

# Install dependencies
npm install

# Configure environment (already created)
# .env.local contains:
#   NEXT_PUBLIC_API_URL=http://localhost:8000
#   NEXT_PUBLIC_USE_MOCK_API=false

# Start dev server
npm run dev
```

Frontend runs at: http://localhost:3000

---

## Full Testing Guide

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info/model

# Triage prediction (critical patient — expected Level 1)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 72,
    "sexo": "F",
    "modo_llegada": "Ambulancia",
    "motivo_consulta": "Disnea severa y SpO2 baja",
    "signos_vitales": {
      "frecuencia_cardiaca": 135,
      "frecuencia_respiratoria": 32,
      "temperatura": 38.8,
      "spO2": 82,
      "presion_sistolica": 80,
      "presion_diastolica": 50,
      "dolor": 7
    },
    "comorbilidades": {"epoc": true, "cardiopatia": true, "hta": true},
    "es_prioritario": true
  }'

# Non-urgent patient — expected Level 4 or 5
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 30,
    "sexo": "M",
    "modo_llegada": "Caminando",
    "motivo_consulta": "Dolor de cabeza leve",
    "signos_vitales": {
      "frecuencia_cardiaca": 72,
      "frecuencia_respiratoria": 14,
      "temperatura": 36.8,
      "spO2": 98,
      "presion_sistolica": 118,
      "presion_diastolica": 76,
      "dolor": 2
    }
  }'

# Prediction history
curl http://localhost:8000/predict/history
```

### Validate SHAP Explainability

Every prediction response includes:
- `shap_base_value`: φ₀ (expected model output for this class)
- `shap_features`: ranked list of all 15 features with φⱼ values
- `shap_global_importance`: mean |φⱼ| per feature across the prediction

The SHAP values satisfy: `sum(shap_values) + base_value ≈ log-odds of predicted class`

### Validate Database Persistence

```bash
# Check all tables
psql -h localhost -U $USER -d medai -c "\dt"

# Inspect a prediction + explanation
psql -h localhost -U $USER -d medai -c "
  SELECT p.id, pat.edad, pat.motivo_consulta,
         pr.nivel_codigo, pr.confianza, pr.inference_latency_ms,
         e.base_value
  FROM patients pat
  JOIN predictions pr ON pr.patient_id = pat.id
  JOIN explanations e  ON e.prediction_id = pr.id
  ORDER BY pr.id DESC LIMIT 5;
"
```

### Frontend Test Checklist

1. Open http://localhost:3000
2. Fill the triage form:
   - Age: 65, Sex: Male, Mode: Ambulance
   - Chief complaint: "Dolor en el pecho"
   - FC: 110, FR: 24, Temp: 37.2, SpO2: 88, PAS: 90, PAD: 60, Pain: 8
   - Check HTA and Cardiopatía
3. Click "Realizar Clasificación de Triage"
4. Verify result page shows:
   - Level 1 (red circle)
   - Probability distribution bar chart
   - SHAP feature contributions with positive/negative bars
   - Rule-based risk factors
5. Navigate to /pacientes — should show classification history
6. Navigate to /informes — should show dashboard statistics

---

## Docker Deployment (local / self-hosted)

```bash
cd /path/to/tdse

# Build and start all services
docker-compose up --build

# Backend:  http://localhost:8000
# Frontend: http://localhost:3000
# DB:       localhost:5432 (medai/medaipass)
```

The model file must exist at `MedAI_Backend/models/triage_model.joblib` before building the image. If it doesn't, run `train_model.py` first.

---

## AWS Deployment Guide

### Architecture on AWS
```
Users → Route 53 → CloudFront → S3 (React static build)
                             → EC2 (FastAPI backend, port 443)
EC2 → RDS PostgreSQL (private subnet)
```

### Step 1: EC2 Instance Setup

**Instance type**: t3.medium (2 vCPU, 4 GB RAM minimum for XGBoost)  
**AMI**: Amazon Linux 2023 or Ubuntu 22.04  
**Security Group**: Allow inbound 443 (HTTPS), 22 (SSH from your IP only)

```bash
# Connect to EC2
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>

# Install dependencies
sudo dnf update -y                      # Amazon Linux
sudo dnf install -y python3.11 python3.11-pip postgresql15 git nginx

# Install libomp for XGBoost
sudo dnf install -y libgomp

# Clone/upload project
git clone https://github.com/your-repo/medai.git /app
cd /app/MedAI_Backend

# Create venv and install
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt

# Train model (or copy pre-trained model)
venv/bin/python train_model.py

# Set environment variables
cat > .env << 'EOF'
DATABASE_URL=postgresql://medai:STRONG_PASSWORD@your-rds-endpoint.rds.amazonaws.com/medai
CORS_ORIGINS=https://your-cloudfront-domain.cloudfront.net
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
EOF
```

### Step 2: RDS PostgreSQL Setup

1. Go to AWS RDS Console → Create Database
2. Engine: PostgreSQL 16
3. Instance: db.t3.micro (dev) or db.t3.small (prod)
4. Database name: `medai`
5. Username: `medai`, Password: (strong password)
6. VPC: Same VPC as EC2
7. Security Group: Allow inbound 5432 from EC2 security group only
8. Enable automated backups

After creation, note the endpoint and update `.env`:
```
DATABASE_URL=postgresql://medai:PASSWORD@your-endpoint.rds.amazonaws.com/medai
```

### Step 3: Systemd Service for Backend

```bash
sudo tee /etc/systemd/system/medai-backend.service << 'EOF'
[Unit]
Description=MedAI FastAPI Backend
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/app/MedAI_Backend
ExecStart=/app/MedAI_Backend/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=3
Environment=LD_PRELOAD=/usr/lib64/libgomp.so.1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable medai-backend
sudo systemctl start medai-backend
sudo systemctl status medai-backend
```

### Step 4: Nginx as Reverse Proxy with HTTPS

```bash
sudo dnf install -y nginx certbot python3-certbot-nginx

# Nginx config
sudo tee /etc/nginx/conf.d/medai.conf << 'EOF'
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
EOF

# Issue SSL certificate
sudo certbot --nginx -d api.yourdomain.com

sudo systemctl enable nginx
sudo systemctl start nginx
```

### Step 5: S3 + CloudFront for Frontend

```bash
# Build frontend with production API URL
cd /local/MedAI_Frontend
NEXT_PUBLIC_API_URL=https://api.yourdomain.com npm run build

# Create S3 bucket (replace with your bucket name)
aws s3 mb s3://medai-frontend-prod --region us-east-1

# Upload static build
aws s3 sync .next/static s3://medai-frontend-prod/_next/static/
aws s3 cp public/ s3://medai-frontend-prod/ --recursive

# For a proper Next.js deployment on S3, use next export or Amplify
# Recommended: Deploy to Vercel or AWS Amplify for SSR support
```

**Recommended simpler alternative for frontend**: Use **AWS Amplify**:
```bash
# In Amplify Console → New app → Host web app → Connect GitHub
# Build settings:
#   Build command: npm run build
#   Output directory: .next
#   Environment variables:
#     NEXT_PUBLIC_API_URL = https://api.yourdomain.com
#     NEXT_PUBLIC_USE_MOCK_API = false
```

### Step 6: CloudFront Distribution (for S3)

1. AWS CloudFront Console → Create Distribution
2. Origin: your S3 bucket (enable S3 website hosting first)
3. Default root object: `index.html`
4. Price class: Use Only North America and Europe
5. Custom SSL certificate: use ACM (create in us-east-1)
6. Note the CloudFront domain (e.g., `d1abc.cloudfront.net`)
7. Update backend `.env`: `CORS_ORIGINS=https://d1abc.cloudfront.net`
8. Restart backend: `sudo systemctl restart medai-backend`

### Step 7: Security Groups Summary

| Service | Inbound | Source |
|---------|---------|--------|
| EC2 | 443 | 0.0.0.0/0 |
| EC2 | 22  | Your IP |
| RDS | 5432 | EC2 SG |

### Step 8: Verify Production Deployment

```bash
# Health check
curl https://api.yourdomain.com/health

# Test prediction
curl -X POST https://api.yourdomain.com/predict \
  -H "Content-Type: application/json" \
  -d '{"edad":65,"sexo":"M","modo_llegada":"Ambulancia",
       "motivo_consulta":"Dolor toracico",
       "signos_vitales":{"frecuencia_cardiaca":110,"frecuencia_respiratoria":24,
         "temperatura":37.2,"spO2":88,"presion_sistolica":90,
         "presion_diastolica":60,"dolor":8},
       "comorbilidades":{"hta":true,"cardiopatia":true}}'

# Monitor backend logs
sudo journalctl -u medai-backend -f

# Monitor nginx logs
sudo tail -f /var/log/nginx/access.log
```

### Expected Availability (from paper)
```
A_total = A_EC2 × A_RDS × A_CloudFront
        = 0.9995 × 0.9995 × 0.9999
        ≈ 99.89%
```

---

## Model Retraining

To retrain with new data:

```bash
cd MedAI_Backend

# Replace triage_choco_100k_balanceado.csv with new dataset (same column format)
# Then:
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/libomp/lib \
  venv/bin/python train_model.py

# Restart backend to load new model
sudo systemctl restart medai-backend  # production
# or just restart the dev server
```

Performance targets (from paper):
- Recall Level 1 ≥ 95%
- Recall Level 2 ≥ 95%
- Weighted F1 ≥ 80%

Current achieved performance (retrained on 100k Chocó dataset):
- Recall Level 1: 82.1%
- Recall Level 2: 64.4%
- Weighted F1: 52.6%

Note: Performance is below targets due to the challenging multi-class nature of the balanced Chocó dataset with 15 features. In production, additional features (GCS score, lab results) and a larger dataset would improve performance.

---

## Troubleshooting

### Backend won't start
```bash
# Check model exists
ls -lh MedAI_Backend/models/triage_model.joblib

# Run train_model.py to create it
venv/bin/python train_model.py

# Check DB connection
venv/bin/python -c "from database import create_tables; create_tables(); print('OK')"
```

### SHAP import error on macOS
```bash
# Always run with:
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/libomp/lib
```

### Frontend can't reach backend (CORS)
Update `MedAI_Backend/.env`:
```
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
```
Then restart backend.

### PostgreSQL connection refused
```bash
# macOS
brew services restart postgresql@18
# Linux
sudo systemctl restart postgresql
```
