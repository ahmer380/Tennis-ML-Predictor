# Tennis ML Predictor API

This is the FastAPI service for the Tennis ML Predictor project.

It exposes trained machine learning models (Elo, MLP, XGBoost) through a REST API to predict tennis match outcomes and retrieve player profiles built from historical ATP data.

**API is currently not deployed, but deployment is planned soon!**

## Setup

### Step 0: Ensure Python and pip are installed

```bash
python --version
pip --version
```

### Step 1: Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Start the FastAPI server

```bash
python -m uvicorn api.main:app --reload
```

## API Documentation

Once the server is running, open the Swagger UI playground at http://localhost:8000/docs#/ to explore and test the endpoints interactively. You can try different player matchups, adjust model settings, and experiment with inputs directly from the browser.