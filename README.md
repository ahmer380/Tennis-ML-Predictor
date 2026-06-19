# Tennis ML Predictor

Add good intro paragraph(s)

This repository includes:
- ...

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

## Train a model

Training is implemented in `src/train.py`.

```bash
python -m src.train
```

Supported executable parameters:
- `--model`: The type of model to train tennis data on, including `elo`, `mlp`, and `xgboost` (default).

## Tennis ML Predictor API
Coming soon!
