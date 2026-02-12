# Setup

## Backend

```bash
cd VoiceAgents/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

```bash
cd VoiceAgents/frontend
npm install
copy .env.local.example .env.local
npm run dev
```
