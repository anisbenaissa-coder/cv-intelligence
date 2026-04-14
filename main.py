from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import PyPDF2
import io
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

app = FastAPI(title="CV Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clé API Anthropic (tu la mets dans les variables d'environnement sur Render)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Limite gratuite : 1 CV par IP toutes les 24h
rate_limit_store = defaultdict(lambda: {"count": 0, "reset_at": datetime.now() + timedelta(hours=24)})

def check_rate_limit(ip: str) -> bool:
    data = rate_limit_store[ip]
    if datetime.now() > data["reset_at"]:
        data["count"] = 0
        data["reset_at"] = datetime.now() + timedelta(hours=24)
    if data["count"] >= 1:
        return False
    data["count"] += 1
    return True

def extract_text_from_pdf(content: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

@app.get("/")
def root():
    return {"message": "CV Intelligence API en ligne ✅"}

@app.post("/analyze")
async def analyze_cv(request: Request, file: UploadFile = File(...)):
    ip = request.client.host

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    if not check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail="Limite gratuite atteinte : 1 analyse par 24h. Passez au plan Pro pour 500 CV à 15€."
        )

    content = await file.read()

    if len(content) > 5 * 1024 * 1024:  # 5MB max
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (max 5MB).")

    cv_text = extract_text_from_pdf(content)

    if len(cv_text) < 100:
        raise HTTPException(status_code=400, detail="Le CV semble vide ou illisible. Vérifiez que le PDF contient du texte.")

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Analyse ce CV de façon professionnelle en français.
Retourne UNIQUEMENT un objet JSON valide, sans texte avant ni après, avec cette structure:
{{
  "score": <nombre entier entre 0 et 100>,
  "niveau": "<Junior|Confirmé|Senior|Expert>",
  "points_forts": ["<point fort 1>", "<point fort 2>", "<point fort 3>"],
  "points_amelioration": ["<amélioration 1>", "<amélioration 2>"],
  "resume": "<2 phrases qui résument le profil du candidat>",
  "metiers_compatibles": ["<métier 1>", "<métier 2>", "<métier 3>"],
  "conseil_rh": "<un conseil concret pour améliorer ce CV>"
}}

CV à analyser:
{cv_text[:4000]}"""
        }]
    )

    try:
        result = json.loads(message.content[0].text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erreur lors de l'analyse. Veuillez réessayer.")

    return result
