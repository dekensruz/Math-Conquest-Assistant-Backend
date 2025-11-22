"""
Math Assistant Backend - FastAPI
Gère l'upload d'images, l'extraction LaTeX via OpenAI Vision,
la résolution via WolframAlpha, et l'explication via OpenAI LLM.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import openai
import requests
import base64
from typing import Optional, List
import logging
import json
import re
from pydantic import BaseModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Initialiser FastAPI
app = FastAPI(
    title="Math Assistant API",
    description="API pour résoudre et expliquer des problèmes mathématiques",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://mathconquestassistant.vercel.app",
    ],  # Vite en local + domaine Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY non trouvée dans les variables d'environnement")

if not WOLFRAM_APP_ID:
    logger.warning("WOLFRAM_APP_ID non trouvée dans les variables d'environnement")

# Initialiser le client OpenAI (nouvelle API v1.x)
client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Modèles de données Pydantic
class SolveRequest(BaseModel):
    latex: str
    language: str = 'fr'

class ConversationMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    problem: str
    solution: dict
    question: str
    language: str = 'fr'
    history: Optional[List[ConversationMessage]] = None


@app.get("/")
async def root():
    """Endpoint de santé pour vérifier que l'API fonctionne."""
    return {"message": "Math Assistant API is running", "status": "ok"}


@app.post("/api/extract-latex")
async def extract_latex(file: UploadFile = File(...)):
    """
    Extrait le contenu mathématique d'une image et le convertit en LaTeX.
    Utilise OpenAI Vision API (GPT-4.1 nano).
    """
    try:
        # Lire le contenu de l'image
        image_content = await file.read()
        
        # Encoder en base64 pour OpenAI Vision
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        # Préparer la requête pour OpenAI Vision
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI API key non configurée")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en extraction de contenu mathématique. "
                               "Extrais uniquement le contenu mathématique de l'image et retourne-le en LaTeX propre. "
                               "Ne retourne que le LaTeX, sans explication supplémentaire. "
                               "Si l'image contient du texte explicatif, ignore-le et concentre-toi uniquement sur les équations et formules mathématiques."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extrais le contenu mathématique de cette image et retourne-le en LaTeX propre. "
                                    "Retourne uniquement le LaTeX, sans commentaire."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{file.content_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        latex_content = response.choices[0].message.content.strip()
        
        # Nettoyer le LaTeX (enlever les markdown code blocks si présents)
        if latex_content.startswith("```"):
            latex_content = latex_content.split("```")[1]
            if latex_content.startswith("latex"):
                latex_content = latex_content[5:]
            latex_content = latex_content.strip()
        
        logger.info(f"LaTeX extrait: {latex_content[:100]}...")
        
        return JSONResponse({
            "success": True,
            "latex": latex_content
        })
        
    except openai.OpenAIError as e:
        logger.error(f"Erreur OpenAI: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'extraction LaTeX avec OpenAI: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de l'image: {str(e)}"
        )


@app.post("/api/solve")
async def solve_problem(request: SolveRequest):
    """
    Résout un problème mathématique en LaTeX en utilisant WolframAlpha,
    puis génère une explication pédagogique via OpenAI LLM dans la langue demandée.
    """
    latex_problem = request.latex
    language = request.language
    if not latex_problem:
        raise HTTPException(status_code=400, detail="Le LaTeX du problème est requis")
    
    try:
        # Étape 1: Résoudre avec WolframAlpha
        wolfram_result = await call_wolframalpha(latex_problem)
        
        # Étape 2: Générer l'explication pédagogique avec OpenAI dans la langue demandée
        explanation = await generate_explanation(latex_problem, wolfram_result, language)
        
        # S'assurer que explanation est un dict et non une string
        if isinstance(explanation, str):
            logger.warning("L'explication est une string au lieu d'un dict, tentative de parsing...")
            try:
                explanation = json.loads(explanation)
            except json.JSONDecodeError:
                logger.error("Impossible de parser l'explication comme JSON")
                explanation = {
                    "type": "Erreur",
                    "method": "N/A",
                    "steps": [{
                        "step_number": 1,
                        "description": "Erreur lors du parsing de l'explication",
                        "latex": "",
                        "explanation": ""
                    }],
                    "final_answer": wolfram_result.get("result", "N/A"),
                    "summary": "Erreur lors du parsing"
                }
        
        logger.info(f"Retour de la solution avec {len(explanation.get('steps', []))} étapes")
        
        return JSONResponse({
            "success": True,
            "problem": latex_problem,
            "wolfram_result": wolfram_result,
            "explanation": explanation
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la résolution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la résolution du problème: {str(e)}"
        )


@app.post("/api/chat")
async def chat_about_solution(request: ChatRequest):
    """
    Endpoint pour poser des questions de suivi sur une solution.
    Prend en entrée : le problème original, la solution, la question de l'utilisateur et la langue.
    """
    problem = request.problem
    solution = request.solution
    question = request.question
    language = request.language
    history_turns = request.history or []
    
    if not problem or not solution or not question:
        raise HTTPException(
            status_code=400,
            detail="Les champs 'problem', 'solution' et 'question' sont requis"
        )
    
    try:
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI API key non configurée")
        
        # Préparer le contexte pour le chat
        solution_text = json.dumps(solution, ensure_ascii=False, indent=2)
        
        # Instructions en fonction de la langue
        lang_instruction = "Réponds en français" if language == 'fr' else "Answer in English"
        
        history_text = ""
        if history_turns:
            recent_turns = history_turns[-8:]
            formatted_turns = []
            for turn in recent_turns:
                role_label = "Étudiant" if turn.role == "user" else "Assistant"
                formatted_turns.append(f"{role_label}: {turn.content}")
            history_text = "\nHistorique récent de la discussion:\n" + "\n".join(formatted_turns) + "\n"
        
        prompt = f"""Tu es un professeur de mathématiques patient et pédagogue. 
Un étudiant vient de résoudre un problème mathématique et a maintenant une question de suivi.

Problème original (en LaTeX): {problem}

Solution complète:
{solution_text}

{history_text}Question de l'étudiant: {question}

Réponds à la question de l'étudiant de manière claire et pédagogique. 
- Si la question concerne une étape spécifique, explique cette étape en détail
- Si la question demande un exemple similaire, fournis-en un avec explication
- Si la question demande une clarification, explique le concept de manière simple
- Utilise du LaTeX inline avec \( ... \) pour les formules dans le texte
- Utilise du LaTeX block avec \[ ... \] pour les formules importantes sur leur propre ligne
- Structure ta réponse avec des titres (### pour les sections principales)
- Utilise des listes à puces (-) pour organiser les étapes ou points importants
- Sois encourageant et pédagogique
- Si l'étudiant remercie ou clôt la discussion, réponds très brièvement (ex: "Avec plaisir !")

IMPORTANT: {lang_instruction}

Formatte ta réponse en Markdown avec cette structure :
### Titre de la section principale

Paragraphe d'introduction...

- Point important 1
- Point important 2

\[
formule LaTeX importante
\]

Paragraphe de conclusion..."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es un professeur de mathématiques expert et patient. {lang_instruction}."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Nettoyer les symboles $ du LaTeX dans la réponse
        answer = clean_latex_string(answer)
        
        return JSONResponse({
            "success": True,
            "answer": answer
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse: {str(e)}"
        )


def clean_latex_string(text: str) -> str:
    """
    Nettoie les symboles $ d'une chaîne et supprime tout caractère de contrôle non imprimable.
    """
    if not isinstance(text, str):
        return text

    if text is None:
        return ""

    # Supprimer les caractères de contrôle sauf retours à la ligne
    text = "".join(ch for ch in text if ch == '\n' or ord(ch) >= 32)

    # Supprimer les séquences échappées problématiques (\f, \x0c, etc.)
    text = text.replace('\\f', '').replace('\f', '').replace('\x0c', '')

    # Enlever les $ et $$ qui entourent le LaTeX
    text = re.sub(r'\$\$?([^$]+)\$?\$?', r'\1', text)

    return text.strip()


async def call_wolframalpha(latex_problem: str) -> dict:
    """
    Appelle l'API WolframAlpha pour résoudre le problème mathématique.
    """
    if not WOLFRAM_APP_ID:
        # Si pas de clé WolframAlpha, retourner un résultat simulé
        logger.warning("WOLFRAM_APP_ID non configurée, utilisation d'un résultat simulé")
        return {
            "result": "Résultat calculé (WolframAlpha non configuré)",
            "steps": ["Étape 1: Analyse du problème", "Étape 2: Application de la méthode"],
            "error": "WolframAlpha API key non configurée"
        }
    
    try:
        # Convertir le LaTeX en texte simple (Wolfram gère mal certains symboles LaTeX)
        query = latex_problem.replace("\\", "").replace("{", "").replace("}", "")

        url = "https://api.wolframalpha.com/v2/query"
        params = {
            "input": query,
            "appid": WOLFRAM_APP_ID,
            "output": "json",
            "format": "plaintext",
            "podstate": "Result__Step-by-step solution"
        }

        logger.info(f"Appel WolframAlpha avec query: {query}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        result_text = None
        pods = data.get("queryresult", {}).get("pods", [])

        priority_ids = {
            "Result", "Solution", "Solutions", "ExactResult", "DecimalApproximation", "Exact result"
        }

        # Stratégie 1 : pods prioritaires
        for pod in pods:
            if pod.get("id") in priority_ids or pod.get("title") in priority_ids:
                subpods = pod.get("subpods", [])
                if subpods:
                    candidate = subpods[0].get("plaintext")
                    if candidate:
                        result_text = candidate
                        break

        # Stratégie 2 : premier pod non Input
        if not result_text:
            for pod in pods:
                if pod.get("id") in ["Input", "InputInterpretation"] or pod.get("title") in ["Input", "Input interpretation"]:
                    continue
                subpods = pod.get("subpods", [])
                if subpods:
                    candidate = subpods[0].get("plaintext")
                    if candidate:
                        result_text = candidate
                        break

        if not result_text:
            result_text = "Result unavailable"

        result_text = clean_latex_string(result_text)
        logger.info(f"Résultat WolframAlpha trouvé: {result_text}")

        return {
            "result": result_text,
            "raw_response": data
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur WolframAlpha API: {str(e)}")
        return {
            "result": "Erreur de connexion à WolframAlpha",
            "error": str(e)
        }


def fix_latex_shortcuts(text: str) -> str:
    """
    Corrige les raccourcis LaTeX mal formatés dans un texte.
    Ex: "rac{-4}{2}" -> "\frac{-4}{2}"
    """
    if not isinstance(text, str):
        return text
    
    # Corriger rac{}{} -> \frac{}{}
    text = re.sub(r'rac\{([^}]+)\}\{([^}]+)\}', r'\\frac{\1}{\2}', text)
    
    # Corriger sqrt{...} sans backslash -> \sqrt{...}
    text = re.sub(r'(?<!\\)sqrt\{', r'\\sqrt{', text)
    
    # Corriger d'autres raccourcis courants
    text = re.sub(r'(?<!\\)\bint\b', r'\\int', text)
    text = re.sub(r'(?<!\\)\bsum\b', r'\\sum', text)
    text = re.sub(r'(?<!\\)\bpi\b', r'\\pi', text)
    
    return text


def clean_latex_symbols(data: dict) -> dict:
    """
    Nettoie les symboles $ et $$ du LaTeX dans les données d'explication.
    Corrige aussi les raccourcis LaTeX mal formatés.
    """
    def clean_string(text: str) -> str:
        if not isinstance(text, str):
            return text
        # Corriger les raccourcis LaTeX
        text = fix_latex_shortcuts(text)
        # Enlever les $ et $$ qui entourent le LaTeX
        text = re.sub(r'^\$\$?(.*?)\$?\$?$', r'\1', text, flags=re.MULTILINE)
        # Enlever les $ isolés
        text = text.replace('$$', '').strip()
        return text
    
    # Nettoyer le final_answer
    if 'final_answer' in data and isinstance(data['final_answer'], str):
        data['final_answer'] = clean_string(data['final_answer'])
    
    # Nettoyer les steps
    if 'steps' in data and isinstance(data['steps'], list):
        for step in data['steps']:
            if isinstance(step, dict):
                if 'latex' in step and isinstance(step['latex'], str):
                    step['latex'] = clean_string(step['latex'])
                if 'description' in step and isinstance(step['description'], str):
                    step['description'] = clean_string(step['description'])
    
    return data


async def generate_explanation(latex_problem: str, wolfram_result: dict, language: str = 'fr') -> dict:
    """
    Génère une explication pédagogique étape par étape en utilisant OpenAI LLM.
    Support multilingue (fr/en).
    """
    try:
        # S'assurer que la réponse de WolframAlpha est bien encodée
        wolfram_answer_raw = wolfram_result.get("result", "Resultat non disponible")
        # Nettoyer l'encodage - utiliser ASCII simple pour éviter les problèmes
        if isinstance(wolfram_answer_raw, bytes):
            wolfram_answer = wolfram_answer_raw.decode('utf-8', errors='replace')
        else:
            wolfram_answer = str(wolfram_answer_raw)
        # Normaliser les caractères accentués pour éviter les problèmes d'encodage
        wolfram_answer = wolfram_answer.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
        wolfram_answer = wolfram_answer.replace('à', 'a').replace('â', 'a')
        wolfram_answer = wolfram_answer.replace('ù', 'u').replace('û', 'u')
        wolfram_answer = wolfram_answer.replace('ç', 'c')
        # Normaliser les espaces
        wolfram_answer = ' '.join(wolfram_answer.split())
        
        # Instructions de langue
        if language == 'en':
            lang_instruction = "Respond ONLY in English."
            json_format_hint = 'Ex: {"type": "equation", "method": "linear equation solving", "steps": [{"step_number": 1, "description": "Add 3 to both sides", "latex": "2x - 3 + 3 = -7 + 3", "explanation": "To isolate x"}], "final_answer": "-2", "summary": "Solution found"}'
        else:
            lang_instruction = "Réponds UNIQUEMENT en français."
            json_format_hint = 'Ex: {"type": "equation", "method": "résolution d\'équation linéaire", "steps": [{"step_number": 1, "description": "Ajouter 3 des deux côtés", "latex": "2x - 3 + 3 = -7 + 3", "explanation": "Pour isoler x"}], "final_answer": "-2", "summary": "Solution trouvée"}'

        prompt = f"""Tu es un professeur de mathématiques patient et pédagogue. 
Un étudiant a besoin d'aide pour comprendre comment résoudre ce problème mathématique.

Problème (en LaTeX): {latex_problem}

Résultat de WolframAlpha: {wolfram_answer}

Génère une explication pédagogique étape par étape qui:
1. Identifie le type de problème (dérivée, intégrale, équation, etc.)
2. Explique la méthode à utiliser
3. Montre chaque étape de calcul de manière claire
4. Utilise du LaTeX COMPLET et CORRECT pour toutes les formules mathématiques
5. Explique pourquoi chaque étape est nécessaire
6. Conclut avec la réponse finale

IMPORTANT: {lang_instruction}

IMPORTANT pour le LaTeX - RÈGLES STRICTES :
- Utilise TOUJOURS des commandes LaTeX complètes et valides
- Pour les fractions : TOUJOURS \frac{{numérateur}}{{dénominateur}} (ex: \frac{{-4}}{{2}})
- JAMAIS de raccourcis comme "rac{{-4}}{{2}}" ou "frac{{-4}}{{2}}" sans backslash
- Pour les racines : TOUJOURS \sqrt{{x}} ou \sqrt[n]{{x}}
- Pour les formules dans les descriptions de texte, utilise \( formule \) pour inline
- Pour les formules importantes, utilise le champ "latex" avec la formule complète
- Exemples CORRECTS : \( \frac{{-4}}{{2}} \), \( x^2 + 3x \), \( \sqrt{{16}} \)
- Exemples INCORRECTS à éviter : rac{{-4}}{{2}}, frac{{-4}}{{2}}, sqrt{{16}}

Formatte ta réponse en JSON avec cette structure:
{{
  "type": "type de problème",
  "method": "méthode utilisée",
  "steps": [
    {{
      "step_number": 1,
      "description": "description textuelle de l'étape. Si tu mentionnes une formule, utilise \( \\frac{{-4}}{{2}} \) pour les fractions inline.",
      "latex": "formule LaTeX complète de l'étape (optionnel, pour formules importantes)",
      "explanation": "pourquoi cette étape est importante"
    }}
  ],
  "final_answer": "réponse finale en LaTeX complet (ex: \\frac{{-4}}{{2}} ou -2)",
  "summary": "résumé de la solution"
}}

IMPORTANT - Format de réponse STRICT :
- Tu DOIS répondre UNIQUEMENT avec du JSON valide
- Le JSON doit commencer IMMÉDIATEMENT par {{ (pas de texte avant)
- Le JSON doit se terminer par }} (pas de texte après)
- PAS de blocs de code markdown (PAS de ```json ou ```)
- PAS de texte explicatif avant ou après le JSON
- Utilise UNIQUEMENT des guillemets droits " (pas de guillemets courbes " ou ")
- Utilise UNIQUEMENT des apostrophes droites ' (pas d'apostrophes courbes ' ou ')
- Le JSON DOIT être valide et parsable par json.loads() sans erreur
- Si tu ne respectes pas ce format, la réponse sera rejetée

Exemple de format CORRECT (copie exactement cette structure) :
{json_format_hint}

Rappel : Réponds UNIQUEMENT avec le JSON, rien d'autre."""

        if not client:
            raise Exception("OpenAI API key non configurée")
        
        # Utiliser response_format pour forcer le JSON (si supporté par le modèle)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Tu es un professeur de mathématiques expert. {lang_instruction} Tu réponds UNIQUEMENT en JSON valide."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
        except Exception as format_error:
            # Si response_format n'est pas supporté, essayer sans
            logger.warning(f"response_format non supporté, tentative sans: {str(format_error)}")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Tu es un professeur de mathématiques expert. {lang_instruction}"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
        
        explanation_text = response.choices[0].message.content.strip()
        
        # Logger le texte brut reçu pour débogage
        logger.info(f"Texte brut reçu d'OpenAI (200 premiers caractères): {explanation_text[:200]}")
        
        # Nettoyer le texte des caractères problématiques
        # S'assurer que le texte est en UTF-8 valide
        if isinstance(explanation_text, bytes):
            explanation_text = explanation_text.decode('utf-8', errors='replace')
        else:
            # Nettoyer les caractères mal encodés
            explanation_text = explanation_text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Extraire le JSON du bloc de code markdown si présent
        # Pattern: ```json ... ``` ou ``` ... ```
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', explanation_text, re.DOTALL)
        if json_match:
            explanation_text = json_match.group(1).strip()
            logger.info("JSON extrait d'un bloc markdown")
        else:
            # Si pas de bloc de code, chercher directement du JSON
            # Enlever tout texte avant le premier {
            first_brace = explanation_text.find('{')
            if first_brace != -1:
                explanation_text = explanation_text[first_brace:]
            # Enlever tout texte après le dernier }
            last_brace = explanation_text.rfind('}')
            if last_brace != -1:
                explanation_text = explanation_text[:last_brace + 1]
            logger.info("JSON extrait entre les accolades")
        
        # Nettoyer les caractères problématiques dans le JSON
        # Remplacer les guillemets courbes par des guillemets droits
        explanation_text = explanation_text.replace('"', '"').replace('"', '"')
        explanation_text = explanation_text.replace(''', "'").replace(''', "'")
        
        # Essayer de parser le JSON
        try:
            logger.info(f"Tentative de parsing JSON (100 premiers caractères): {explanation_text[:100]}")
            explanation_json = json.loads(explanation_text)
            # Nettoyer les symboles $ et $$ du LaTeX
            explanation_json = clean_latex_symbols(explanation_json)
            # S'assurer que c'est bien un dict (objet JSON)
            if not isinstance(explanation_json, dict):
                logger.warning(f"Le JSON parsé n'est pas un dict, type: {type(explanation_json)}")
                raise ValueError("Le JSON parsé n'est pas un objet")
            
            # Valider la structure
            if 'steps' not in explanation_json:
                logger.warning("Le JSON ne contient pas de 'steps'")
                explanation_json['steps'] = []
            
            logger.info(f"Explication parsée avec succès: {len(explanation_json.get('steps', []))} étapes")
            return explanation_json
        except (json.JSONDecodeError, ValueError) as e:
            # Si ce n'est pas du JSON valide, logger l'erreur et essayer de récupérer ce qui est possible
            logger.error(f"Erreur lors du parsing JSON: {str(e)}")
            logger.error(f"Texte reçu (1000 premiers caractères): {explanation_text[:1000]}")
            logger.error(f"Position de l'erreur: {getattr(e, 'pos', 'N/A')}")
            
            # Essayer une approche plus permissive : extraire juste les steps si possible
            try:
                # Chercher le tableau steps en comptant les accolades
                steps_start = explanation_text.find('"steps"')
                if steps_start != -1:
                    bracket_start = explanation_text.find('[', steps_start)
                    if bracket_start != -1:
                        # Compter les accolades pour trouver la fin du tableau
                        bracket_count = 0
                        bracket_end = bracket_start
                        for i in range(bracket_start, len(explanation_text)):
                            if explanation_text[i] == '[':
                                bracket_count += 1
                            elif explanation_text[i] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    bracket_end = i
                                    break
                        
                        if bracket_end > bracket_start:
                            steps_text = explanation_text[bracket_start:bracket_end + 1]
                            steps = json.loads(steps_text)
                            return {
                                "type": "équation",
                                "method": "résolution d'équation",
                                "steps": steps,
                                "final_answer": wolfram_answer,
                                "summary": "Solution générée avec récupération partielle des données"
                            }
            except Exception as recovery_error:
                logger.warning(f"Récupération partielle échouée: {str(recovery_error)}")
                pass
            
            # Si tout échoue, retourner une structure minimale avec la réponse de WolframAlpha
            # Au moins on peut afficher la réponse calculée même si l'explication échoue
            return {
                "type": "Problème mathématique",
                "method": "Résolution automatique",
                "steps": [
                    {
                        "step_number": 1,
                        "description": f"Résolution du problème : {latex_problem}",
                        "latex": latex_problem,
                        "explanation": "Le problème a été résolu par WolframAlpha, mais l'explication détaillée n'a pas pu être générée. Veuillez réessayer."
                    }
                ],
                "final_answer": wolfram_answer if wolfram_answer else "Resultat non disponible",
                "summary": f"Reponse calculee : {wolfram_answer if wolfram_answer else 'Non disponible'}. L'explication detaillee n'a pas pu etre generee."
            }
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'explication: {str(e)}")
        return {
            "type": "Erreur",
            "method": "N/A",
            "steps": [
                {
                    "step_number": 1,
                    "description": f"Erreur lors de la génération de l'explication: {str(e)}",
                    "latex": "",
                    "explanation": ""
                }
            ],
            "final_answer": wolfram_result.get("result", "N/A"),
            "summary": "Une erreur est survenue lors de la génération de l'explication."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
