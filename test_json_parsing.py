"""
Script de test pour vérifier le parsing JSON
"""
import json
import re

def test_json_parsing():
    # Simuler différentes réponses possibles d'OpenAI
    test_cases = [
        # Cas 1: JSON valide simple
        '{"type": "equation", "method": "resolution", "steps": [], "final_answer": "-2", "summary": "Solution"}',
        
        # Cas 2: JSON dans un bloc markdown
        '```json\n{"type": "equation", "steps": []}\n```',
        
        # Cas 3: JSON avec texte avant/après
        'Voici la réponse:\n{"type": "equation", "steps": []}\nC\'est la fin.',
        
        # Cas 4: JSON avec guillemets courbes
        '{"type": "équation", "method": "résolution"}',
        
        # Cas 5: JSON malformé
        '{"type": "equation", "steps": [{"step": 1}]',  # Manque une accolade
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test {i} ===")
        print(f"Input: {test_case[:50]}...")
        
        explanation_text = test_case.strip()
        
        # Nettoyer l'encodage
        if isinstance(explanation_text, bytes):
            explanation_text = explanation_text.decode('utf-8', errors='replace')
        else:
            explanation_text = explanation_text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Extraire le JSON
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', explanation_text, re.DOTALL)
        if json_match:
            explanation_text = json_match.group(1).strip()
            print("Extrait d'un bloc markdown")
        else:
            first_brace = explanation_text.find('{')
            if first_brace != -1:
                explanation_text = explanation_text[first_brace:]
            last_brace = explanation_text.rfind('}')
            if last_brace != -1:
                explanation_text = explanation_text[:last_brace + 1]
            print("Extrait entre les accolades")
        
        # Nettoyer les guillemets
        explanation_text = explanation_text.replace('"', '"').replace('"', '"')
        explanation_text = explanation_text.replace(''', "'").replace(''', "'")
        
        # Parser
        try:
            result = json.loads(explanation_text)
            print(f"✅ Succès: {result.get('type', 'N/A')}")
        except json.JSONDecodeError as e:
            print(f"❌ Erreur: {str(e)}")
            print(f"Texte nettoyé: {explanation_text[:100]}")

if __name__ == "__main__":
    test_json_parsing()

