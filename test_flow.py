"""
Script de test pour valider le flux complet de l'application Math Assistant.
Teste : Upload -> Extraction LaTeX -> Résolution -> Explication
"""

import requests
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

BASE_URL = "http://localhost:8000"

def test_api_health():
    """Teste que l'API est accessible."""
    print("🔍 Test 1: Vérification de l'API...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API accessible")
            print(f"   Réponse: {response.json()}")
            return True
        else:
            print(f"❌ API retourne le code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion: {str(e)}")
        print("   Assurez-vous que le backend est lancé (python main.py)")
        return False

def test_extract_latex(image_path):
    """Teste l'extraction LaTeX depuis une image."""
    print("\n🔍 Test 2: Extraction LaTeX depuis une image...")
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image non trouvée: {image_path}")
        print("   Créez une image de test ou utilisez une image existante")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/api/extract-latex",
                files=files
            )
        
        if response.status_code == 200:
            data = response.json()
            latex = data.get('latex', '')
            print("✅ Extraction LaTeX réussie")
            print(f"   LaTeX extrait: {latex[:100]}...")
            return latex
        else:
            print(f"❌ Erreur lors de l'extraction: {response.status_code}")
            print(f"   Détail: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None

def test_solve(latex_problem):
    """Teste la résolution d'un problème mathématique."""
    print("\n🔍 Test 3: Résolution du problème...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/solve",
            json={"latex": latex_problem}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Résolution réussie")
            print(f"   Problème: {data.get('problem', '')[:50]}...")
            print(f"   Résultat Wolfram: {str(data.get('wolfram_result', {}).get('result', 'N/A'))[:100]}...")
            
            explanation = data.get('explanation', {})
            if explanation:
                print(f"   Type: {explanation.get('type', 'N/A')}")
                print(f"   Méthode: {explanation.get('method', 'N/A')}")
                print(f"   Nombre d'étapes: {len(explanation.get('steps', []))}")
            
            return data
        else:
            print(f"❌ Erreur lors de la résolution: {response.status_code}")
            print(f"   Détail: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None

def main():
    """Fonction principale de test."""
    print("=" * 60)
    print("🧪 Test du flux complet Math Assistant")
    print("=" * 60)
    
    # Test 1: Vérification de l'API
    if not test_api_health():
        print("\n❌ L'API n'est pas accessible. Arrêt des tests.")
        return
    
    # Test 2: Extraction LaTeX (optionnel - nécessite une image)
    print("\n" + "=" * 60)
    print("💡 Pour tester l'extraction LaTeX, placez une image de test")
    print("   dans le dossier backend/ et modifiez le chemin ci-dessous")
    print("=" * 60)
    
    # Exemple de test avec un LaTeX direct (sans image)
    test_latex = "f(x) = x^2 + 3x + 2"
    print(f"\n📝 Test avec LaTeX direct: {test_latex}")
    
    # Test 3: Résolution
    solution = test_solve(test_latex)
    
    if solution:
        print("\n" + "=" * 60)
        print("✅ Tous les tests sont passés avec succès !")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  Certains tests ont échoué")
        print("=" * 60)

if __name__ == "__main__":
    main()

