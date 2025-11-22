#!/bin/bash

echo "========================================"
echo "  Installation des dépendances Python"
echo "========================================"
echo ""

# Vérifier que l'environnement virtuel est activé
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[INFO] Activation de l'environnement virtuel..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "[ERREUR] Environnement virtuel non trouvé. Créez-le d'abord avec: python3 -m venv venv"
        exit 1
    fi
fi

echo "[1/7] Mise à jour de pip..."
python -m pip install --upgrade pip

echo ""
echo "[2/7] Installation de fastapi..."
pip install fastapi

echo ""
echo "[3/7] Installation de uvicorn..."
pip install uvicorn

echo ""
echo "[4/7] Installation de python-multipart..."
pip install python-multipart

echo ""
echo "[5/7] Installation de python-dotenv..."
pip install python-dotenv

echo ""
echo "[6/7] Installation de openai..."
pip install openai

echo ""
echo "[7/7] Installation de requests et pydantic..."
pip install requests pydantic

echo ""
echo "========================================"
echo "  Vérification de l'installation..."
echo "========================================"

python -c "import fastapi; print('✅ FastAPI installé !')" || {
    echo "❌ Erreur: FastAPI n'est pas installé correctement"
    exit 1
}

python -c "import uvicorn; print('✅ Uvicorn installé !')" || {
    echo "❌ Erreur: Uvicorn n'est pas installé correctement"
    exit 1
}

echo ""
echo "========================================"
echo "  ✅ Installation terminée avec succès !"
echo "========================================"
echo ""
echo "Prochaines étapes:"
echo "1. Créer le fichier .env (voir ENV_SETUP.txt)"
echo "2. Lancer le backend: python main.py"
echo ""

