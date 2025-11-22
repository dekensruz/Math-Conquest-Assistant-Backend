@echo off
echo ========================================
echo   Installation des dependances Python
echo ========================================
echo.

REM Vérifier que l'environnement virtuel est activé
if not defined VIRTUAL_ENV (
    echo [INFO] Activation de l'environnement virtuel...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else (
        echo [ERREUR] Environnement virtuel non trouve. Creer-le d'abord avec: python -m venv venv
        pause
        exit /b 1
    )
)

echo [1/7] Mise a jour de pip...
python -m pip install --upgrade pip

echo.
echo [2/7] Installation de fastapi...
pip install fastapi

echo.
echo [3/7] Installation de uvicorn...
pip install uvicorn

echo.
echo [4/7] Installation de python-multipart...
pip install python-multipart

echo.
echo [5/7] Installation de python-dotenv...
pip install python-dotenv

echo.
echo [6/7] Installation de openai...
pip install openai

echo.
echo [7/7] Installation de requests et pydantic...
pip install requests pydantic

echo.
echo ========================================
echo   Verification de l'installation...
echo ========================================
python -c "import fastapi; print('✅ FastAPI installe !')" 2>nul
if errorlevel 1 (
    echo ❌ Erreur: FastAPI n'est pas installe correctement
    pause
    exit /b 1
)

python -c "import uvicorn; print('✅ Uvicorn installe !')" 2>nul
if errorlevel 1 (
    echo ❌ Erreur: Uvicorn n'est pas installe correctement
    pause
    exit /b 1
)

echo.
echo ========================================
echo   ✅ Installation terminee avec succes !
echo ========================================
echo.
echo Prochaines etapes:
echo 1. Creer le fichier .env (voir ENV_SETUP.txt)
echo 2. Lancer le backend: python main.py
echo.
pause

