# Utiliser une image Python officielle
FROM python:3.9

RUN pip install --upgrade pip
RUN mkdir -p /app/src
RUN mkdir -p /app/app
WORKDIR /app/app

# Copier les fichiers de votre application dans le conteneur
COPY . /app/app

# Installer les dépendances nécessaires
RUN pip install -r /app/app/requirements.txt
RUN pip install -r /app/app/requirements_new.txt

# Ajouter le répertoire "src" au PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# Exécuter les tests unitaires lors du démarrage du conteneur
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]