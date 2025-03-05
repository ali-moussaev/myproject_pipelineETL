Pipeline ETL - Analyse des Salaires Data Science 📊
Description
Ce projet implémente un pipeline ETL (Extract, Transform, Load) pour l'analyse des salaires dans le domaine de la Data Science. Le pipeline collecte, nettoie, transforme et analyse les données salariales, en effectuant une conversion automatique des salaires USD/EUR, puis génère des rapports statistiques détaillés sur les salaires des professionnels de la Data Science.

Objectifs :
Extraire les données des fichiers CSV.
Transformer les données en nettoyant, normalisant et en convertissant les salaires dans la devise désirée.
Charger les données dans une base de données PostgreSQL.
Analyser les données en fonction des différents critères : salaires par poste, niveau d'expérience, mode de travail, etc.
🌟 Fonctionnalités
Extraction des Données
Lecture optimisée de fichiers CSV avec validation automatique des données sources.
Gestion intelligente des types de données pour éviter les erreurs lors de l'importation.
Transformation des Données
Nettoyage et normalisation des données pour garantir leur qualité.
Conversion automatique des salaires USD → EUR via l'API ExchangeRate-API.
Catégorisation des postes et niveaux d'expérience pour mieux comprendre la répartition des salaires.
Gestion des valeurs manquantes et aberrantes pour assurer la cohérence des données.
Chargement des Données
Stockage des données dans une base de données PostgreSQL avec une gestion optimisée des connexions.
Validation des données avant insertion pour éviter d'introduire des erreurs dans la base de données.
Analyses Statistiques
Statistiques par poste : analyse des salaires moyens par poste.
Analyses par niveau d'expérience : étude de l'impact de l'expérience sur les salaires.
Étude des modes de travail (télétravail/présentiel/hybride) et leur influence sur les salaires.
Génération de rapports détaillés avec logging pour assurer la traçabilité des opérations.
🛠 Technologies Utilisées
Python 3.8+ : Langage de programmation principal.
Pandas : Manipulation et analyse des données.
SQLAlchemy : ORM pour faciliter la gestion des interactions avec la base de données.
PostgreSQL : Base de données relationnelle pour le stockage des données.
Requests : Utilisé pour effectuer des appels API pour la conversion de devise.
python-dotenv : Gestion des variables d'environnement sensibles.
logging : Gestion et traçabilité des logs pour le suivi des actions.
📋 Prérequis
Avant de commencer, assurez-vous que vous avez les éléments suivants :

Python 3.8 ou version supérieure.
Une base de données PostgreSQL installée et en cours d'exécution.
Une clé API pour ExchangeRate-API (si vous souhaitez effectuer des conversions de devises).
🚀 Installation
1. Cloner le repository
bash
Copier
Modifier
git clone https://github.com/votre-username/myproject_pipelineETL.git
cd myproject_pipelineETL
2. Créer un environnement virtuel
bash
Copier
Modifier
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# ou
.venv\Scripts\activate     # Windows
3. Installer les dépendances
bash
Copier
Modifier
pip install -r requirements.txt
4. Configurer les variables d'environnement
Copiez le fichier .env.example vers .env :
bash
Copier
Modifier
cp .env.example .env
Remplissez les variables d'environnement dans le fichier .env :
env
Copier
Modifier
EXCHANGE_RATE_API_KEY=votre_clé_api
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe
DB_NAME=postgres
📝 Structure du Projet
Voici la structure du projet pour mieux comprendre son organisation :

bash
Copier
Modifier
myproject_pipelineETL/
│
├── data/               # Dossier pour les fichiers CSV
├── logs/               # Dossier contenant les logs d'exécution
├── src/                # Dossier contenant le code source
│   ├── main.py         # Fichier principal pour exécuter le pipeline ETL
│   └── utils.py        # Fichier utilitaire pour les fonctions secondaires
├── tests/              # Dossier contenant les tests unitaires
├── .env                # Fichier des variables d'environnement
├── requirements.txt    # Liste des dépendances Python
└── README.md           # Ce fichier
🔧 Configuration de la Base de Données
Créer la base de données et la table
Avant de charger les données, vous devez configurer la base de données. Exécutez les commandes suivantes dans votre client PostgreSQL :

sql
Copier
Modifier
CREATE DATABASE data_science_salaries;
CREATE TABLE data_scientists_salaries (
    id SERIAL PRIMARY KEY,
    job_title VARCHAR(255),
    salary_in_usd NUMERIC,
    salary_in_eur NUMERIC,
    experience_level VARCHAR(50),
    work_setting VARCHAR(50)
);
Variables d'Environnement
EXCHANGE_RATE_API_KEY : Clé API pour accéder aux taux de change.
DB_HOST : Hôte de la base de données (par exemple localhost).
DB_PORT : Port de la base de données (par défaut 5432).
DB_USER : Utilisateur de la base de données.
DB_PASSWORD : Mot de passe de l'utilisateur.
DB_NAME : Nom de la base de données.
📊 Utilisation
1. Préparation des données
Placez votre fichier CSV dans le dossier data/.
Assurez-vous que le fichier CSV contient les colonnes nécessaires :
job_title : Intitulé du poste.
salary_in_usd : Salaire en USD.
experience_level : Niveau d'expérience (par exemple, Junior, Senior).
work_setting : Mode de travail (télétravail, hybride, présentiel).
2. Exécution du pipeline ETL
Lancez le pipeline avec la commande suivante :

bash
Copier
Modifier
python src/main.py
3. Consultation des logs
Les logs sont disponibles dans le fichier logs/app.log. Les erreurs sont tracées avec les niveaux appropriés (INFO, WARNING, ERROR). Vous pouvez utiliser ces logs pour suivre l'exécution du pipeline.

📈 Exemples d'Analyses
Voici quelques exemples de requêtes SQL pour analyser les données chargées dans PostgreSQL :

Salaires par Niveau d'Expérience
sql
Copier
Modifier
SELECT 
    experience_level,
    ROUND(AVG(salary_in_eur)) AS avg_salary,
    COUNT(*) AS count
FROM data_scientists_salaries
GROUP BY experience_level
ORDER BY avg_salary DESC;
Distribution par Mode de Travail
sql
Copier
Modifier
SELECT 
    work_setting,
    COUNT(*) AS count,
    ROUND(AVG(salary_in_eur)) AS avg_salary
FROM data_scientists_salaries
GROUP BY work_setting;
🔍 Tests
Des tests unitaires sont inclus pour valider les différentes étapes du pipeline. Pour les exécuter, utilisez la commande suivante :

bash
Copier
Modifier
pytest tests/
📝 Logging
Le logging est configuré pour assurer une traçabilité des opérations. Le format des logs est le suivant :

pgsql
Copier
Modifier
timestamp - level - message
Les niveaux de log disponibles sont :

INFO : Pour les informations générales.
WARNING : Pour les avertissements.
ERROR : Pour les erreurs critiques.
🛡 Sécurité
Les variables sensibles sont stockées dans le fichier .env, qui ne doit pas être partagé.
Toutes les entrées utilisateur et les données importées sont validées avant traitement.
Les connexions à la base de données sont sécurisées avec les bonnes pratiques de gestion des identifiants.
🤝 Contribution
Vous souhaitez contribuer au projet ? Voici comment faire :

Forkez le projet sur GitHub.
Créez une branche pour votre fonctionnalité (git checkout -b feature/AmazingFeature).
Commitez vos changements (git commit -m 'Add AmazingFeature').
Poussez votre branche (git push origin feature/AmazingFeature).
Ouvrez une Pull Request pour discuter des modifications.
📄 Licence
Ce projet est sous licence MIT.

👥 Auteurs
Votre Nom - Travail initial - VotreGitHub
🙏 Remerciements
ExchangeRate-API : Pour la fourniture des taux de change.
La communauté Python : Pour les packages utilisés dans ce projet.