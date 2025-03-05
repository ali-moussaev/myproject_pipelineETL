import pandas as pd
from sqlalchemy import create_engine, text
import requests
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
from dataclasses import dataclass, field
from time import time
from typing import Optional, Dict, List
from retrying import retry

# =============================================================================
# CONFIGURATION DU LOGGING
# =============================================================================
# Configuration du système de logging pour tracer l'exécution du script
# - Format: timestamp - nom du logger - niveau - message
# - Niveau: INFO pour capturer les informations importantes
# - Handlers: fichier (pour persistance) et console (pour debug en direct)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),  # Stockage permanent des logs
        logging.StreamHandler()  # Affichage en temps réel
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# GESTION DES VARIABLES D'ENVIRONNEMENT
# =============================================================================
# Vérification de l'existence du fichier .env contenant les clés API
env_path = Path('api.env')
if not env_path.exists():
    logger.error(f"Le fichier .env n'existe pas dans le répertoire courant. Chemin recherché : {env_path.absolute()}")
    exit()

# Chargement des variables d'environnement avec verbose=True pour le debug
load_dotenv(verbose=True)

# =============================================================================
# CONFIGURATION DU SERVICE DE TAUX DE CHANGE
# =============================================================================
# Récupération de la clé API depuis les variables d'environnement
API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')
logger.info(f"Vérification de la clé API : {'Trouvée' if API_KEY else 'Non trouvée'}")

# Gestion du taux de change avec fallback sur une valeur par défaut
if not API_KEY:
    # Si pas de clé API, utilisation d'un taux fixe de référence
    logger.warning("La clé API n'a pas été trouvée dans le fichier .env")
    logger.info("Utilisation du taux de change de référence")
    TAUX_REFERENCE = 0.94  # Taux EUR/USD de référence
    USD_TO_EUR_RATE = TAUX_REFERENCE
    logger.info(f"Taux de conversion USD → EUR de référence : {TAUX_REFERENCE}")
else:
    # Si clé API disponible, tentative de récupération du taux en temps réel
    BASE_URL = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/USD/EUR"
    TAUX_REFERENCE = 0.94  # Taux de fallback en cas d'échec

    try:
        # Appel à l'API de taux de change
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            data = response.json()
            USD_TO_EUR_RATE = data['conversion_rate']
            logger.info(f"Taux de conversion USD → EUR actuel : {USD_TO_EUR_RATE}")
        else:
            # En cas d'échec de l'API, utilisation du taux de référence
            logger.warning(f"L'API n'étant pas accessible, utilisation du taux de change de référence : {TAUX_REFERENCE}")
            USD_TO_EUR_RATE = TAUX_REFERENCE
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à l'API de taux de change: {e}")
        logger.warning(f"Utilisation du taux de change de référence : {TAUX_REFERENCE}")
        USD_TO_EUR_RATE = TAUX_REFERENCE

# =============================================================================
# CLASSES DE CONFIGURATION
# =============================================================================
@dataclass
class APIConfig:
    """
    Configuration pour l'API de taux de change
    
    Attributes:
        base_url (str): URL de base de l'API
        api_key (Optional[str]): Clé API pour l'authentification
        timeout (int): Délai maximum pour les requêtes en secondes
    """
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 5

@dataclass
class DatabaseConfig:
    """
    Configuration pour la connexion à la base de données PostgreSQL
    
    Attributes:
        host (str): Hôte de la base de données
        port (int): Port de connexion
        user (str): Nom d'utilisateur
        password (str): Mot de passe
        database (str): Nom de la base de données
    """
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def connection_string(self) -> str:
        """
        Génère la chaîne de connexion SQLAlchemy
        Returns:
            str: Chaîne de connexion formatée
        """
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# =============================================================================
# SERVICES
# =============================================================================
class ExchangeRateService:
    """
    Service de gestion des taux de change avec système de cache
    
    Attributes:
        api_config (APIConfig): Configuration de l'API
        default_rate (float): Taux de change par défaut
        _last_update (float): Timestamp de la dernière mise à jour
        _cache (Optional[float]): Valeur en cache du taux
    """
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.default_rate = 0.94  # Taux de change par défaut
        self._last_update = 0  # Timestamp de la dernière mise à jour
        self._cache = None  # Cache du taux de change

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
    def get_exchange_rate(self) -> float:
        """
        Récupère le taux de change avec gestion de cache
        - Met en cache le résultat pendant 1 heure
        - Réessaie 3 fois en cas d'échec avec délai exponentiel
        
        Returns:
            float: Taux de change USD vers EUR
        """
        current_time = time()
        # Vérification de la validité du cache (1 heure)
        if self._cache is not None and current_time - self._last_update < 3600:
            return self._cache

        # Si pas de clé API, retourne le taux par défaut
        if not self.api_config.api_key:
            logger.warning("Utilisation du taux de change par défaut")
            return self.default_rate

        try:
            # Tentative de récupération du nouveau taux
            response = requests.get(
                f"{self.api_config.base_url}/v6/{self.api_config.api_key}/pair/USD/EUR",
                timeout=self.api_config.timeout
            )
            if response.status_code == 200:
                rate = response.json()['conversion_rate']
                self._cache = rate
                self._last_update = current_time
                return rate
        except Exception as e:
            logger.error(f"Erreur API: {e}")
        
        return self.default_rate

# =============================================================================
# CLASSES DE TRAITEMENT DES DONNÉES
# =============================================================================
class DataValidator:
    """
    Classe de validation des données du DataFrame
    Vérifie la structure et la qualité des données
    """
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """
        Valide la structure et les données du DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame à valider
            
        Returns:
            bool: True si valide, False sinon
            
        Checks:
            - Présence des colonnes requises
            - Absence de valeurs nulles
            - Salaires positifs
        """
        required_columns = ['job_title', 'salary_in_usd', 'experience_level']
        
        try:
            # Vérification des colonnes requises
            assert all(col in df.columns for col in required_columns)
            # Vérification des valeurs nulles
            assert not df[required_columns].isnull().any().any()
            # Vérification des salaires positifs
            assert df['salary_in_usd'].gt(0).all()
            return True
        except AssertionError:
            logger.error("Validation du DataFrame échouée")
            return False

class DataCleaner:
    """
    Classe de nettoyage et normalisation des données
    Applique des transformations pour uniformiser les données
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Nettoie et normalise une chaîne de texte
        
        Args:
            text (str): Texte à nettoyer
            
        Returns:
            str: Texte nettoyé et normalisé
            
        Transformations:
            - Conversion en minuscules
            - Suppression des espaces en début/fin
            - Gestion des valeurs nulles
        """
        if pd.isna(text):
            return text
        return str(text).strip().lower()

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie l'ensemble du DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame nettoyé
            
        Transformations:
            - Copie du DataFrame pour éviter les modifications en place
            - Nettoyage des colonnes textuelles
        """
        df = df.copy()
        
        # Liste des colonnes textuelles à nettoyer
        text_columns = ['job_title', 'job_category', 'experience_level', 'work_setting']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
                
        return df

# =============================================================================
# PIPELINE ETL PRINCIPAL
# =============================================================================

# === 1. EXTRACTION ===
csv_file_path = "C:/Users/Shonan/Desktop/MyDataEngineerProject/myproject_pipelineETL/jobs_in_data.csv"
logger.info("Extraction : Chargement du fichier CSV...")
try:
    # Lecture optimisée du CSV avec types spécifiés
    df = pd.read_csv(
        csv_file_path,
        usecols=['job_title', 'salary_in_usd', 'experience_level', 'work_setting'],
        dtype={
            'job_title': 'category',  # Optimisation mémoire pour les chaînes répétées
            'experience_level': 'category',
            'work_setting': 'category'
        }
    )
    # Logs des informations sur les données chargées
    logger.info(f"✓ Fichier CSV chargé avec succès!")
    logger.info(f"→ Nombre de lignes : {len(df)}")
    logger.info(f"→ Colonnes présentes : {', '.join(df.columns)}")
    
    # Analyse détaillée de la structure des données
    logger.info("\n=== Analyse de la structure des données ===")
    logger.info("\n→ Types de données pour chaque colonne :")
    logger.info(df.dtypes)
    
    logger.info("\n→ Statistiques descriptives des colonnes numériques :")
    logger.info(df.describe())
    
    logger.info("\n→ Vérification des valeurs manquantes :")
    missing_values = df.isnull().sum()
    logger.info(missing_values[missing_values > 0])
    
    logger.info("\n→ Exemple des premières lignes :")
    logger.info(df.head(3))
    
    # Vérification des valeurs uniques dans les colonnes catégorielles
    categorical_columns = df.select_dtypes(include=['object']).columns
    logger.info("\n→ Valeurs uniques dans les colonnes catégorielles :")
    for col in categorical_columns:
        n_unique = df[col].nunique()
        logger.info(f"{col}: {n_unique} valeurs uniques")
        if n_unique < 10:  # Afficher les valeurs uniques seulement si leur nombre est raisonnable
            logger.info(f"Valeurs : {sorted(df[col].unique())}")

except FileNotFoundError:
    logger.error(f"❌ Erreur : Le fichier {csv_file_path} n'a pas été trouvé.")
    exit()
except Exception as e:
    logger.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
    exit()

# === 2. Nettoyage des données ===
logger.info("\n=== Nettoyage et uniformisation des données ===")

# Uniformiser les noms de colonnes en minuscules
df.columns = df.columns.str.lower()
logger.info("✓ Noms des colonnes uniformisés en minuscules")

# Nettoyage des colonnes textuelles
text_columns = ['job_title', 'job_category', 'experience_level', 'employment_type', 'company_location', 'company_size']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(DataCleaner.clean_text)
        logger.info(f"✓ Colonne {col} : texte uniformisé en minuscules et nettoyé")

# Conversion et nettoyage des colonnes numériques
numeric_columns = ['salary', 'salary_in_usd']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Suppression des valeurs aberrantes (optionnel)
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        iqr = q3 - q1
        df = df[df[col].between(q1 - 1.5*iqr, q3 + 1.5*iqr)]
        logger.info(f"✓ Colonne {col} : convertie en numérique et nettoyée des valeurs aberrantes")

# Conversion en EUR avec le taux actuel
if 'salary_in_usd' in df.columns:
    df['salary_in_eur'] = df['salary_in_usd'] * USD_TO_EUR_RATE
    logger.info("✓ Conversion des salaires en EUR effectuée")

# Gestion des dates si présentes
date_columns = ['work_year']  # Ajoutez d'autres colonnes de dates si nécessaire
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        logger.info(f"✓ Colonne {col} : convertie en année")

# Vérification des valeurs manquantes après nettoyage
missing_values = df.isnull().sum()
if missing_values.any():
    logger.info("\n→ Valeurs manquantes après nettoyage :")
    logger.info(missing_values[missing_values > 0])
    # Suppression des lignes avec des valeurs manquantes dans les colonnes critiques
    critical_columns = ['job_title', 'salary_in_usd', 'salary_in_eur']
    df = df.dropna(subset=critical_columns)
    logger.info("✓ Suppression des lignes avec des valeurs manquantes dans les colonnes critiques")

# Affichage des statistiques après nettoyage
logger.info("\n=== Statistiques après nettoyage ===")
logger.info(f"Nombre de lignes final : {len(df)}")
logger.info("\nAperçu des données nettoyées :")
logger.info(df.head(3))

# === 3. Chargement dans PostgreSQL via SQLAlchemy ===
logger.info("Chargement des données dans PostgreSQL...")
# Remplace 'user' et 'password' par tes identifiants, et 'mydatas' par le nom de ta base
engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
table_name = "data_scientists_salaries"  # Tu peux choisir un autre nom de table si nécessaire

try:
    # if_exists='replace' recrée la table à chaque exécution (ou utilise 'append' pour ajouter)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    logger.info(f"Données chargées avec succès dans la table '{table_name}'.")
except Exception as e:
    logger.error("Erreur lors du chargement des données :", e)

# === 4. Analyse des salaires avec des requêtes SQL ===
logger.info("\n=== Analyses des salaires ===")

# Vérification des colonnes disponibles
logger.info("\nColonnes disponibles dans la table :")
query_columns = f"""
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = '{table_name}';
"""
with engine.connect() as conn:
    columns = conn.execute(text(query_columns))
    available_columns = [col[0] for col in columns]
    logger.info("Colonnes disponibles:", available_columns)

# 1. Analyse par intitulé de poste
logger.info("\n1. Salaires moyens par intitulé de poste :")
query_by_title = f"""
SELECT 
    job_title,
    CAST(AVG(salary_in_eur) AS DECIMAL(10,2)) AS avg_salary_eur,
    COUNT(*) AS total
FROM 
    {table_name}
GROUP BY 
    job_title
ORDER BY 
    avg_salary_eur DESC
LIMIT 10;
"""

# 2. Analyse détaillée par niveau d'expérience
logger.info("\n2. Analyse des salaires par niveau d'expérience :")
query_by_experience = f"""
SELECT 
    experience_level,
    CAST(AVG(salary_in_eur) AS DECIMAL(10,2)) AS salaire_moyen,
    CAST(MIN(salary_in_eur) AS DECIMAL(10,2)) AS salaire_min,
    CAST(MAX(salary_in_eur) AS DECIMAL(10,2)) AS salaire_max,
    CAST(STDDEV(salary_in_eur) AS DECIMAL(10,2)) AS ecart_type,
    COUNT(*) AS nombre_postes
FROM 
    {table_name}
GROUP BY 
    experience_level
ORDER BY 
    CASE 
        WHEN experience_level = 'entry' THEN 1
        WHEN experience_level = 'mid' THEN 2
        WHEN experience_level = 'senior' THEN 3
        WHEN experience_level = 'executive' THEN 4
        ELSE 5
    END;
"""

# 3. Analyse des salaires selon le mode de travail (remote/présentiel)
logger.info("\n3. Analyse des salaires selon le mode de travail :")
query_by_remote = f"""
WITH stats_by_remote AS (
    SELECT 
        CASE 
            WHEN work_setting = 'Remote' THEN 'Télétravail'
            WHEN work_setting = 'In-person' THEN 'Présentiel'
            WHEN work_setting = 'Hybrid' THEN 'Hybride'
        END AS mode_travail,
        CAST(AVG(salary_in_eur) AS DECIMAL(10,2)) AS salaire_moyen,
        CAST(MIN(salary_in_eur) AS DECIMAL(10,2)) AS salaire_min,
        CAST(MAX(salary_in_eur) AS DECIMAL(10,2)) AS salaire_max,
        CAST(STDDEV(salary_in_eur) AS DECIMAL(10,2)) AS ecart_type,
        COUNT(*) AS nombre_postes,
        CAST(AVG(CASE WHEN experience_level = 'senior' THEN salary_in_eur END) AS DECIMAL(10,2)) AS moy_senior,
        CAST(AVG(CASE WHEN experience_level = 'mid' THEN salary_in_eur END) AS DECIMAL(10,2)) AS moy_mid,
        CAST(AVG(CASE WHEN experience_level = 'entry' THEN salary_in_eur END) AS DECIMAL(10,2)) AS moy_entry
    FROM 
        {table_name}
    GROUP BY 
        work_setting
)
SELECT 
    *,
    CAST(100.0 * nombre_postes / SUM(nombre_postes) OVER () AS DECIMAL(5,2)) as pourcentage_postes
FROM 
    stats_by_remote
ORDER BY 
    salaire_moyen DESC;
"""

# Exécution des requêtes
with engine.connect() as conn:
    # Analyse par poste
    logger.info("\n=== Top 10 des postes les mieux rémunérés ===")
    result = conn.execute(text(query_by_title))
    for row in result:
        logger.info(f"📊 {row[0]}: {format(int(row[1]), ' ')}€ (Nombre de postes: {row[2]})")
    
    # Analyse par niveau d'expérience
    logger.info("\n=== Analyse détaillée par niveau d'expérience ===")
    result = conn.execute(text(query_by_experience))
    for row in result:
        logger.info(f"\n📊 Niveau : {row[0].upper()}")
        logger.info(f"   → Salaire moyen : {format(int(row[1]), ' ')}€")
        logger.info(f"   → Fourchette : {format(int(row[2]), ' ')}€ - {format(int(row[3]), ' ')}€")
        logger.info(f"   → Écart-type : {format(int(row[4]), ' ')}€")
        logger.info(f"   → Nombre de postes : {row[5]}")

    # Analyse par mode de travail
    logger.info("\n=== Analyse détaillée par mode de travail ===")
    result = conn.execute(text(query_by_remote))
    for row in result:
        logger.info(f"\n📊 Mode : {row[0]}")
        logger.info(f"   → Salaire moyen global : {format(int(row[1]), ' ')}€")
        logger.info(f"   → Fourchette : {format(int(row[2]), ' ')}€ - {format(int(row[3]), ' ')}€")
        logger.info(f"   → Écart-type : {format(int(row[4]), ' ')}€")
        logger.info(f"   → Nombre de postes : {row[5]} ({row[9]}% du total)")
        logger.info(f"   → Moyenne par niveau :")
        if row[6]: logger.info(f"      • Senior : {format(int(row[6]), ' ')}€")
        if row[7]: logger.info(f"      • Mid : {format(int(row[7]), ' ')}€")
        if row[8]: logger.info(f"      • Entry : {format(int(row[8]), ' ')}€")

logger.info("\nAnalyses terminées.")

logger.info("Pipeline ETL terminé.")

# =============================================================================
# CONFIGURATION INTERNATIONALE
# =============================================================================
# Dictionnaire de traduction pour l'internationalisation des modes de travail
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    'fr': {
        'remote': 'Télétravail',
        'in_person': 'Présentiel',
        'hybrid': 'Hybride'
    }
}

