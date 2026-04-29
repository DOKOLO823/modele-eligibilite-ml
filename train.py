import pandas as pd
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

# Mots vides génériques (structure du texte, pas le contenu)
# On retire UNIQUEMENT les mots fonctionnels sans valeur sémantique
# On ne retire PAS les mots thématiques — le modèle doit les apprendre lui-même
STOP_WORDS = [
    # Français fonctionnel
    'le', 'la', 'les', 'de', 'des', 'un', 'une', 'du', 'au', 'aux',
    'et', 'en', 'ce', 'ces', 'se', 'sa', 'son', 'ses',
    'nous', 'vous', 'ils', 'elles', 'je', 'tu', 'il', 'elle',
    'qui', 'que', 'par', 'sur', 'sous', 'dans', 'avec', 'pour',
    'pas', 'plus', 'ou', 'mais', 'donc', 'ni', 'y', 'on',
    'ne', 'si', 'tout', 'tous', 'cette', 'cet', 'dont',
    # Anglais fonctionnel
    'the', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
    'in', 'and', 'for', 'with', 'that', 'this', 'it', 'as', 'at', 'be',
    'its', 'by', 'from', 'has', 'have', 'will', 'can', 'which',
    # Étiquettes structurelles du dataset (parasites sans valeur prédictive)
    'titre', 'theme', 'solution', 'description', 'candidat',
    'nom', 'desc', 'regles', 'regle', 'proposee', 'particularite',
    'resoudre', 'defi', 'probleme',
]

def nettoyer_texte(texte):
    """Nettoyage universel : accents, casse, ponctuation."""
    if not isinstance(texte, str):
        return ""
    # Suppression des accents
    texte = "".join(
        c for c in unicodedata.normalize('NFD', texte)
        if unicodedata.category(c) != 'Mn'
    )
    texte = texte.lower()
    # Suppression ponctuation, garde les espaces et alphanumériques
    texte = re.sub(r'[^\w\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def preparer_texte(texte):
    """Pipeline de préparation — uniquement nettoyage, pas d'injection de règles."""
    return nettoyer_texte(texte)

def entrainer():
    df = pd.read_csv('donnees_entrainement.csv')
    print(f"Dataset charge : {len(df)} exemples")
    print(f"  Positifs (eligibles)    : {df['label'].sum()}")
    print(f"  Negatifs (non eligibles): {(df['label'] == 0).sum()}")

    # Chaque ligne du dataset contient DEJA challenge + projet concatenes
    # Le modele apprend directement la relation entre les deux
    df['text_prepare'] = df['text'].apply(preparer_texte)

    # TF-IDF generaliste
    # - ngram_range=(1,3) : capture les expressions comme "maladie des plantes", "don de sang"
    # - sublinear_tf=True : evite qu'un mot tres frequent ecrase les autres
    # - min_df=1 : garde tous les tokens, meme rares (important pour la generalisation)
    # - max_df=0.90 : retire les tokens presents dans +90% des docs (trop generiques)
    # - max_features=10000 : espace vectoriel large pour capturer la diversite
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words=STOP_WORDS,
        max_df=0.90,
        min_df=1,
        max_features=10000,
        sublinear_tf=True,
        analyzer='word',
    )

    X = vectorizer.fit_transform(df['text_prepare'])
    y = df['label']

    # Régression logistique
    # - C=3.0 : regularisation moderee, evite le surapprentissage sur les domaines connus
    # - class_weight='balanced' : compense si positifs/negatifs sont desequilibres
    # - max_iter=2000 : assure la convergence
    model = LogisticRegression(
        C=3.0,
        class_weight='balanced',
        max_iter=2000,
        solver='lbfgs',
    )
    model.fit(X, y)

    # Évaluation par cross-validation stratifiée (respecte la proportion +/-)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    scores_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nEvaluation cross-validation (5 folds) :")
    print(f"  F1       : {scores_f1.mean():.3f} (+/- {scores_f1.std():.3f})")
    print(f"  Accuracy : {scores_acc.mean():.3f} (+/- {scores_acc.std():.3f})")

    # Sauvegarde
    joblib.dump(model, 'modele_eligibilite.pkl')
    joblib.dump(vectorizer, 'vectoriseur.pkl')
    print("\nModele et vectoriseur sauvegardes.")

    # Tests de validation sur des cas representatifs
    # Ces tests couvrent des situations variees, pas seulement les domaines du dataset
    test_cas = [
        # (projet, challenge, label_attendu, description)
        (
            "Planto application mobile pour detecter la maladie des plantes via photo smartphone",
            "Titre: Agritech Innovation Challenge. Theme: Transformation durable du secteur agropastoral",
            1, "Detection maladie plantes → agropastoral"
        ),
        (
            "plateforme de don de sang mise en relation donneurs et hopitaux en temps reel",
            "Titre: Hackathon sante. Theme: Le numerique au service de la sante",
            1, "Don de sang → sante"
        ),
        (
            "application pour acceder aux contenus educatifs et cours en ligne pour etudiants",
            "Titre: Hackathon sante. Theme: Le numerique au service de la sante",
            0, "Education → hors sujet sante"
        ),
        (
            "vente de vetements mode et bijoux africains livraison rapide",
            "Titre: Agritech Innovation Challenge. Theme: Transformation durable du secteur agropastoral",
            0, "Mode → hors sujet agropastoral"
        ),
        (
            "systeme de covoiturage electrique pour reduire emissions CO2 transport urbain",
            "Titre: Hackathon Orange. Theme: Climate Change Challenge solutions numeriques rechauffement",
            1, "Covoiturage electrique → climat"
        ),
        (
            "application paris sportifs football gains rapides",
            "Titre: DevsFest. Theme: Intelligence artificielle au service de l impact local",
            0, "Paris sportifs → hors sujet IA locale"
        ),
        (
            "chatbot traduction automatique francais fulfulde arabe pour administration rurale",
            "Titre: Hackathon DevsFest. Theme: Intelligence artificielle au service de l impact local",
            1, "Chatbot traduction → IA impact local"
        ),
        (
            "logiciel gestion garage pieces detachees voitures motos",
            "Titre: Hackathon sante Grand Nord. Theme: Le numerique au service de la sante",
            0, "Garage → hors sujet sante"
        ),
    ]

    print("\n--- Tests de validation ---")
    erreurs = 0
    for projet, challenge, attendu, description in test_cas:
        texte = preparer_texte(f"{projet} {challenge}")
        vecteur = vectorizer.transform([texte])
        prob = model.predict_proba(vecteur)[0][1]
        pred = 1 if prob >= 0.50 else 0
        statut = "OK  " if pred == attendu else "ECHEC"
        if pred != attendu:
            erreurs += 1
        print(f"[{statut}] conf={prob:.2f} | {description}")

    print(f"\nResultat : {len(test_cas) - erreurs}/{len(test_cas)} tests passes")

if __name__ == "__main__":
    entrainer()