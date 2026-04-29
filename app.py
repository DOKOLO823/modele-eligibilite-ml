import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import unicodedata
import re

app = Flask(__name__)
CORS(app)

model = joblib.load('modele_eligibilite.pkl')
vectorizer = joblib.load('vectoriseur.pkl')

def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return ""
    texte = "".join(
        c for c in unicodedata.normalize('NFD', texte)
        if unicodedata.category(c) != 'Mn'
    )
    texte = texte.lower()
    texte = re.sub(r'[^\w\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def preparer_texte(texte):
    return nettoyer_texte(texte)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"erreur": "Corps JSON manquant ou invalide"}), 400

    projet_raw    = data.get('projet_data', '').strip()
    challenge_raw = data.get('challenge_info', '').strip()

    if not projet_raw:
        return jsonify({"erreur": "Le champ 'projet_data' est requis"}), 400
    if not challenge_raw:
        return jsonify({"erreur": "Le champ 'challenge_info' est requis"}), 400

    projet_clean    = preparer_texte(projet_raw)
    challenge_clean = preparer_texte(challenge_raw)
    texte_final     = f"{projet_clean} {projet_clean} {challenge_clean}"

    vecteur     = vectorizer.transform([texte_final])
    probabilite = float(model.predict_proba(vecteur)[0][1])

    seuil      = 0.50
    prediction = 1 if probabilite >= seuil else 0

    raison = ""
    if prediction == 0:
        if probabilite < 0.30:
            raison = "Le projet ne correspond pas au theme et aux exigences du challenge."
        else:
            raison = "Le projet ne semble pas suffisamment aligne avec les objectifs du challenge."

    return jsonify({
        "eligible":  prediction,
        "confiance": round(probabilite, 2),
        "raison":    raison
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"statut": "ok", "modele": "charge"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)