from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

import re
from rapidfuzz import process, fuzz
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pubchempy as pcp

# ============================================
# 🔥 LOAD FILES
# ============================================
DATA_PATH = "Data/"
MODEL_PATH = "Model/"

smiles_cache = joblib.load(DATA_PATH + "smiles_cache.pkl")
INTERACTION_LOOKUP = joblib.load(DATA_PATH + "interaction_lookup.pkl")
drug_encoder = joblib.load(DATA_PATH + "drug_encoder.pkl")
label_encoder = joblib.load(DATA_PATH + "label_encoder.pkl")

model_binary = joblib.load(MODEL_PATH + "model_binary.pkl")
model_severity = joblib.load(MODEL_PATH + "model_severity.pkl")
xgb_model = joblib.load(MODEL_PATH + "xgb_model.pkl")
rf_model = joblib.load(MODEL_PATH + "rf_model.pkl")

# ============================================
# 🔹 APP
# ============================================

app = FastAPI(title="🔥 Drug Interaction Hybrid API")

class DrugRequest(BaseModel):
    drugs: list[str]

# ============================================
# 🔹 KNOWLEDGE BASE 🔥
# ============================================

KNOWN_MAJOR_INTERACTIONS = {
    tuple(sorted(["warfarin", "ibuprofen"])): "Major",
    tuple(sorted(["warfarin", "aspirin"])): "Major",
    tuple(sorted(["isotretinoin", "retinol"])): "Major",
    tuple(sorted(["clarithromycin", "simvastatin"])): "Major"
}

SAFE_COMBINATIONS = {
    tuple(sorted(["ibuprofen", "ascorbic acid"])),
    tuple(sorted(["paracetamol", "ascorbic acid"])),
    tuple(sorted(["ibuprofen", "paracetamol"]))
}

# ============================================
# 🔹 CLEAN + NORMALIZE 🔥
# ============================================

def basic_clean(text):
    return re.sub(r'[^a-zA-Z0-9\s/()-]', '', str(text).lower().strip())

def remove_dose(text):
    return re.sub(r'\b\d+\.?\d*\s*(mg|g|ml|mcg|µg|iu)\b', '', str(text).lower()).strip()

def normalize_drug(name, threshold=90):

    query = basic_clean(remove_dose(name))

    if not query:
        return None

    # vitamins fix 🔥
    if "vitamin c" in query or "vit c" in query:
        return "ascorbic acid"
    if "vitamin a" in query or "vit a" in query:
        return "retinol"

    # mapping
    mapping = {
        "paracetamol": ["panadol", "acetaminophen"],
        "ibuprofen": ["brufen", "advil"],
        "amoxicillin/clavulanic acid": ["augmentin"],
        "clarithromycin": ["klacid"],
        "warfarin": ["coumadin"]
    }

    for k, v in mapping.items():
        if query == k or query in v:
            return k

    # exact
    if query in drug_encoder.classes_:
        return query

    # fuzzy
    result = process.extractOne(query, drug_encoder.classes_, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]

    return query

# ============================================
# 🔹 SMILES
# ============================================

def get_smiles(drug):
    if drug in smiles_cache:
        return smiles_cache[drug]

    try:
        compounds = pcp.get_compounds(drug, 'name')
        if compounds:
            return compounds[0].connectivity_smiles
    except:
        pass

    return None

def fingerprint(smiles):
    if not smiles:
        return np.zeros(166)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(166)

    return np.array(MACCSkeys.GenMACCSKeys(mol))

def build_features(sm1, sm2):
    f1 = fingerprint(sm1)
    f2 = fingerprint(sm2)

    if f1.sum() == 0 or f2.sum() == 0:
        return None

    return np.concatenate([
        f1, f2,
        np.abs(f1 - f2),
        f1 * f2,
        f1 + f2,
        (f1 + f2) / 2
    ]).reshape(1, -1)

# ============================================
# 🔥 PREDICTION ENGINE (FULL 🔥🔥)
# ============================================

def predict(drug1, drug2):

    d1 = normalize_drug(drug1)
    d2 = normalize_drug(drug2)

    if not d1 or not d2:
        return None

    pair = tuple(sorted([d1, d2]))

    # 🥇 KNOWLEDGE
    if pair in KNOWN_MAJOR_INTERACTIONS:
        return {
            "drug1": d1,
            "drug2": d2,
            "severity": "Major",
            "source": "Knowledge",
            "confidence": 1.0
        }

    # 🥈 SAFE
    if pair in SAFE_COMBINATIONS:
        return None

    # 🥉 DATABASE
    if pair in INTERACTION_LOOKUP:
        r = INTERACTION_LOOKUP[pair]

        if r["level"] == "No Interaction":
            return None

        return {
            "drug1": d1,
            "drug2": d2,
            "severity": r["level"],
            "source": "Database",
            "confidence": 0.95
        }

    # 🧠 ML
    try:
        if d1 in drug_encoder.classes_ and d2 in drug_encoder.classes_:

            a = drug_encoder.transform([d1])[0]
            b = drug_encoder.transform([d2])[0]

            prob = model_binary.predict_proba([[a, b]])[0]
            pred = np.argmax(prob)
            conf = prob[pred]

            if conf < 0.6:
                raise Exception()

            if pred == 0:
                return None

            prob2 = model_severity.predict_proba([[a, b]])[0]
            pred2 = np.argmax(prob2)
            conf2 = prob2[pred2]

            sev = label_encoder.inverse_transform([pred2])[0]

            if sev == "Major":
                sev = "Moderate"

            if conf2 < 0.8:
                return None

            return {
                "drug1": d1,
                "drug2": d2,
                "severity": sev,
                "source": "ML",
                "confidence": float(conf2)
            }

    except:
        pass

    # 🧪 SMILES (ENSEMBLE 🔥)
    sm1, sm2 = get_smiles(d1), get_smiles(d2)

    if sm1 and sm2:
        f = build_features(sm1, sm2)

        if f is not None:
            xgb_prob = xgb_model.predict_proba(f)[0]
            rf_prob = rf_model.predict_proba(f)[0]

            final_prob = (xgb_prob * 0.7) + (rf_prob * 0.3)

            pred = np.argmax(final_prob)
            conf = float(np.max(final_prob))

            labels = ["No Interaction", "Minor", "Moderate", "Major"]

            if conf > 0.9 and labels[pred] != "No Interaction":
                return {
                    "drug1": d1,
                    "drug2": d2,
                    "severity": labels[pred],
                    "source": "SMILES",
                    "confidence": conf
                }

    return None

# ============================================
# 🔹 CHECK ALL
# ============================================

def check_all(drugs):

    results = []
    seen = set()

    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):

            pair = tuple(sorted([drugs[i], drugs[j]]))

            if pair in seen:
                continue
            seen.add(pair)

            res = predict(drugs[i], drugs[j])

            if res:
                results.append(res)

    return results

# ============================================
# 🔹 ROUTES
# ============================================

@app.get("/")
def home():
    return {"message": "🚀 API Running"}

@app.post("/check")
def check(request: DrugRequest):

    drugs = [normalize_drug(d) for d in request.drugs]
    drugs = [d for d in drugs if d]

    if len(drugs) < 2:
        return {"error": "Enter at least 2 drugs"}

    results = check_all(drugs)

    return {
        "input": drugs,
        "has_interaction": len(results) > 0,
        "message": "⚠️ Interaction detected" if results else "✅ No interaction",
        "conflicts": results
    }