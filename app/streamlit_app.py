import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import streamlit as st
from streamlit_ketcher import st_ketcher

import pandas as pd
import matplotlib.pyplot as plt
import joblib

literature_df = pd.read_csv(
    "data/raw_surfactant.csv"
)

# ============================================================
# LOAD TRAINING RESULTS
# ============================================================

processed_df = pd.read_csv(
    "data/processed_surfactants.csv"
)

model_xgb = joblib.load(
    "models/trained_model_xgb.pkl"
)

# Remove non-feature columns
exclude_cols = [
    "surfactant_name",
    "smiles",
    "pCMC",
    "ionic_type",
    "temperature",
    "HLB",
    "Source"
]

feature_columns = [
    col for col in processed_df.columns
    if col not in exclude_cols
]

X_plot = processed_df[feature_columns]

y_true = processed_df["pCMC"]

y_pred = model_xgb.predict(X_plot)

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
from PIL import Image

# ============================================================
# MORGAN FINGERPRINT GENERATOR
# ============================================================

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=128
)

# ============================================================
# GENERATE FINGERPRINT
# ============================================================

def get_fp(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return morgan_gen.GetFingerprint(mol)

# ============================================================
# FIND MOST SIMILAR SURFACTANT
# ============================================================

def find_most_similar_surfacant(query_smiles):

    query_fp = get_fp(query_smiles)

    if query_fp is None:
        return None

    best_similarity = -1

    best_row = None

    for _, row in literature_df.iterrows():

        db_smiles = row["smiles"]

        db_fp = get_fp(db_smiles)

        if db_fp is None:
            continue

        similarity = DataStructs.TanimotoSimilarity(
            query_fp,
            db_fp
        )

        if similarity > best_similarity:

            best_similarity = similarity

            best_row = row

    return {
        "name": best_row["surfactant_name"],
        "smiles": best_row["smiles"],
        "pCMC": best_row["pCMC"],
        "similarity": best_similarity
    }

# ============================================================
# PARITY PLOT
# ============================================================

def make_parity_plot(
    y_true,
    y_pred,
    literature_pcmc,
    literature_pred_pcmc,
    query_pred_pcmc,
    similar_name
):

    fig, ax = plt.subplots(
        figsize=(8,8)
    )

    # ========================================================
    # MAIN TRAINING DATA
    # ========================================================

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.7,
        label="Training Surfactants"
    )

    # ========================================================
    # PERFECT PREDICTION LINE
    # ========================================================

    min_val = min(
        min(y_true),
        min(y_pred)
    )

    max_val = max(
        max(y_true),
        max(y_pred)
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Perfect Prediction"
    )

    # ========================================================
    # CLOSEST LITERATURE SURFACTANT
    # ========================================================

    ax.scatter(
        literature_pcmc,
        literature_pred_pcmc,
        s=350,
        marker="*",
        label=f"Closest Match: {similar_name}"
    )

    # ========================================================
    # USER QUERY PREDICTION
    # ========================================================

    ax.scatter(
        literature_pcmc,
        query_pred_pcmc,
        s=250,
        marker="X",
        label="Your Molecule"
    )

    # ========================================================
    # LABELS
    # ========================================================

    ax.set_xlabel(
        "Experimental Literature pCMC",
        fontsize=12
    )

    ax.set_ylabel(
        "XGBoost Predicted pCMC",
        fontsize=12
    )

    ax.set_title(
        "XGBoost Model Parity Plot",
        fontsize=16,
        pad=20
    )

    ax.legend()

    return fig

from src.predict import predict_pcmc

# ============================================================
# COMMON SURFACTANTS
# ============================================================

common_surfactants = {
    "SDS": {
        "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]",
        "ionic": "anionic"
    },

    "CTABr": {
        "smiles": "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]",
        "ionic": "cationic"
    },

    "Tween 80": {
        "smiles": "CCCCCCCC=CCCCCCCCCCCC(=O)OCC(CO)OCC(CO)OCC(CO)O",
        "ionic": "nonionic"
    },

    "Brij 35": {
        "smiles": "CCCCCCCCCCCCOCCOCCOCCO",
        "ionic": "nonionic"
    },

    "SLES": {
        "smiles": "CCCCCCCCCCCCOCCOCCOS(=O)(=O)[O-].[Na+]",
        "ionic": "anionic"
    }
}

# ============================================================
# PAGE TITLE
# ============================================================

st.title("Surfactant CMC Predictor")

st.write(
    "Draw a surfactant molecule or paste a SMILES string." \
    " Press Apply once complete"
)

#Dropdown for common surfactant
selected = st.selectbox(
    "Load common surfactant",
    ["Custom"] + list(common_surfactants.keys())
)

#Set default
default_smiles = ""

default_ionic = "nonionic"

if selected != "Custom":

    default_smiles = common_surfactants[selected]["smiles"]

    default_ionic = common_surfactants[selected]["ionic"]

# ============================================================
# MOLECULE DRAWER
# ============================================================

smiles = st_ketcher(default_smiles)

# ============================================================
# IONIC TYPE
# ============================================================

ionic_type = st.selectbox(
    "Select ionic type",
    [
        "anionic",
        "cationic",
        "zwitterionic",
        "nonionic"
    ],
    index=[
        "anionic",
        "cationic",
        "zwitterionic",
        "nonionic"
    ].index(default_ionic)
)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Predict CMC"):

    if smiles:
        try:
            pred_pcmc, pred_cmc = predict_pcmc(
                smiles,
                ionic_type
            )

            #literature_pcmc = lookup_literature(smiles)
            similar_result = find_most_similar_surfacant(smiles)

            st.success("Prediction Complete")
            st.write(f"SMILES: {smiles}")
            st.metric(
                "Predicted pCMC",
                f"{pred_pcmc:.3f}"
            )

            st.metric(
                "Predicted CMC (mM)",
                f"{pred_cmc:.4f}"
            )

            # ============================================================
            # Similarity Search Results
            # ============================================================
            st.subheader("Closest Literature Surfactant")
            st.write(f"Name: {similar_result['name']}")
            
            similar_mol = Chem.MolFromSmiles(similar_result["smiles"])
            if similar_mol is not None:
                mol_image = Draw.MolToImage(
                    similar_mol,
                    size = (400, 400)
                )

            lit_cmc = 10 ** (-similar_result["pCMC"]) * 1000

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Literature pCMC",
                    f"{similar_result['pCMC']}"
                )

                st.metric(
                    "Simialrity",
                    f"{similar_result['similarity']:.3f}"
                )
            with col2:
                st.image(
                    mol_image,
                    caption = similar_result["name"]
                )

            # ============================================================
            # PARITY PLOT
            # ============================================================
            literature_pred_pcmc, _ = predict_pcmc(
                similar_result["smiles"],
                ionic_type
            )

            # ============================================================
            # MODEL PERFORMANCE SECTION
            # ============================================================

            st.markdown("---")

            st.markdown(
                "<h2 style='text-align: center;'>"
                "Model Performance"
                "</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style='text-align: center; font-size: 18px;'>

                This parity plot compares the XGBoost model predictions
                against experimental literature pCMC values for all
                surfactants in the training dataset.
                
                The dashed diagonal line represents perfect prediction.

                ★ Star Marker:
                Closest literature surfactant found using Morgan
                fingerprint similarity.

                ✖ X Marker:
                Prediction for your queried molecule.

                </div>
                """,
                unsafe_allow_html=True
            )

            fig = make_parity_plot(
                y_true,
                y_pred,
                similar_result["pCMC"],
                literature_pred_pcmc,
                pred_pcmc,
                similar_result["name"]
            )

            col1, col2, col3 = st.columns([1, 3, 1])

            with col2:
                st.pyplot(fig)
                

        except Exception as e:
            st.error(str(e))

    else:
        st.warning("Please draw a molecule.")


#Run streamlit run app/streamlit_app.py
#End by pressing control^ + C (NOT command + C) on mac

# Can try to plot predicted vs literature??