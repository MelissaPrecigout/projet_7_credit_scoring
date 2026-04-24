import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Scoring crédit",
    layout="wide"
)


def get_prediction(client_id):
    try:
        response = requests.post(
            f"{API_URL}/prediction",
            json={"client_id": client_id},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def get_client_info(client_id):
    try:
        response = requests.get(
            f"{API_URL}/client_info/{client_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def get_local_feature_importance(client_id):
    try:
        response = requests.get(
            f"{API_URL}/local_feature_importance/{client_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def get_global_feature_importance():
    try:
        response = requests.get(
            f"{API_URL}/global_feature_importance",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data
def load_train_data():
    return pd.read_csv("train_mean_sample.csv")


def format_prediction(payload):
    if payload is None:
        return None

    if "prediction" in payload:
        proba = float(payload["prediction"])
    else:
        proba = float(payload.get("prediction_proba", 0))

    threshold = float(payload.get("threshold", 0.2))
    decision = "Crédit refusé" if proba >= threshold else "Crédit accordé"

    return proba, threshold, decision


st.title("Scoring crédit - Prêt à dépenser")

st.write(
    "Cette interface permet de consulter le score de défaut d’un client, "
    "la décision associée et les principales variables explicatives."
)

with st.sidebar:
    st.header("Recherche client")
    client_id = st.number_input("Identifiant client", min_value=1, step=1, value=1)
    analyse = st.button("Lancer l'analyse")

    st.markdown("---")
    st.caption("API locale")
    st.code(API_URL)


client_info = get_client_info(client_id)

if client_info is None:
    st.warning(
        "Impossible de récupérer les informations client. "
        "Vérifiez que l'API est lancée avec `python api.py`."
    )
    st.stop()


if analyse:
    prediction_payload = get_prediction(client_id)
    prediction = format_prediction(prediction_payload)

    if prediction is None:
        st.error("La prédiction n'a pas pu être récupérée.")
        st.stop()

    proba, threshold, decision = prediction

    st.subheader("Résultat de l'analyse")

    col1, col2, col3 = st.columns(3)

    col1.metric("Probabilité de défaut", f"{proba:.1%}")
    col2.metric("Seuil retenu", f"{threshold:.1%}")
    col3.metric("Décision", decision)

    if decision == "Crédit accordé":
        st.success("Le dossier est classé comme favorable selon le modèle.")
    else:
        st.error("Le dossier est classé comme risqué selon le modèle.")

    st.progress(min(max(proba, 0), 1))

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Synthèse client",
        "Explication individuelle",
        "Variables du modèle",
        "Comparaison"
    ])

    with tab1:
        st.subheader("Informations du client")

        client_df = pd.DataFrame(
            client_info.items(),
            columns=["Variable", "Valeur"]
        )

        st.dataframe(
            client_df,
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.subheader("Variables influençant le plus ce dossier")

        local_importance = get_local_feature_importance(client_id)

        if local_importance:
            local_df = pd.DataFrame(
                local_importance.items(),
                columns=["Variable", "Importance"]
            ).sort_values("Importance", ascending=False).head(10)

            fig = px.bar(
                local_df,
                x="Importance",
                y="Variable",
                orientation="h",
                title="Importance locale des variables"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("L'explication individuelle n'est pas disponible.")

    with tab3:
        st.subheader("Variables les plus importantes globalement")

        global_importance = get_global_feature_importance()

        if global_importance:
            global_df = pd.DataFrame(
                global_importance.items(),
                columns=["Variable", "Importance"]
            )
            global_df["Importance"] = pd.to_numeric(
                global_df["Importance"],
                errors="coerce"
            )
            global_df = global_df.sort_values(
                "Importance",
                ascending=False
            ).head(10)

            fig = px.bar(
                global_df,
                x="Importance",
                y="Variable",
                orientation="h",
                title="Importance globale des variables"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("L'importance globale n'est pas disponible.")

    with tab4:
        st.subheader("Comparaison avec la population")

        train_data = load_train_data()

        available_features = [
            col for col in train_data.columns
            if col in client_info and col != "TARGET"
        ]

        if available_features:
            selected_feature = st.selectbox(
                "Variable à comparer",
                available_features
            )

            fig = px.histogram(
                train_data,
                x=selected_feature,
                title=f"Distribution de {selected_feature}"
            )

            try:
                fig.add_vline(
                    x=float(client_info[selected_feature]),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Client"
                )
            except Exception:
                pass

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune variable comparable disponible.")

else:
    st.info("Sélectionnez un client puis lancez l'analyse.")