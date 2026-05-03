import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

#API_URL = "http://127.0.0.1:8000"
API_URL = "https://projet7creditscoring.azurewebsites.net"

st.set_page_config(
    page_title="Dashboard scoring crédit",
    layout="wide"
)


# -----------------------------
# Fonctions API
# -----------------------------

def get_prediction(client_id, client_data=None):
    try:
        if client_data is None:
            response = requests.post(
                f"{API_URL}/prediction",
                json={"client_id": int(client_id)},
                timeout=10
            )
        else:
            response = requests.post(
                f"{API_URL}/prediction",
                json={
                    "client_id": int(client_id),
                    "client_data": client_data
                },
                timeout=10
            )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException:
        return None


def get_client_info(client_id):
    try:
        response = requests.get(
            f"{API_URL}/client_info/{int(client_id)}",
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
            f"{API_URL}/local_feature_importance/{int(client_id)}",
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


# -----------------------------
# Fonctions utiles
# -----------------------------

def format_prediction(payload):
    if payload is None:
        return None

    if "prediction" in payload:
        proba = float(payload["prediction"])
    else:
        proba = float(payload.get("prediction_proba", 0))

    threshold = float(payload.get("threshold", 0.2))

    if proba >= threshold:
        decision = "Crédit refusé"
    else:
        decision = "Crédit accordé"

    return proba, threshold, decision


def interpretation_score(proba, threshold):
    distance = abs(proba - threshold)

    if proba >= threshold:
        if distance < 0.05:
            return (
                "Le dossier est refusé, mais il est proche du seuil. "
                "La décision peut être considérée comme limite."
            )
        return (
            "Le dossier est refusé car la probabilité de défaut est supérieure "
            "au seuil retenu par le modèle."
        )

    if distance < 0.05:
        return (
            "Le dossier est accepté, mais il est proche du seuil. "
            "La décision peut être considérée comme limite."
        )

    return (
        "Le dossier est accepté car la probabilité de défaut est inférieure "
        "au seuil retenu par le modèle."
    )


def make_gauge(proba, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba,
            number={"valueformat": ".1%"},
            title={"text": "Probabilité de défaut"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%"},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, threshold], "color": "#D9EAD3"},
                    {"range": [threshold, 1], "color": "#F4CCCC"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold
                }
            }
        )
    )

    fig.update_layout(height=300)
    return fig


def prepare_importance_df(importance_dict, top_n=10):
    df = pd.DataFrame(
        importance_dict.items(),
        columns=["Variable", "Importance"]
    )

    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce")
    df = df.dropna()
    df["Importance_absolue"] = df["Importance"].abs()

    return df.sort_values("Importance_absolue", ascending=False).head(top_n)


def get_numeric_features(data, client_info):
    features = []

    for col in data.columns:
        if col == "TARGET":
            continue

        if col in client_info and pd.api.types.is_numeric_dtype(data[col]):
            features.append(col)

    return features


# -----------------------------
# Interface
# -----------------------------

st.title("Dashboard de scoring crédit")
st.write(
    "Ce dashboard permet de consulter le score d'un client, "
    "de comprendre les principales variables qui influencent la décision "
    "et de comparer le client avec les autres clients."
)

with st.sidebar:
    st.header("Recherche client")

    client_id = st.number_input(
        "Identifiant client",
        min_value=1,
        step=1,
        value=1
    )

    analyse = st.button("Lancer l'analyse")

    st.markdown("---")
    st.write("API utilisée :")
    st.code(API_URL)

    st.markdown("---")
    st.caption(
        "Les couleurs utilisées sont simples : vert pour une situation favorable, "
        "rouge pour une situation risquée. Les graphiques sont accompagnés "
        "d'explications textuelles."
    )


client_info = get_client_info(client_id)

if client_info is None:
    st.warning(
        "Impossible de récupérer les informations du client. "
        "Vérifiez que l'API est bien disponible."
    )
    st.stop()


if not analyse:
    st.info("Sélectionnez un client puis cliquez sur 'Lancer l'analyse'.")
    st.stop()


prediction_payload = get_prediction(client_id)
prediction = format_prediction(prediction_payload)

if prediction is None:
    st.error("La prédiction n'a pas pu être récupérée.")
    st.stop()

proba, threshold, decision = prediction

train_data = load_train_data()


# -----------------------------
# Résultat principal
# -----------------------------

st.subheader("Résultat du scoring")

col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(
        make_gauge(proba, threshold),
        use_container_width=True
    )

with col2:
    st.metric("Probabilité de défaut", f"{proba:.1%}")
    st.metric("Seuil de décision", f"{threshold:.1%}")
    st.metric("Décision", decision)

    if decision == "Crédit accordé":
        st.success("Décision du modèle : crédit accordé")
    else:
        st.error("Décision du modèle : crédit refusé")

    st.write(interpretation_score(proba, threshold))

st.markdown("---")


# -----------------------------
# Onglets
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Informations client",
    "Explication du score",
    "Comparaison avec les autres clients",
    "Analyse bi-variée",
    "Simulation"
])


# -----------------------------
# Onglet 1 : infos client
# -----------------------------

with tab1:
    st.subheader("Principales informations du client")

    client_df = pd.DataFrame(
        client_info.items(),
        columns=["Variable", "Valeur"]
    )

    st.dataframe(
        client_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "Ce tableau reprend les informations disponibles pour le client sélectionné."
    )


# -----------------------------
# Onglet 2 : explication
# -----------------------------

with tab2:
    st.subheader("Variables qui influencent le plus ce dossier")

    local_importance = get_local_feature_importance(client_id)
    global_importance = get_global_feature_importance()

    if local_importance:
        local_df = prepare_importance_df(local_importance, top_n=10)

        fig_local = px.bar(
            local_df,
            x="Importance",
            y="Variable",
            orientation="h",
            title="Importance locale des variables"
        )

        fig_local.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Impact sur la prédiction",
            yaxis_title="Variable"
        )

        st.plotly_chart(fig_local, use_container_width=True)

        st.write(
            "Ce graphique montre les variables qui ont le plus influencé "
            "la décision pour ce client précis."
        )

    else:
        st.info("L'importance locale n'est pas disponible.")

    st.markdown("---")

    st.subheader("Comparaison avec l'importance globale")

    if local_importance and global_importance:
        local_df = prepare_importance_df(local_importance, top_n=10)
        global_df = prepare_importance_df(global_importance, top_n=20)

        compare_df = local_df[["Variable", "Importance"]].merge(
            global_df[["Variable", "Importance"]],
            on="Variable",
            how="left",
            suffixes=("_locale", "_globale")
        )

        fig_compare = px.bar(
            compare_df,
            x="Variable",
            y=["Importance_locale", "Importance_globale"],
            barmode="group",
            title="Importance locale et importance globale"
        )

        fig_compare.update_layout(
            xaxis_title="Variable",
            yaxis_title="Importance"
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        st.write(
            "L'importance locale concerne uniquement le client étudié. "
            "L'importance globale correspond au comportement moyen du modèle "
            "sur l'ensemble des clients."
        )

    elif global_importance:
        global_df = prepare_importance_df(global_importance, top_n=10)

        fig_global = px.bar(
            global_df,
            x="Importance",
            y="Variable",
            orientation="h",
            title="Importance globale des variables"
        )

        fig_global.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Importance moyenne",
            yaxis_title="Variable"
        )

        st.plotly_chart(fig_global, use_container_width=True)

    else:
        st.info("L'importance globale n'est pas disponible.")


# -----------------------------
# Onglet 3 : comparaison
# -----------------------------

with tab3:
    st.subheader("Comparer le client avec les autres clients")

    numeric_features = get_numeric_features(train_data, client_info)

    if numeric_features:
        selected_feature = st.selectbox(
            "Variable à comparer",
            numeric_features
        )

        fig = px.histogram(
            train_data,
            x=selected_feature,
            nbins=40,
            title=f"Distribution de la variable : {selected_feature}"
        )

        try:
            client_value = float(client_info[selected_feature])

            fig.add_vline(
                x=client_value,
                line_dash="dash",
                line_color="black",
                annotation_text="Client sélectionné"
            )

            st.write(
                f"Valeur du client pour cette variable : **{client_value:.2f}**"
            )

        except Exception:
            pass

        fig.update_layout(
            xaxis_title=selected_feature,
            yaxis_title="Nombre de clients"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "Ce graphique permet de voir si le client se situe plutôt dans "
            "les valeurs basses, moyennes ou élevées par rapport aux autres clients."
        )

        if "TARGET" in train_data.columns:
            st.markdown("---")
            st.subheader("Comparaison par groupe de clients")

            fig_box = px.box(
                train_data,
                x="TARGET",
                y=selected_feature,
                title=f"{selected_feature} selon le statut réel du client"
            )

            fig_box.update_layout(
                xaxis_title="TARGET",
                yaxis_title=selected_feature
            )

            st.plotly_chart(fig_box, use_container_width=True)

            st.write(
                "Ce graphique compare la variable sélectionnée selon les groupes "
                "de clients de la base d'entraînement."
            )

    else:
        st.info("Aucune variable numérique comparable n'est disponible.")


# -----------------------------
# Onglet 4 : analyse bi-variée
# -----------------------------

with tab4:
    st.subheader("Analyse entre deux variables")

    numeric_features = get_numeric_features(train_data, client_info)

    if len(numeric_features) >= 2:
        col_x, col_y = st.columns(2)

        with col_x:
            x_feature = st.selectbox(
                "Variable en abscisse",
                numeric_features,
                index=0
            )

        with col_y:
            y_feature = st.selectbox(
                "Variable en ordonnée",
                numeric_features,
                index=1
            )

        color_col = "TARGET" if "TARGET" in train_data.columns else None

        fig_scatter = px.scatter(
            train_data,
            x=x_feature,
            y=y_feature,
            color=color_col,
            opacity=0.5,
            title=f"Relation entre {x_feature} et {y_feature}"
        )

        try:
            fig_scatter.add_scatter(
                x=[float(client_info[x_feature])],
                y=[float(client_info[y_feature])],
                mode="markers",
                marker={
                    "size": 14,
                    "color": "black",
                    "symbol": "x"
                },
                name="Client sélectionné"
            )
        except Exception:
            pass

        fig_scatter.update_layout(
            xaxis_title=x_feature,
            yaxis_title=y_feature
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.write(
            "Ce graphique permet d'observer la position du client par rapport "
            "aux autres clients sur deux variables choisies."
        )

    else:
        st.info("Il faut au moins deux variables numériques pour cette analyse.")


# -----------------------------
# Onglet 5 : simulation
# -----------------------------

with tab5:
    st.subheader("Simulation d'une modification du dossier")

    st.write(
        "Cette partie permet de modifier certaines informations du client "
        "et de demander un nouveau score à l'API, si l'API accepte ce format."
    )

    numeric_features = get_numeric_features(train_data, client_info)

    if numeric_features:
        selected_features = st.multiselect(
            "Variables à modifier",
            numeric_features,
            default=numeric_features[:3]
        )

        modified_client = client_info.copy()

        for feature in selected_features:
            try:
                current_value = float(client_info[feature])
                min_value = float(train_data[feature].min())
                max_value = float(train_data[feature].max())

                modified_value = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=current_value
                )

                modified_client[feature] = modified_value

            except Exception:
                st.write(f"{feature} ne peut pas être modifiée ici.")

        if st.button("Calculer un nouveau score"):
            new_prediction_payload = get_prediction(
                client_id,
                client_data=modified_client
            )

            new_prediction = format_prediction(new_prediction_payload)

            if new_prediction is None:
                st.warning(
                    "La simulation n'a pas pu être effectuée. "
                    "L'API actuelle ne permet peut-être pas encore "
                    "de recalculer un score avec des données modifiées."
                )
            else:
                new_proba, new_threshold, new_decision = new_prediction

                col_a, col_b, col_c = st.columns(3)

                col_a.metric(
                    "Ancienne probabilité",
                    f"{proba:.1%}"
                )

                col_b.metric(
                    "Nouvelle probabilité",
                    f"{new_proba:.1%}",
                    delta=f"{new_proba - proba:.1%}"
                )

                col_c.metric(
                    "Nouvelle décision",
                    new_decision
                )

                st.write(interpretation_score(new_proba, new_threshold))

    else:
        st.info("Aucune variable numérique modifiable n'est disponible.")

