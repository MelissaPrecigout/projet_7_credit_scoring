import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

import lightgbm
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.models.evaluation import evaluate
import shap
import gc

from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric

''' FONCTIONS GRAPHIQUES'''

def correlation_matrix(data, var_num, figsize=(10,8)):
    """
    Affiche une heatmap de la matrice de corrélation entre les colonnes numériques du DataFrame.

    """

    #Calcul de la matrice de corrélation : 
    correlation_matrix = data[var_num].corr()

    #Création d'un masque pour masquer la partie supérieure du triangle : 
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    #Affichage de la heatmap :
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, cmap='coolwarm', mask=mask, annot=True, fmt='.2f')
    plt.title("Matrice de Corrélation des variables numériques")
    plt.show()

def plot_distrib_var_object(data, var_categorielle, figsize=(18, 6)):
    """
    Affiche la répartition d'une variable catégorielle et le pourcentage de défaillants dans chaque catégorie.

    Cette fonction génère deux graphiques barplot pour analyser la répartition d'une variable catégorielle dans un DataFrame.
    Le premier graphique affiche le nombre d'occurrences de chaque catégorie, tandis que le deuxième graphique affiche le
    pourcentage de défaillants (TARGET = 1) dans chaque catégorie.

    Args:
        data (DataFrame): Le DataFrame contenant les données.
        var_categorielle (str): Le nom de la variable catégorielle à analyser.
        figsize (tuple, optional): La taille de la figure pour afficher les graphiques. Par défaut, (18, 6).

    """
    
    # 1. Calcul du nombre d'occurrences total: 
    data_to_plot = data[var_categorielle].value_counts().sort_values(ascending=False).to_frame('Nombre').reset_index()
    
    # 2. Calcul du % de defaillants dans chaque catégories : 
    pourcentage_defaillants_par_categories = data.loc[data['TARGET'] == 1, var_categorielle].value_counts() / data[var_categorielle].value_counts()  * 100
    pourcentage_defaillants_par_categories = pourcentage_defaillants_par_categories.sort_values(ascending=False).to_frame('%').reset_index()
    
    # 3. Création des figures : 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 4. Graphique_1 : 
    sns.barplot(
        data=data_to_plot, 
        x=var_categorielle,
        y='Nombre',
        palette='Blues',
        ax=ax1,
    )
    
    for p in ax1.patches: 
        ax1.annotate(
            text=f"{round(p.get_height() / data_to_plot['Nombre'].sum() * 100, 2)}%",
            xy=(p.get_x(), p.get_height()),
            xytext=(-p.get_width()/2, 4),
            textcoords='offset points',
            fontsize='x-small',
        )
    
    ax1.set_title('Distribution des catégories \nsocio-professionnelles')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    # 5. Graphique_2 : 
    
    sns.barplot(
        data=pourcentage_defaillants_par_categories,
        x=var_categorielle,
        y='%',
        palette='Blues',
        ax=ax2
        )

    for p in ax2.patches: 
        ax2.annotate(
            text=f"{round(p.get_height(), 2)}%",
            xy=(p.get_x(), p.get_height()),
            xytext=(-p.get_width()/2, 4),
            textcoords='offset points',
            fontsize='x-small',
        )

    ax2.set_title('Distribution des catégories \nsocio-professionnelles chez les \'Défaillants\'')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    plt.show() 




''' FONCTIONS PREPROCESSING'''

def timer(title):
    '''Calcul du temps d'éxécution d'autres fonctions.
    '''
    t0 = time.time()
    yield
    print("{} - réalisé en {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True):
    '''Encodage des variables catégorielles.
    Keyword arguments:
    df -- dataframe
    nan_as_category -- ajout d'une colonne indiquant les NaN (default True)
    Returns:
    df -- dataframe encodé
    new_columns -- nouvelles colonnes créées par l'encodage.
    '''
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, dtype=int)
    new_columns = [c for c in df.columns if c not in original_columns]
    
    return df, new_columns


''' FEATURES SELECTION'''

def features_importance_lightgbm(df, num_folds=10, stratified=False, debug=False, class_weight=None): 

    # 1. Nettoyage des colonnes avec caractères JSON non supportés
    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    # 2. Séparation de df en test / train en fonction de la cible : 
    train = df.loc[df['TARGET'].notna()]
    test = df.loc[df['TARGET'].isna()]
    
    print(f"Dimension de train : {train.shape}")
    print(f"Dimension de test : {test.shape}")
    
    # 3. Validation croisée : 
    folds = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=42,
        )
    
    # 4. Création de X et y à partir de train :
    feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = train[feats]
    y = train['TARGET']

    # 5. Création des données de stockage : 
    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    final_preds = np.zeros(train.shape[0])

    # 6. Boucle sur les différents folds : 
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train['TARGET'])):
        train_x, train_y = train[feats].iloc[train_idx], train['TARGET'].iloc[train_idx]
        valid_x, valid_y = train[feats].iloc[valid_idx], train['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1,
            class_weight=class_weight, # Ajout de la gestion du déséquilibre des classes
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc')

        # Prédiction des probabilités de l'appartenance à la classe 1 sur la validation
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]

        # Prédiction des probabilités de l'appartenance à la classe 1 sur le test
        sub_preds += clf.predict_proba(test[feats])[:, 1] / folds.n_splits

        # Prédiction des classes sur la validation
        final_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)

        # Complétion de feature_importance_df
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    # Affichage du résultats total : 
    print(f"AUC score total : {round(roc_auc_score(y, oof_preds),6)}")
    
    # Création de la matrice de confusion
    cm = confusion_matrix(y, final_preds)
    print("Matrice de confusion :\n", cm)
    
    # Affichage de la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    return feature_importance_df

'''PREPROCESSING DATAFRAME'''

def pre_process_dataframe(df, fillna_strategy):
    """
    Prétraitement des données d'entraînement et de test.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données d'entraînement et de test.
        fillna_strategy (str): La stratégie d'imputation pour les valeurs manquantes.
            Doit être l'un des suivants : 'median', 'mean', 'mode'.

    Raises:
        ValueError: Si fillna_strategy n'est pas 'median', 'mean' ou 'mode'.

    Returns:
        pandas.DataFrame, pandas.DataFrame: Deux DataFrames, l'un pour les données d'entraînement prétraitées,
        l'autre pour les données de test prétraitées.
    """
    
    # 1. Vérification de fillna_strategy : 
    if fillna_strategy not in ['median', 'mean', 'mode']:
        raise ValueError("fillna_strategy doit être égal à 'median', 'mean' ou 'mode'.")
    
    # 2. Création de l'imputer : 
    if fillna_strategy == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    else:
        imputer = SimpleImputer(strategy=fillna_strategy)
        
    # 3. Séparation en train et test : 
    df_train = df.loc[df['TARGET'].notna()].reset_index(drop=True)
    df_test = df.loc[df['TARGET'].isna()].reset_index(drop=True)
    
    # 4. Récupération de la colonne TARGET : 
    target = df_train['TARGET'].copy()
    target = target.astype('int8')
    
    df_train = df_train.drop('TARGET', axis=1)
    df_test = df_test.drop('TARGET', axis=1)
   
    # 5. Gestion des valeurs infinies : 
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
   
    # 6. Imputation des valeurs manquantes : 
    df_train_imputed = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns)
    df_test_imputed = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns)
   
    # 7. Conversion des colonnes booléennes en int8 :
    bool_columns = df_train.select_dtypes(include=['bool']).columns
   
    df_train_imputed[bool_columns] = df_train_imputed[bool_columns].astype('int8')
    df_test_imputed[bool_columns] = df_test_imputed[bool_columns].astype('int8')
   
    # 8. Conversion des colonnes float64 en float32 :
    float_columns = df_train.select_dtypes(include=['float64']).columns
   
    df_train_imputed[float_columns] = df_train_imputed[float_columns].astype('float32')
    df_test_imputed[float_columns] = df_test_imputed[float_columns].astype('float32')
   
    # 9. Conversion des colonnes int64 en int8 :
    int_columns = df_train.select_dtypes(include=['int64']).columns
   
    df_train_imputed[int_columns] = df_train_imputed[int_columns].astype('int8')
    df_test_imputed[int_columns] = df_test_imputed[int_columns].astype('int8')
   
    # 10. Ajout de la colonne TARGET à df_train : 
    df_train_imputed['TARGET'] = target
    
    # 11. Affichage : 
    print('**'*50)
    print(f"\nTRAIN\n"
          f"Nombre de colonnes avec au moins une valeur manquante : {df_train_imputed.isna().any().sum()}"
          f"\ntype de données : {df_train_imputed.dtypes.unique().tolist()}\n"
          )
    print('**'*50)
    
    print('**'*50)
    print(f"\nTEST\n"
          f"Nombre de colonnes avec au moins une valeur manquante : {df_test_imputed.isna().any().sum()}"
          f"\ntype de données : {df_test_imputed.dtypes.unique().tolist()}\n"
          )
    print('**'*50)
    
    return df_train_imputed, df_test_imputed


def preprocess_data(data, reduce_fraction=None):
    # Séparation des données et de la Target : 
    X = data.drop('TARGET', axis=1)
    y = data['TARGET'].copy()
    
    # Séparation en ensemble de train et test : 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        
    # Gestion du déséquilibre de classe avec SMOTE : 
    #smote = SMOTE(sampling_strategy='auto', random_state=42)
    #X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Standardisation du test et du train :     
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test



'''MODELISATION'''

def score_metier(y_test, y_pred):
    '''Créer un score métier à partir de la matrice de confusion.
    :param: y_test (vraies valeurs), y_pred (valeurs prédites par le modèle)
    :return: gain (score métier)
    '''
    TP_coeff = 0       # Vrais positifs (prédit correctement une observation comme positive (1) lorsque la vraie classe est également positive (1).)
    FP_coeff = 1       # Faux positifs (prédit incorrectement une observation comme positive (1) alors que la vraie classe est négative (0). Aussi appelés erreurs de type I.)
    FN_coeff = +10     # Faux négatifs (prédit incorrectement une observation comme négative (0) alors que la vraie classe est positive (1). Aussi appelés erreurs de type II.)
    TN_coeff = 0       # Vrais négatifs (prédit correctement une observation comme négative (0) lorsque la vraie classe est également négative (0).)
    
    (TN, FP, FN, TP) = metrics.confusion_matrix(y_test, y_pred).ravel()
    
    gain = (TP*TP_coeff + TN*TN_coeff + FP*FP_coeff + FN*FN_coeff)/(TN+FP+FN+TP)
    
    return gain

def seuil_metier(model, X, y, thresholds):
    y_probs = model.predict_proba(X)[:, 1]
    best_score = float('inf')
    best_threshold = 0
    scores = []
    
    for threshold in thresholds:
        y_pred = [1 if prob > threshold else 0 for prob in y_probs]
        score = score_metier(y, y_pred)
        scores.append(score)
        
        if score < best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold, best_score, scores


def eval_score(model, X_test, y_test, seuil=0.5):
    '''Calcule, affiche et enregistre les différentes métriques.
    :param: model, X_test (dataframe de validation), y_test (vraies valeurs),
    seuil (seuil de détermination des prédictions)
    :return: affiche (et return) les métriques, la matrice de confusion et la courbe ROC.
    '''    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_pred_proba > seuil, 1, 0)
    
    metier = score_metier(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, zero_division=1)
    f1_score = metrics.f1_score(y_test, y_pred, zero_division=1)
    fbeta_score = metrics.fbeta_score(y_test, y_pred, beta=2, zero_division=1)
    rocauc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    print('Score métier :{:.2f}'.format(metier))
    print('Accuracy score : {:.2f}'.format(accuracy))
    print('Precision score : {:.2f}'.format(precision))
    print('Recall score : {:.2f}'.format(recall))
    print('F1 score : {:.2f}'.format(f1_score))
    print('Fbeta score : {:.2f}'.format(fbeta_score))
    print('ROC AUC score : {:.2f}'.format(rocauc))
    
    
    # Matrice de confusion
    conf_mat = metrics.confusion_matrix(y_test,y_pred)
    plt.figure(figsize = (6,4))
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Matrice de confusion')
    plt.show()

    # Courbe ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Courbe ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    return metier, accuracy, precision, recall, f1_score, fbeta_score, rocauc, y_pred_proba