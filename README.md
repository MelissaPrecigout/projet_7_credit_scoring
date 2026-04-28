\# Projet 7 - Scoring crédit



\## Objectif du projet



Ce projet a pour objectif de développer un modèle de scoring crédit permettant d’estimer le risque de défaut d’un client.  

Le modèle est ensuite déployé sous forme d’API afin d’être consommé par une interface utilisateur Streamlit.



Le projet répond à une problématique métier : aider à la décision d’octroi de crédit tout en fournissant des éléments d’explicabilité.



\## Contenu du repository



```text

projet\_7\_credit\_scoring/

│

├── notebook\_modelisation.ipynb

├── api.py

├── dashboard.py

├── test\_api.py

├── requirements.txt

├── runtime.txt

├── startup.sh

├── README.md

│

├── models/

│   ├── final\_pipeline.joblib

│   └── final\_threshold.joblib

│

├── reports/

│   └── data\_drift\_report.html

│

└── .github/

&#x20;   └── workflows/

&#x20;       ├── main\_projet7creditscoring.yml

&#x20;       └── main\_projet7creditscoringdashboard.yml

