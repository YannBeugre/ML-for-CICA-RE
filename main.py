import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor



st.write('''
    # Application de Tarification des Facultatives de Réassurance.
    Cette App calcul la prime de réassurance des facultatives.
    ''')

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_coding = load_lottiefile("/Users/apple/Desktop/Dossier Yahn/coding.json")


st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
)

st.markdown("***Powered by : Yann BEUGRE, Actuaire Non-vie chez CICA-RE***")

chemin_image = '/Users/apple/Desktop/Dossier Yahn/AG.PNG'
st.image(chemin_image, caption='Bienvenue', use_column_width=True)


st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background: url('/Users/apple/Desktop/Dossier Yahn/cic.PNG');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with st.expander("See notes"):
    st.markdown("""
* Veuillez renseigner les champs dans l'onglet à gauche 😊.
* Cette application calcul la prime de réassurance des facultatives pour le compte de la CICA-RE .

See also:
* [Visiter le site de la CICA-RE](https://cica-re.com)
""")

data = pd.read_excel('/Users/apple/Desktop/Base 10k 50k inc_rt_rd.xlsx')

df = data
df_sample = df.sample(100)
st.sidebar.header("Les paramètres d'entrée")
if st.sidebar.checkbox("Afficher les Données brutes", False):
    st.subheader("Données CICA-RE : Échantillon de 100 observations.")
    st.write(df_sample)

Libellé_Département = st.sidebar.selectbox('Quelle est la situation géographique du risque?',
                                             ['ASIE ORIENT E-ORIENT', 'CIMA AFRIQ. OCCIDENT', 'HORS CIMA AF DU NORD', 'CIMA AFRIQ. CENTRALE', 'HORS CIMA AFRIQ ANGL', 'CIMA A.C. ASSIMILES', 'HORS CIMA AUTRES A.C', 'CIMA A.O. ASSIMILES', 'AMERIQUE LATINE'])

LIBELLE_FAMILLE= st.sidebar.selectbox('Quelle est la branche du risque ?',
                                          ['Incendie & RA', 'Risq Techniques', 'Risq Divers'])

Apporteur = st.sidebar.selectbox('Canal de transmission du risque ?', ['Courtier', 'Directe'])

#Part_Acceptée = st.sidebar.number_input('Quelle est votre part du risque', 0.0, 100.0)

Capitaux_Réassurance_CV = st.sidebar.number_input('Quelle est le montant des Capitaux ?', 0.0, 314121440150.961)

Taux_commission = st.sidebar.number_input('Quelle est le taux de Commission ?', 0.0, 9029131.344)

Part = st.sidebar.number_input('Quelle part de risque souhaitez vous conserver ?', 0.0, 100.0)

user_input = pd.DataFrame({
            'Libellé_Département': [Libellé_Département],
            'LIBELLE_FAMILLE': [LIBELLE_FAMILLE],
            'Apporteur': [Apporteur],
            'Capitaux_Réassurance_CV': [Capitaux_Réassurance_CV],
            'Taux_commission': [Taux_commission]
        })
#'Part_Acceptée': [Part_Acceptée],

# ENCODAGE DES DONNEES DEPARTEMENT ENTREE PAR L'UTILISATEUR ca été périeux hahaha
if Libellé_Département == "ASIE ORIENT E-ORIENT":
    Libellé_Département_e = 1

elif Libellé_Département == "CIMA AFRIQ. OCCIDENT":
    Libellé_Département_e = 5

elif Libellé_Département == "HORS CIMA AF DU NORD":
    Libellé_Département_e = 6

elif Libellé_Département == "CIMA AFRIQ. CENTRALE":
    Libellé_Département_e = 4

elif Libellé_Département == "HORS CIMA AFRIQ ANGL":
    Libellé_Département_e = 7

elif Libellé_Département == "CIMA A.C. ASSIMILES":
    Libellé_Département_e = 2

elif Libellé_Département == "HORS CIMA AUTRES A.C":
    Libellé_Département_e = 8

elif Libellé_Département == "CIMA A.O. ASSIMILES":
    Libellé_Département_e = 3

elif Libellé_Département == "AMERIQUE LATINE":
    Libellé_Département_e = 0

# ENCODAGE DE LA BRANCHE RENSEIGNEZ
if LIBELLE_FAMILLE == "Incendie & RA":
    LIBELLE_FAMILLE_e = 0

elif LIBELLE_FAMILLE == "Risq Techniques":
    LIBELLE_FAMILLE_e = 1

elif LIBELLE_FAMILLE == "Risq Divers":
    LIBELLE_FAMILLE_e = 2

# ENCODAGE DE L'APPORTEUR RENSEIGNEZ
if Apporteur == "Courtier":
    Apporteur_e = 0

elif Apporteur == "Directe":
    Apporteur_e = 1


user_input2 = pd.DataFrame({
            'LIBELLE_FAMILLE': [LIBELLE_FAMILLE_e],
            'Libellé_Département': [Libellé_Département_e],
            'Apporteur': [Apporteur_e],
            'Capitaux_Réassurance_CV': [Capitaux_Réassurance_CV],
            'Taux_commission': [Taux_commission]
        })
#'Part_Acceptée': [Part_Acceptée],
#engagement = Capitaux_Réassurance_CV * Part_Acceptée
new_data = np.array((LIBELLE_FAMILLE_e, Libellé_Département_e, Apporteur_e, Capitaux_Réassurance_CV, Taux_commission)) #Part_Acceptée

# IMPORTATION BASE DE DONNEES PRE-PROCESSED
df = pd.read_excel('/Users/apple/Desktop/data_preprocessed0.xlsx')
X = df[['LIBELLE_FAMILLE','Libellé_Département','Apporteur', 'Capitaux_Réassurance_CV', 'Taux_commission']]#'Part_Acceptée'
y = df['Taux_prime']

# Découpage Trainset/Testset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

model = XGBRegressor(random_state=0, objective = 'reg:squarederror')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

if st.sidebar.button("Executer"):
    prediction = model.predict(user_input2)

    st.markdown("***LES RESULTATS DE LA PREDICTIONS SONT LISTÉS COMME SUIT:***")
    st.write("Avec une précision de :", r2)
    st.write("Le taux de Prime applicable aux capitaux est:", prediction)

    Premium = Capitaux_Réassurance_CV * prediction
    st.write("La Prime à 100% s'élève à :", Premium)

    engagement = (Part * Premium)/100
    st.write("Si vous souhaitez vous engagez sur cette affaire, la Prime serait de :", engagement)

    #st.write("Pour un engagement de :", engagement)


    lottie_coding3 = load_lottiefile("/Users/apple/Desktop/Dossier Yahn/coding3.json")

    st_lottie(
        lottie_coding3,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
    )



#st.markdown(" CE PROJET S'INSCRIT DANS LA VISION DE LA CICA-RE ")
#st.markdown('<span style="color: 33E6FF;">CE PROJET S INSCRIT DANS LA VISION DE LA CICA-RE</span>', unsafe_allow_html=True)


st.markdown("***CE PROJET S'INSCRIT DANS***")
chemin_image2 = '/Users/apple/Desktop/Dossier Yahn/vision.PNG'
st.image(chemin_image2, use_column_width=True)

#st.camera_input("Take a photo")

#year_option = data['EXERCICE REASSURANCE'].unique().tolist()
#ANNEE_REASSURANCE = st.selectbox('Quelle année voulez vous voir ?', year_option, 0)
#data=data[data['EXERCICE REASSURANCE']==ANNEE_REASSURANCE]

#fig = px.scatter(data, x="Libellé_Département", y="Aliment_notre_part", color="LIBELLE REGION", hover_name="LIBELLE REGION",
                # log_x=True, size_max=55, )
#fig.update_layout(width=800)
#st.write(fig)


with st.expander("**REMERCIEMENTS**"):
    chemin_image2 = '/Users/apple/Desktop/remer.PNG'
    st.image(chemin_image2, use_column_width=True)


lottie_coding2 = load_lottiefile("/Users/apple/Desktop/ML.json")


st_lottie(
    lottie_coding2,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
)