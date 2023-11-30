import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Charger les modèles
model_english = load_model('best_model.h5')
model_french = load_model('best_model_french.h5')

# Charger les tokenizers
tokenizer_english = Tokenizer()
tokenizer_french = Tokenizer()

# Fonctions de prétraitement
def nettoyage(texte):
    texte = texte.lower()
    texte = re.sub(r'http\S+|www\S+|https\S+', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'@[\w_-]+', '', texte)
    texte = re.sub(r'#', '', texte)
    texte = re.sub(r'[^a-zA-Z\s]', '', texte)

    # Étendre les contractions
    texte = ' '.join([contractions.fix(word) for word in texte.split()])

    mots = word_tokenize(texte)
    mots = [mot for mot in mots if mot not in stopwords.words('english')]
    lemmatiseur = WordNetLemmatizer()
    mots = [lemmatiseur.lemmatize(mot) for mot in mots]
    texte_nettoye = ' '.join(mots)

    return texte_nettoye

# Interface utilisateur Streamlit
st.title("Test de l'analyse de sentiment")

# Choix du modèle
langue = st.radio("Choisissez la langue du modèle:", ["Anglais", "Français"])

# Champ de saisie de texte
texte_utilisateur = st.text_input("Entrez votre texte ici", "write your sentence here")

# Bouton de prédiction
if st.button("Prévoir le sentiment"):
    # Prétraitement du texte utilisateur
    texte_utilisateur_nettoye = nettoyage(texte_utilisateur)

    # Convertir le texte utilisateur en séquence d'indices avec le tokenizer approprié
    if langue == "Anglais":
        tokenizer_english.fit_on_texts(texte_utilisateur_nettoye)
        sequence = tokenizer_english.texts_to_sequences([texte_utilisateur_nettoye])
        max_length = 20
         # Padding de la séquence
        sequence_padded = pad_sequences(sequence, maxlen=max_length)
    else:
        tokenizer_french.fit_on_texts(texte_utilisateur_nettoye)
        sequence = tokenizer_french.texts_to_sequences([texte_utilisateur_nettoye])

        max_length = 41
        # Padding de la séquence
        sequence_padded = pad_sequences(sequence, maxlen=max_length)

    # Prédire le sentiment avec le modèle approprié
    if langue == "Anglais":
        prediction = model_english.predict(sequence_padded)[0, 0]
    else:
        try :
            prediction = model_french.predict(sequence_padded)[0, 0]
            
        except :
              prediction=0      

    # Afficher le résultat
    #st.write(f"Score de confiance: {prediction}")
    st.write(f"Sentiment prédit: {'Positif' if prediction > 0.5 else 'Négatif'}")
    #st.write(f"Confiance: {prediction * 100:.2f}%")
