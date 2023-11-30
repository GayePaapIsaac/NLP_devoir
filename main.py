import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer  # Ajoutez cette ligne
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Charger le modèle
model = load_model('best_model.h5')
model_franch = load_model('best_model.h5')

# Charger le tokenizer
tokenizer = Tokenizer()


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

# Champ de saisie de texte
texte_utilisateur = st.text_input("Entrez votre texte ici", "write your sentence here")

# Bouton de prédiction
if st.button("Prévoir le sentiment"):
    # Prétraitement du texte utilisateur
    texte_utilisateur_nettoye = nettoyage(texte_utilisateur)
    st.write("text_clean :", texte_utilisateur_nettoye)
    tokenizer.fit_on_texts(texte_utilisateur_nettoye)
    # Convertir le texte utilisateur en séquence d'indices avec le tokenizer
    sequence = tokenizer.texts_to_sequences(texte_utilisateur_nettoye)
    
    max_length=20
     # Padding de la séquence
    
    sequence_padded = pad_sequences(sequence, maxlen=max_length)


    # Prédire le sentiment
    prediction = model.predict(sequence_padded)[0, 0]

    # Afficher le résultat
    st.write(f"Score de confiance: {prediction}")
    st.write(f"Sentiment prédit: {'Positif' if prediction > 0.5 else 'Négatif'}")
    st.write(f"Confiance: {prediction * 100:.2f}%")