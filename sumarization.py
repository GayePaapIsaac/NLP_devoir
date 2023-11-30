import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import re

# Fonction pour nettoyer le texte
def cleaning(text):
    text = re.sub(r'[[\w]*]', ' ', text)
    text = re.sub(r'\xa0|\u200c', ' ', text)
    text = re.sub(r'/s+', ' ', text)
    text = re.sub(r'^\s|\s$', '', text)
    text = re.sub(r'^\n|\n$', '', text)
    return text

# Interface utilisateur Streamlit avec un thème sombre
st.markdown(
    """
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .main {
            max-width: 1200px;
            margin: auto;
        }
        .stApp {
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Text Abstractor: English & French')
texts = st.text_area("", placeholder="Entrez votre texte ici")

# Nettoyer le texte
text = cleaning(texts)

# Langue par défaut
language = 'fr'

# Boutons pour choisir la langue
if st.button('Anglais', type="primary"):
    language = 'en'

if st.button('Français', type="primary"):
    language = 'fr'

# Créer un text parser utilisant la tokenisation
parser = PlaintextParser.from_string(text, Tokenizer('english' if language == 'en' else 'french'))
stopwords = nltk.corpus.stopwords.words('french' if language == 'fr' else 'english')

# Définir la taille de la police souhaitée
font_size = "20px"

# Définir la police de caractères souhaitée
font_family = "Times New Roman, serif"

# Définir la couleur du texte souhaitée (code hexadécimal)
text_color = "white"

# Configurer les paramètres nltk
nltk.download('punkt')

# Nombre de phrases à résumer (paramètre de l'interface utilisateur)
number_of_sentences_slider = st.slider("Nombre de phrases à résumer :", min_value=1, max_value=10, value=1)

# Boutons pour chaque algorithme de résumé
if st.button('TextRank Summarizer', type='primary'):
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, number_of_sentences_slider)
    text_summary = " ".join(str(sentence) for sentence in summary)
    st.write(f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">{text_summary}</div>', unsafe_allow_html=True)

if st.button('LexRank Summarizer', type="primary"):
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, number_of_sentences_slider)
    text_summary = " ".join(str(sentence) for sentence in summary)
    st.write(f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">{text_summary}</div>', unsafe_allow_html=True)

if st.button('Lsa Summarizer', type="primary"):
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, number_of_sentences_slider)
    text_summary = " ".join(str(sentence) for sentence in summary)
    st.write(f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">{text_summary}</div>', unsafe_allow_html=True)
