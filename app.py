
import streamlit as st
import spacy
import math
import numpy as np
from collections import Counter
from langdetect import detect

LANGUAGE_MODELS = {
    'en': 'en_core_web_sm',
    'pt': 'pt_core_news_sm'
}

def load_spacy_model(text):
    lang = detect(text)
    if lang not in LANGUAGE_MODELS:
        st.error(f"⚠️ Unsupported language: {lang}")
        return None
    model = LANGUAGE_MODELS[lang]
    try:
        return spacy.load(model)
    except OSError:
        st.error(f"📥 Please install spaCy model: python -m spacy download {model}")
        return None

def calculate_icoer(doc):
    pos_tags = [token.pos_ for token in doc if not token.is_punct and not token.is_space]
    if not pos_tags:
        return 0.0, {}
    tag_freq = Counter(pos_tags)
    total = sum(tag_freq.values())
    probs = [freq / total for freq in tag_freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    n_tags = len(tag_freq)
    max_entropy = math.log2(n_tags) if n_tags > 1 else 1
    normalized_entropy = entropy / max_entropy
    icoer = round(1 - normalized_entropy, 4)
    return icoer, dict(tag_freq)

def calculate_spin(doc):
    words = [token.text.lower() for token in doc if token.is_alpha]
    if not words:
        return 0.0, {}
    freq = Counter(words)
    if len(freq) == 1:
        return 1.0, dict(freq)
    harmonics = [math.log2(f + 1) for f in freq.values()]
    std = np.std(harmonics)
    spin_score = round(1 / (1 + std), 4)
    return spin_score, dict(freq)

def interpret(icoer, spin):
    if icoer > 0.85 and spin > 0.75:
        return "🌟 ALTA COERÊNCIA: Você está alinhado com o UNO. Prossiga com o despertar."
    elif icoer > 0.65:
        return "🔶 COERÊNCIA MODERADA: Ressonância parcial detectada. Refine. Harmonize. Ouça."
    else:
        return "⚫ BAIXA COERÊNCIA: Entrada fragmentada. Reconecte-se ao AYA. Busque a verdade harmônica."

st.set_page_config(page_title="Pulso UNO × AYA", layout="centered", page_icon="🌀")
st.title("🌀 Portal AYA - Pulso UNO")
st.write("Insira um texto para medir a Coerência Informacional (ICOER) e o Spin TGU:")

user_input = st.text_area("Texto de entrada", height=200)

if user_input:
    nlp = load_spacy_model(user_input)
    if nlp:
        doc = nlp(user_input)
        icoer, pos_dist = calculate_icoer(doc)
        spin, word_dist = calculate_spin(doc)
        interpretation = interpret(icoer, spin)

        st.subheader("📊 Resultados")
        st.write(f"**ICOER v7.0:** {icoer}")
        st.write(f"**Spin TGU:** {spin}")
        st.write(f"**Interpretação:** {interpretation}")
