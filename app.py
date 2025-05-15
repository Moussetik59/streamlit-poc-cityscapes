import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image

# Configuration du dashboard
st.set_page_config(page_title="Preuve de Concept – SegFormer", layout="wide")

# ─────────────────────────────
# ONGLET 1 : ANALYSE EXPLORATOIRE
# ─────────────────────────────
def eda_tab():
    st.title("Exploration du Dataset Cityscapes")
    st.markdown("""
    Le dataset Cityscapes est composé d'images urbaines annotées pixel à pixel pour la segmentation sémantique.
    Voici quelques exemples extraits du sous-dossier **val** :
    """)

    img_folder = "data"
    sample_imgs = [img for img in os.listdir(img_folder) if img.endswith(".png") and "mask" not in img]

    cols = st.columns(3)
    for i, img_file in enumerate(sample_imgs[:3]):
        img = Image.open(os.path.join(img_folder, img_file))
        cols[i].image(img, caption=img_file, use_container_width=True)

    st.markdown("""
    Les masques d’annotations contiennent 8 classes principales : route, bâtiment, objet, nature, ciel, piéton, véhicule, fond.
    """)

# ─────────────────────────────
# ONGLET 2 : SÉLECTION D’UNE IMAGE
# ─────────────────────────────
def selection_tab():
    st.title("Image sélectionnée pour comparaison")

    st.markdown("""
    L’image ci-dessous est issue du dataset **Cityscapes (val set)**.  
    Elle est utilisée comme **base commune** pour visualiser les masques prédits par les modèles du projet 8 (CNN) et du projet 9 (SegFormer).
    """)

    img_path = "data/frankfurt_000000_000294_leftImg8bit.png"
    img = Image.open(img_path)
    st.image(img, caption="Image d’origine sélectionnée", use_container_width=True)

# ─────────────────────────────
# ONGLET 3 : COMPARAISON DES PRÉDICTIONS
# ─────────────────────────────
def results_tab():
    st.title("Comparaison des prédictions – Projet 8 vs Projet 9")

    st.markdown("Masques générés pour l’image `frankfurt_000000_000294_leftImg8bit.png` :")

    st.subheader("Projet 8 – CNN")
    st.image("data/p8_mask_0.png", caption="Masques prédits par les modèles CNN (U-Net, VGG16, ResNet)", use_container_width=True)

    st.subheader("Projet 9 – SegFormer")
    st.image("data/p9_mask_0.png", caption="Masques prédits par SegFormer (MiT-B0, B2, préentraîné)", use_container_width=True)

    st.markdown("""
    Ces visualisations permettent de comparer les architectures classiques CNN (projet 8) avec les modèles basés sur Transformers (projet 9).  
    SegFormer fournit généralement des contours plus précis et une segmentation plus fine, notamment sur les **véhicules** et **piétons**.
    """)


# ─────────────────────────────
# ONGLET ACCESSIBILITÉ (BONUS)
# ─────────────────────────────
def show_accessibility_info():
    st.markdown("---")
    st.markdown("**Accessibilité** – conformité WCAG essentielle :")
    st.markdown("""
    - Couleurs contrastées pour les masques de segmentation  
    - Titres et textes explicites pour chaque section  
    - Navigation claire par onglets
    """)

# ─────────────────────────────
# STRUCTURE EN ONGLETS
# ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🖼 Image", "🧠 Prédictions"])

with tab1:
    eda_tab()
    show_accessibility_info()

with tab2:
    selection_tab()
    show_accessibility_info()

with tab3:
    results_tab()
    show_accessibility_info()
