import streamlit as st


def run():
    st.set_page_config(
        page_title="Audio Cockpit Analyzer",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.write("# Welcome to the Audio Cockpit Analyzer !")
    st.write(
        """L'Audio Cockpit Analyzer est un des modules du projet "Siilent Sky Connect" (anciennement "IHM Cockpit et Handicap")
        dont l'objectif est de permettre aux pilotes présentant des déficiences auditives ou d'élocution, ou HSI pour "Hearing or Speaking Impaired".
        En effet, les communications dans l’aviation actuelle se font principalement par radio, format qui n'est pas adapté aux pilotes HSI,
        ce qui restreint leur autonomie et leur accès à l'aviation."""
    )

    cols = st.columns(3)
    with cols[1]:
        st.image("static/pilote.png")

    st.write(
        """Pour répondre à cette problématique, le projet "Siilent Sky Connect" a développé différents modules interagissant les uns avec les
        autres pour permettre aux pilotes HSI de recevoir et d'envoyer des messages de manière visuelle et haptique, leur offrant ainsi la
        possibilité de piloter sans accompagnateur. Ces différents modules incluent :
        """
    )

    st.markdown("""
            -	Une tablette pilote, qui permet au pilote de recevoir des messages visuels (sous forme de texte et pictogrammes) plutôt qu'auditifs, d'envoyer des messages à la tour de contrôle, et de consulter les informations de l’ATIS , … ;
            -	Une veste haptique qui fournit un retour par vibration au pilote pour le notifier lors de la réception d’un message et de la criticité de ce dernier ;
            -	Un service de reconnaissance vocale spécifique à l’ATIS ;
            -	Un service de reconnaissance vocale spécifique à l’ATC ;
            -	Un HUD  pour expérimenter une nouvelle méthode de présentation des informations;
            -	Un service de collecte d'informations GPS ;
            -   Une interface web qui simule les messages envoyés par l’ATC pour des démonstrations et certains tests ;
            -	Un nouveau module d'identification des bruits du cockpit.
""")

    st.write(
        """L'Audio Cockpit Analyzer est ce nouveau module d'identification des bruits du cockpit. Cette interface a pour objectif de monitorer la chaîne de traitement visible ci-dessous :
        """
    )
    st.image("static/schema_chaine_traitement_v1.png")
    st.write(""" La première étape prend donc en entrée un mélange de bruits moteur, d’alarmes et de bruit qui seront isolés après la première séparation de sources, puis d’un côté les alarmes
    seront séparées les unes des autres et leurs caractéristiques seront extraites à l’aide de traitement d’image, plus précisément de la segmentation d’image. De l’autre côté, l’extraction de la fréquence
    fondamentale des bruits moteurs permettra d’étudier leurs variations et potentiellement par la suite identifier les anomalies.  Pour les deux étapes de séparation de sources audio des modèles du toolkit
    Asteroid sont utilisés mais je vous laisse consulter mon rapport de stage (Anabel Delaporte - stage 2024) pour avoir plus de détails sur les outils utilisés, pourquoi et la chaîne de traitement de manière plus générale.  """)

    st.write(""" Plusieurs pages sont disponibles avec chacune son rôle :""")
    st.write("##### Audio Visualizer ")
    st.write(""" Cette première page ne fait pas vraiment partie de le chaîne de traitement, elle été utile pour comprendre les différentes composantes d'un audio. Elle permet concrètement de visualiser différentes métriaques pour
    un audio chargé, comme le melspectrogramme, le spectre de fréquence ... J'ai laissé cette page car elle m'a été utile pour éliminer des méthodes d'identification de sources et parce qu'elle pourrait être utile à quelqu'un plus tard,
    mais elle n'est pas vraiment centrale pour la chaîne de traitement actuelle. """)
    st.write("##### Dataset Builder ")
    st.write(
        """ Cette seconde page permet de créer les datasets nécessaires aux entraînements. Elle se découpe en plusieurs parties
          """
    )
    st.write("##### Trainer ")
    st.write("""  """)
    st.write("##### Inferencer ")
    st.write("""  """)


if __name__ == "__main__":
    run()
