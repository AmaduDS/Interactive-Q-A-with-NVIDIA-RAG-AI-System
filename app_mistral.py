import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Charger les variables d'environnement depuis le fichier .env
#from dotenv import load_dotenv
#load_dotenv()

# Récupérer la clé API NVIDIA pour Mistral
#mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Remplacez l'accès aux clés API par st.secrets
mistral_api_key = st.secrets["MISTRAL_API_KEY"]

# Interface utilisateur
st.sidebar.title("Paramètres")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
top_p = st.sidebar.slider("Top_p", 0.0, 1.0, 1.0)

# Fonction pour créer les embeddings pour le guide en français
def vector_embedding_mistral():
    if "vectors_fr" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(api_key=mistral_api_key)
        st.session_state.loader = PyPDFDirectoryLoader("./doc_fr")  # Charger le document en français
        st.session_state.docs = st.session_state.loader.load()

        if len(st.session_state.docs) == 0:
            st.error("Erreur : Aucun document n'a été chargé. Vérifiez le chemin du fichier.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)

            if len(st.session_state.final_documents) == 0:
                st.error("Erreur : Impossible de diviser les documents en chunks.")
            else:
                try:
                    st.session_state.vectors_fr = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                    st.sidebar.success("FAISS Vector Store DB est prête avec les embeddings NVIDIA (Français).")
                except IndexError as e:
                    st.error(f"Erreur lors de la création des vecteurs : {e}")


# Bouton pour faire l'embedding dans la barre latérale
if st.sidebar.button("Documents Embedding/Représentations vectorielles de documents (Français)"):
    vector_embedding_mistral()

# Titre de l'application
# Ajouter du CSS personnalisé pour ajuster la taille du titre
st.markdown("""
    <style>
    .custom-title {
        font-size: 24px;  /* Ajustez ici la taille de la police à votre convenance */
        font-weight: bold;
        color: #333333;  /* Couleur de la police */
        margin-bottom: 20px;  /* Espacement en dessous du titre */
    }
    </style>
    <h1 class="custom-title">📚 Guide de préparation à l'examen de citoyenneté canadienne : Q&R interactif avec le système RAG de NVIDIA AI</h1>
""", unsafe_allow_html=True)


# Modèle NVIDIA pour le français en utilisant le `base_url`
try:
    llm = ChatNVIDIA(
        model="mistralai/mistral-7b-instruct-v0.2",
        api_key=mistral_api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    st.write("Modèle : Mistral 7B-Instruct v0.2 activé pour le français.")
except ValueError as e:
    st.error(f"Erreur lors de l'initialisation du modèle : {e}")

# Créer un prompt template pour le français
prompt = ChatPromptTemplate.from_template(
    """
    Répondez uniquement en fonction du contexte fourni ci-dessous.
    La réponse doit être factuelle, concise, et en français uniquement.
    Si le contexte ne contient pas suffisamment d'informations, dites simplement "Le contexte ne contient pas assez d'informations pour répondre à cette question.
    <contexte>
    {context}
    <contexte>
    Questions : {input}
    """
)

# Entrée de l'utilisateur pour la question
prompt1 = st.text_input("Posez votre question (en français)")

# Si une question est posée
if prompt1:
    if "vectors_fr" not in st.session_state:
        st.error("Veuillez d'abord créer des embeddings pour les documents avec le bouton 'Documents Embedding'.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors_fr.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Temps de réponse : {time.process_time() - start}")
        st.write(response['answer'])

        # Afficher les documents similaires
        with st.expander("Recherche de similarité des documents"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Document {i + 1} : {doc.page_content[:500]}...")
                st.write("--------------------------------")
