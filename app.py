import streamlit as st
import pandas as pd
from PIL import Image
import os
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import folium
from streamlit_folium import folium_static
from transformers import pipeline
import torch

st.set_page_config(page_title="BioScout Islamabad", layout="wide")

# --- Configuration & Setup ---
OBSERVATIONS_FILE = 'observations.csv'
KB_DOCS_DIR = 'kb_docs/'
UPLOADED_IMAGES_DIR = 'uploaded_images/'
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
os.makedirs(KB_DOCS_DIR, exist_ok=True)

# Initialize models
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_image_classifier():
    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)
        return classifier
    except Exception as e:
        st.warning(f"Image classifier not loaded: {e}. Using mock identification.")
        return None

sentence_model = load_sentence_model()
image_classifier = load_image_classifier()

# --- Helper Functions ---
def load_observations():
    if os.path.exists(OBSERVATIONS_FILE):
        return pd.read_csv(OBSERVATIONS_FILE)
    return pd.DataFrame(columns=['observation_id', 'user_id', 'species_name', 'common_name', 
                               'date_observed', 'location_text', 'latitude', 'longitude', 
                               'image_filename', 'notes'])

def save_observation(data):
    df = load_observations()
    new_id = df['observation_id'].max() + 1 if not df.empty else 1
    data['observation_id'] = new_id
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(OBSERVATIONS_FILE, index=False)

def get_ai_species_id(image_path):
    if image_classifier is None:
        return [
            {"score": 0.75, "label": "Corvus splendens (House Crow)"},
            {"score": 0.60, "label": "Psittacula krameri (Rose-ringed Parakeet)"},
            {"score": 0.50, "label": "Ficus religiosa (Peepal Tree)"}
        ]

    try:
        img = Image.open(image_path)
        predictions = image_classifier(img)
        return [{"score": round(p['score'], 2), "label": p['label']} for p in predictions[:3]]
    except Exception as e:
        st.error(f"Error during species identification: {e}")
        return [{"score": 0.0, "label": "Error in identification"}]

def load_kb_documents():
    docs = []
    for filename in os.listdir(KB_DOCS_DIR):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(KB_DOCS_DIR, filename), 'r', encoding='utf-8') as f:
                docs.append({"text": f.read(), "source": filename})
    return docs

def get_observation_texts():
    df = load_observations()
    return [{"text": f"Observation: {row['common_name']} ({row['species_name']}) seen at {row['location_text']} on {row['date_observed']}. Notes: {row['notes']}", 
             "source": f"Observation ID {row['observation_id']}"} for _, row in df.iterrows()]

# --- RAG System ---
@st.cache_data
def get_all_documents_and_embeddings():
    kb_docs = load_kb_documents()
    obs_docs = get_observation_texts()
    all_docs = kb_docs + obs_docs
    
    if not all_docs:
        return [], np.array([])

    doc_texts = [doc['text'] for doc in all_docs]
    try:
        doc_embeddings = sentence_model.encode(doc_texts)
        return all_docs, doc_embeddings
    except Exception as e:
        st.error(f"Could not generate embeddings: {e}")
        return [], np.array([])

def retrieve_relevant_context(query, top_k=3):
    all_docs, doc_embeddings = get_all_documents_and_embeddings()
    if not all_docs or doc_embeddings.size == 0:
        return "No knowledge base available to answer."

    query_embedding = sentence_model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-min(top_k, len(similarities)):][::-1]
    
    context = ""
    for i in top_k_indices:
        context += f"Source: {all_docs[i]['source']}\nContent: {all_docs[i]['text']}\n\n"
    return context

def generate_llm_response(query, context):
    if "No knowledge base available" in context:
        return "I'm sorry, I don't have enough information in my knowledge base to answer that."

    response = f"Based on the available information:\n\n{context}\n"
    response += f"Regarding your question '{query}':\n"

    # Basic keyword matching for simulated responses
    if "leopard" in query.lower() and "margalla" in query.lower():
        response += "Common Leopards are known to inhabit Margalla Hills. Sightings are rare, but pugmarks are sometimes found, indicating their presence."
    elif "birds" in query.lower() and ("rawal lake" in query.lower() or "shakarparian" in query.lower()):
        response += "Rawal Lake and Shakarparian are excellent birdwatching spots. Common birds include parakeets, kites, and various water birds."
    elif "invasive species" in query.lower():
        response += "Invasive species like Lantana camara are a concern in the Islamabad region. They can outcompete native flora and affect local ecosystems."
    else:
        response += "The retrieved context above contains information relevant to your query. Please review it for specific details."
    
    return response

# --- Streamlit App UI ---
st.title("🌿 BioScout Islamabad 🦉")
st.caption("AI for Community Biodiversity & Sustainable Insights")

# Sidebar for user stats
with st.sidebar:
    st.header("Your BioScout Stats")
    df = load_observations()
    if not df.empty:
        user_obs = df['user_id'].value_counts()
        st.write("Top Observers:")
        for user, count in user_obs.head(3).items():
            st.write(f"👤 {user}: {count} observations")
    else:
        st.write("No observations yet. Be the first to contribute!")

tab1, tab2, tab3 = st.tabs(["Submit Observation", "View Observations", "Ask BioScout AI"])

# --- TAB 1: Submit Observation ---
with tab1:
    st.header("Submit a Biodiversity Observation")
    with st.form("observation_form", clear_on_submit=True):
        user_id = st.text_input("Your User ID/Name", value="anonymous_hiker")
        species_name = st.text_input("Species Name (Scientific, e.g., Panthera pardus)")
        common_name = st.text_input("Common Name (e.g., Common Leopard)")
        date_observed = st.date_input("Date of Observation", datetime.date.today())
        location_text = st.text_input("Location", placeholder="e.g., Trail 5, Margalla Hills")
        
        col_lat, col_lon = st.columns(2)
        latitude = col_lat.number_input("Latitude", format="%.4f", value=None, placeholder="33.7480")
        longitude = col_lon.number_input("Longitude", format="%.4f", value=None, placeholder="73.0600")

        uploaded_image = st.file_uploader("Upload Image of Species", type=["jpg", "png", "jpeg"])
        notes = st.text_area("Additional Notes (behavior, habitat, etc.)")
        
        submitted = st.form_submit_button("Submit Observation")

    if submitted:
        if not species_name and not common_name:
            st.error("Please provide at least a common or scientific name.")
        elif not location_text:
            st.error("Please provide the location of the observation.")
        else:
            image_filename = None
            if uploaded_image is not None:
                image_filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_image.name}"
                image_path = os.path.join(UPLOADED_IMAGES_DIR, image_filename)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                st.image(uploaded_image, caption="Uploaded Image", width=200)
                
                with st.spinner("AI analyzing image..."):
                    suggestions = get_ai_species_id(image_path)
                if suggestions:
                    st.subheader("AI Species Suggestions:")
                    for s in suggestions:
                        st.write(f"- {s['label']} (Confidence: {s['score']})")
                    st.info("Please verify AI suggestions or input known species name.")
            
            observation_data = {
                'user_id': user_id,
                'species_name': species_name,
                'common_name': common_name,
                'date_observed': date_observed.strftime('%Y-%m-%d'),
                'location_text': location_text,
                'latitude': latitude if latitude is not None else pd.NA,
                'longitude': longitude if longitude is not None else pd.NA,
                'image_filename': image_filename,
                'notes': notes
            }
            
            save_observation(observation_data)
            st.success("Observation submitted successfully!")

# --- TAB 2: View Observations ---
with tab2:
    st.header("View Biodiversity Observations")
    
    df = load_observations()
    if not df.empty:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            species_filter = st.text_input("Filter by species name")
        with col2:
            location_filter = st.text_input("Filter by location")
        
        # Apply filters
        if species_filter:
            df = df[df['species_name'].str.contains(species_filter, case=False, na=False) | 
                   df['common_name'].str.contains(species_filter, case=False, na=False)]
        if location_filter:
            df = df[df['location_text'].str.contains(location_filter, case=False, na=False)]
        
        # Display observations
        for _, row in df.iterrows():
            with st.expander(f"{row['common_name']} ({row['species_name']}) - {row['date_observed']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    if pd.notna(row['image_filename']):
                        image_path = os.path.join(UPLOADED_IMAGES_DIR, row['image_filename'])
                        if os.path.exists(image_path):
                            st.image(image_path, width=200)
                with col2:
                    st.write(f"**Location:** {row['location_text']}")
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        st.write(f"**Coordinates:** {row['latitude']}, {row['longitude']}")
                    st.write(f"**Notes:** {row['notes']}")
                    st.write(f"**Observer:** {row['user_id']}")
        
        # Map view
        st.subheader("Observation Map")
        if not df[['latitude', 'longitude']].isna().all().all():
            m = folium.Map(location=[33.6844, 73.0479], zoom_start=11)  # Islamabad center
            for _, row in df.dropna(subset=['latitude', 'longitude']).iterrows():
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=f"{row['common_name']} ({row['species_name']})<br>{row['date_observed']}",
                    tooltip=row['common_name']
                ).add_to(m)
            folium_static(m)
    else:
        st.info("No observations submitted yet. Be the first to contribute!")

# --- TAB 3: Ask BioScout AI ---
with tab3:
    st.header("Ask BioScout AI")
    st.write("Ask questions about biodiversity in Islamabad and surrounding areas.")
    
    query = st.text_input("Your question", placeholder="e.g., What birds are common in Margalla Hills?")
    
    if query:
        with st.spinner("Searching knowledge base..."):
            context = retrieve_relevant_context(query)
            response = generate_llm_response(query, context)
            
            st.subheader("Answer")
            st.write(response)
            
            with st.expander("View Retrieved Context"):
                st.write(context) 