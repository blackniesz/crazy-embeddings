import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
import requests
from io import BytesIO

# ================================
# 🔥 CRAZY EMBEDDINGS - AUTO LOADING
# ================================

st.set_page_config(
    page_title="🔥 Crazy Embeddings",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Kompaktowy design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #ff4b4b;
        margin-bottom: 1rem;
    }
    .compact-metric {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .result-header {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        font-weight: bold;
        color: #1f1f1f;
    }
    .zabieg-header {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
        color: white;
    }
    .info-header {
        background: linear-gradient(90deg, #4dabf7, #339af0);
        color: white;
    }
    .result-item {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 0.7rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .similarity-score {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        float: right;
    }
    .url-link {
        color: #0366d6;
        text-decoration: none;
        font-size: 0.85rem;
        word-break: break-all;
    }
    .title-text {
        font-weight: 500;
        color: #24292e;
        margin-top: 0.3rem;
        font-size: 0.9rem;
        line-height: 1.2;
    }
    .stTextArea textarea {
        height: 120px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# KONFIGURACJA I ŁADOWANIE DANYCH
# ================================

# URL do pliku PKL na GitHub (zastąp swoim)
GITHUB_PKL_URL = "https://github.com/blackniesz/crazy-embeddings/blob/main/embeddings_database.pkl"

@st.cache_data(ttl=3600)  # Cache na 1 godzinę
def load_embeddings_from_github(url):
    """Automatyczne ładowanie embeddingów z GitHub"""
    try:
        with st.spinner("📥 Ładuję embeddingi z GitHub..."):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Wczytaj pickle z bytes
            df_embeddings = pickle.load(BytesIO(response.content))
            embeddings_matrix = np.array(df_embeddings['embedding'].tolist())
            
            return df_embeddings, embeddings_matrix
    except Exception as e:
        st.error(f"❌ Nie można załadować embeddingów: {e}")
        st.info("💡 Sprawdź czy URL do pliku PKL na GitHub jest poprawny")
        return None, None

def get_api_key():
    """Pobiera klucz API z secrets lub input"""
    try:
        # Najpierw spróbuj z secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        return api_key
    except:
        # Fallback - input od użytkownika
        with st.sidebar:
            st.warning("⚠️ Brak klucza API w secrets")
            api_key = st.text_input("🔑 Gemini API Key", type="password")
            if api_key:
                return api_key
        return None

def configure_gemini_api(api_key):
    """Konfiguruje API Gemini"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Błąd konfiguracji API: {e}")
        return False

def generate_query_embedding(text):
    """Generuje embedding dla tekstu"""
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-exp-03-07",
            content=[text],
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=3072
        )
        return np.array(result['embedding'][0])
    except Exception as e:
        st.error(f"Błąd API: {e}")
        return None

def extract_title_from_chunk(chunk_text):
    """Wyciąga tytuł z fragmentu tekstu"""
    if pd.isna(chunk_text):
        return "Brak tytułu"
    
    text = str(chunk_text)
    # Weź pierwsze 80 znaków jako tytuł
    title = text[:80].strip()
    if len(text) > 80:
        title += "..."
    return title

def find_similar_articles_balanced(query_text, df_embeddings, embeddings_matrix, 
                                 similarity_threshold=0.75, total_results=10):
    """Równomierne wyszukiwanie podobieństw"""
    
    # Generuj embedding
    with st.spinner("🔄 Analizuję tekst..."):
        query_embedding = generate_query_embedding(query_text)
        if query_embedding is None:
            return None
    
    # Oblicz podobieństwa
    with st.spinner("🔄 Szukam podobieństw..."):
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
        df_results = df_embeddings.copy()
        df_results['similarity'] = similarities
    
    # Kategoryzacja
    df_results['category'] = df_results['source_address'].apply(lambda url: 
        'zabieg' if '/zabiegi/' in url else 'informacyjny' if '/klinikaodadoz/' in url else 'inny'
    )
    
    # Dodaj tytuły
    df_results['title'] = df_results['chunk_text'].apply(extract_title_from_chunk)
    
    # Filtracja
    df_filtered = df_results[df_results['similarity'] > similarity_threshold]
    
    if len(df_filtered) == 0:
        st.warning(f"⚠️ Brak wyników > {similarity_threshold}, pokazuję najlepsze")
        top_results = df_results.nlargest(total_results, 'similarity')
        top_results['category'] = top_results['source_address'].apply(lambda url: 
            'zabieg' if '/zabiegi/' in url else 'informacyjny' if '/klinikaodadoz/' in url else 'inny'
        )
        top_results['title'] = top_results['chunk_text'].apply(extract_title_from_chunk)
        return top_results
    
    # Równomierny podział
    zabieg_results = df_filtered[df_filtered['category'] == 'zabieg'].sort_values('similarity', ascending=False)
    info_results = df_filtered[df_filtered['category'] == 'informacyjny'].sort_values('similarity', ascending=False)
    
    half = total_results // 2
    top_zabiegi = zabieg_results.head(half)
    top_info = info_results.head(half)
    
    # Połącz i uzupełnij
    balanced = pd.concat([top_zabiegi, top_info])
    
    if len(balanced) < total_results:
        remaining = total_results - len(balanced)
        if len(zabieg_results) > len(top_zabiegi):
            extra = zabieg_results.iloc[len(top_zabiegi):len(top_zabiegi)+remaining]
            balanced = pd.concat([balanced, extra])
        elif len(info_results) > len(top_info):
            extra = info_results.iloc[len(top_info):len(top_info)+remaining]
            balanced = pd.concat([balanced, extra])
    
    return balanced.sort_values('similarity', ascending=False).head(total_results)

def display_results_compact(results):
    """Kompaktowe wyświetlanie wyników w dwóch kolumnach"""
    if results is None or len(results) == 0:
        st.error("😔 Brak wyników")
        return
    
    # Podziel wyniki na kategorie
    zabiegi = results[results['category'] == 'zabieg']
    informacyjne = results[results['category'] == 'informacyjny']
    
    # Statystyki górne
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="compact-metric">
            💉 Zabiegi: {len(zabiegi)}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="compact-metric">
            📚 Klinika od A do Z: {len(informacyjne)}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_sim = results['similarity'].mean()
        st.markdown(f"""
        <div class="compact-metric">
            📊 Śr. podobieństwo: {avg_sim:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # Wyniki w dwóch kolumnach
    col_zabiegi, col_info = st.columns(2)
    
    # KOLUMNA 1: ZABIEGI
    with col_zabiegi:
        st.markdown("""
        <div class="result-header zabieg-header">
            💉 ZABIEGI
        </div>
        """, unsafe_allow_html=True)
        
        if len(zabiegi) == 0:
            st.info("Brak zabiegów w wynikach")
        else:
            for _, row in zabiegi.iterrows():
                similarity_pct = f"{row['similarity']:.1%}"
                
                st.markdown(f"""
                <div class="result-item">
                    <div class="similarity-score">{similarity_pct}</div>
                    <a href="{row['source_address']}" target="_blank" class="url-link">
                        {row['source_address']}
                    </a>
                    <div class="title-text">{row['title']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # KOLUMNA 2: KLINIKA OD A DO Z
    with col_info:
        st.markdown("""
        <div class="result-header info-header">
            📚 KLINIKA OD A DO Z
        </div>
        """, unsafe_allow_html=True)
        
        if len(informacyjne) == 0:
            st.info("Brak artykułów informacyjnych w wynikach")
        else:
            for _, row in informacyjne.iterrows():
                similarity_pct = f"{row['similarity']:.1%}"
                
                st.markdown(f"""
                <div class="result-item">
                    <div class="similarity-score">{similarity_pct}</div>
                    <a href="{row['source_address']}" target="_blank" class="url-link">
                        {row['source_address']}
                    </a>
                    <div class="title-text">{row['title']}</div>
                </div>
                """, unsafe_allow_html=True)

# ================================
# GŁÓWNA APLIKACJA
# ================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 Crazy Embeddings</h1>', unsafe_allow_html=True)
    st.markdown("### ⚖️ Równomierne wyszukiwanie podobieństw - Zabiegi vs Klinika od A do Z")
    
    # Automatyczne ładowanie danych
    if 'embeddings_loaded' not in st.session_state:
        # ZMIEŃ TEN URL NA SWÓJ GITHUB REPO!
        df_embeddings, embeddings_matrix = load_embeddings_from_github(GITHUB_PKL_URL)
        
        if df_embeddings is not None:
            st.session_state.df_embeddings = df_embeddings
            st.session_state.embeddings_matrix = embeddings_matrix
            st.session_state.embeddings_loaded = True
            st.success(f"✅ Automatycznie załadowano {len(df_embeddings)} embeddingów z GitHub!")
        else:
            st.error("❌ Nie można załadować embeddingów")
            st.stop()
    
    # Pobierz dane z session state
    df_embeddings = st.session_state.df_embeddings
    embeddings_matrix = st.session_state.embeddings_matrix
    
    # Konfiguracja API
    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ Wprowadź klucz API Gemini")
        st.stop()
    
    if not configure_gemini_api(api_key):
        st.error("❌ Błąd konfiguracji API")
        st.stop()
    
    # Parametry w sidebar (zwinięty domyślnie)
    with st.sidebar:
        st.header("🎛️ Parametry")
        similarity_threshold = st.slider(
            "📏 Próg podobieństwa", 
            min_value=0.5, max_value=0.9, value=0.75, step=0.05
        )
        total_results = st.selectbox(
            "🔢 Liczba wyników", 
            options=[6, 8, 10, 12, 16, 20], 
            index=2
        )
        
        st.markdown("---")
        
        # Statystyki bazy
        zabiegi_count = len(df_embeddings[df_embeddings['source_address'].str.contains('/zabiegi/', na=False)])
        info_count = len(df_embeddings[df_embeddings['source_address'].str.contains('/klinikaodadoz/', na=False)])
        
        st.markdown(f"""
        **📊 Statystyki bazy:**
        - 💉 Zabiegi: {zabiegi_count}
        - 📚 Klinika A-Z: {info_count}
        - 📐 Wymiar: {embeddings_matrix.shape[1]}
        """)
    
    # Formularz wyszukiwania
    st.markdown("### 🔍 Wklej tekst do analizy:")
    
    with st.form("search_form", clear_on_submit=False):
        query_text = st.text_area(
            "",
            height=120,
            placeholder="Wklej tutaj artykuł, opis problemu lub pytanie o zabiegi...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_button = st.form_submit_button("🚀 Szukaj podobieństw", use_container_width=True)
        with col2:
            if query_text:
                st.metric("📊 Znaków", len(query_text))
        with col3:
            if query_text:
                words = len(query_text.split())
                st.metric("📝 Słów", words)
    
    # Wyszukiwanie
    if search_button and query_text.strip():
        if len(query_text.strip()) < 10:
            st.error("⚠️ Tekst za krótki (min. 10 znaków)")
            return
        
        start_time = time.time()
        
        results = find_similar_articles_balanced(
            query_text, 
            df_embeddings, 
            embeddings_matrix,
            similarity_threshold, 
            total_results
        )
        
        search_time = time.time() - start_time
        
        if results is not None and len(results) > 0:
            st.success(f"✅ Znaleziono {len(results)} wyników w {search_time:.1f}s")
            
            # Kompaktowe wyświetlenie wyników
            display_results_compact(results)
            
            # Eksport
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Pobierz wyniki CSV"):
                    export_df = results[['category', 'similarity', 'source_address', 'title']].copy()
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Zapisz plik",
                        data=csv_data,
                        file_name=f"crazy_embeddings_{int(time.time())}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🔄 Nowe wyszukiwanie"):
                    st.rerun()
        else:
            st.error("😔 Brak wyników - spróbuj obniżyć próg podobieństwa")
    
    elif search_button:
        st.error("⚠️ Wprowadź tekst do wyszukiwania")

# ================================
# URUCHOMIENIE
# ================================

if __name__ == "__main__":
    main()

# ================================
# INSTRUKCJE WDROŻENIA
# ================================

"""
INSTRUKCJE WDROŻENIA Z GITHUB:

1. PRZYGOTUJ REPO:
   - Wgraj plik embeddings_database.pkl do swojego GitHub repo
   - Skopiuj URL do raw file (kliknij plik -> Raw -> skopiuj URL)

2. ZMIEŃ URL W KODZIE:
   - Znajdź linię: GITHUB_PKL_URL = "https://github.com/YOUR_USERNAME..."
   - Zastąp swoim URL do pliku PKL

3. STREAMLIT SECRETS:
   - W Streamlit Cloud: Settings -> Secrets
   - Dodaj: GEMINI_API_KEY = "twój_klucz_api"

4. REQUIREMENTS.TXT:
   streamlit
   pandas
   numpy
   scikit-learn
   google-generativeai
   requests

5. DEPLOY:
   - Wgraj app.py + requirements.txt do GitHub
   - Deploy przez Streamlit Cloud
   - Gotowe!

ZALETY TEGO ROZWIĄZANIA:
✅ Automatyczne ładowanie embeddingów z GitHub
✅ Bezpieczny API key w secrets
✅ Kompaktowy interfejs 2-kolumnowy
✅ Szybkie ładowanie z cache
✅ "Klinika od A do Z" zamiast "Informacyjne"
✅ Responsywny design
✅ Minimalne scrollowanie
"""
