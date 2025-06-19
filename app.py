import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time

# ================================
# 🔥 CRAZY EMBEDDINGS
# ================================

st.set_page_config(
    page_title="🔥 Crazy Embeddings",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ff4b4b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
    }
    .zabieg-card {
        border-left: 5px solid #ff4b4b;
    }
    .info-card {
        border-left: 5px solid #4b8bff;
    }
    .similarity-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# FUNKCJE POMOCNICZE
# ================================

@st.cache_data
def load_embeddings_data(uploaded_file):
    """Wczytuje dane z pliku PKL"""
    try:
        df_embeddings = pickle.load(uploaded_file)
        embeddings_matrix = np.array(df_embeddings['embedding'].tolist())
        return df_embeddings, embeddings_matrix
    except Exception as e:
        st.error(f"Błąd wczytywania danych: {e}")
        return None, None

def configure_gemini_api(api_key):
    """Konfiguruje API Gemini"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Błąd konfiguracji API: {e}")
        return False

def generate_query_embedding(text, api_key):
    """Generuje embedding dla tekstu"""
    try:
        with st.spinner("🔄 Generuję embedding..."):
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

def find_similar_articles_balanced(query_text, df_embeddings, embeddings_matrix, api_key, 
                                 similarity_threshold=0.75, total_results=10):
    """Równomierne wyszukiwanie podobieństw"""
    
    # Generuj embedding
    query_embedding = generate_query_embedding(query_text, api_key)
    if query_embedding is None:
        return None
    
    # Oblicz podobieństwa
    with st.spinner("🔄 Obliczam podobieństwa..."):
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
        df_results = df_embeddings.copy()
        df_results['similarity'] = similarities
    
    # Kategoryzacja
    df_results['category'] = df_results['source_address'].apply(lambda url: 
        'zabieg' if '/zabiegi/' in url else 'informacyjny' if '/klinikaodadoz/' in url else 'inny'
    )
    
    # Filtracja
    df_filtered = df_results[df_results['similarity'] > similarity_threshold]
    
    if len(df_filtered) == 0:
        st.warning(f"⚠️ Brak wyników powyżej progu {similarity_threshold}")
        # Weź najlepsze bez podziału na kategorie
        top_results = df_results.nlargest(total_results, 'similarity')
        top_results['category'] = top_results['source_address'].apply(lambda url: 
            'zabieg' if '/zabiegi/' in url else 'informacyjny' if '/klinikaodadoz/' in url else 'inny'
        )
        return top_results
    
    # Równomierny podział
    zabieg_results = df_filtered[df_filtered['category'] == 'zabieg'].sort_values('similarity', ascending=False)
    info_results = df_filtered[df_filtered['category'] == 'informacyjny'].sort_values('similarity', ascending=False)
    
    half = total_results // 2
    
    # Weź najlepsze z każdej kategorii
    top_zabiegi = zabieg_results.head(half)
    top_info = info_results.head(half)
    
    # Połącz wyniki
    balanced = pd.concat([top_zabiegi, top_info])
    
    # Uzupełnij jeśli za mało z jednej kategorii
    if len(balanced) < total_results:
        remaining = total_results - len(balanced)
        if len(zabieg_results) > len(top_zabiegi):
            extra = zabieg_results.iloc[len(top_zabiegi):len(top_zabiegi)+remaining]
            balanced = pd.concat([balanced, extra])
        elif len(info_results) > len(top_info):
            extra = info_results.iloc[len(top_info):len(top_info)+remaining]
            balanced = pd.concat([balanced, extra])
    
    return balanced.sort_values('similarity', ascending=False).head(total_results)

def display_results(results):
    """Wyświetla wyniki wyszukiwania"""
    if results is None or len(results) == 0:
        st.error("😔 Brak wyników")
        return
    
    # Statystyki
    category_counts = results['category'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        zabiegi_count = category_counts.get('zabieg', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>💉 Zabiegi</h3>
            <h2>{zabiegi_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        info_count = category_counts.get('informacyjny', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 Informacyjne</h3>
            <h2>{info_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_similarity = results['similarity'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Śr. podobieństwo</h3>
            <h2>{avg_similarity:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Wyniki
    for i, (_, row) in enumerate(results.iterrows(), 1):
        emoji = "💉" if row['category'] == 'zabieg' else "📚"
        card_class = "zabieg-card" if row['category'] == 'zabieg' else "info-card"
        
        with st.container():
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3>{emoji} WYNIK {i} - {row['category'].upper()}</h3>
                    <span class="similarity-badge">{row['similarity']:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**🌐 URL:** {row['source_address']}")
            
            if 'chunk_text' in row and pd.notna(row['chunk_text']):
                chunk_text = str(row['chunk_text'])
                preview = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
                
                with st.expander("📄 Podgląd fragmentu"):
                    st.write(preview)
                
                # Przycisk kopiowania linku
                if st.button(f"📋 Kopiuj link", key=f"copy_{i}"):
                    st.code(row['source_address'])
            
            st.markdown("---")

# ================================
# GŁÓWNA APLIKACJA
# ================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 Crazy Embeddings</h1>', unsafe_allow_html=True)
    st.markdown("### ⚖️ Inteligentne wyszukiwanie podobieństw z równomiernym podziałem")
    
    # Sidebar - Konfiguracja
    with st.sidebar:
        st.header("⚙️ Konfiguracja")
        
        # API Key
        api_key = st.text_input("🔑 Gemini API Key", type="password", 
                               help="Wprowadź swój klucz API Google Gemini")
        
        if api_key:
            if configure_gemini_api(api_key):
                st.success("✅ API skonfigurowane")
            else:
                st.error("❌ Błąd API")
        
        st.markdown("---")
        
        # Upload pliku
        st.header("📂 Dane")
        uploaded_file = st.file_uploader(
            "Wgraj plik z embeddingami (.pkl)", 
            type=['pkl'],
            help="Plik PKL wygenerowany w pierwszym Colabie"
        )
        
        if uploaded_file:
            df_embeddings, embeddings_matrix = load_embeddings_data(uploaded_file)
            
            if df_embeddings is not None:
                st.success(f"✅ Załadowano {len(df_embeddings)} fragmentów")
                
                # Statystyki bazy
                zabiegi = len(df_embeddings[df_embeddings['source_address'].str.contains('/zabiegi/', na=False)])
                info = len(df_embeddings[df_embeddings['source_address'].str.contains('/klinikaodadoz/', na=False)])
                
                st.info(f"""
                📊 **Statystyki bazy:**
                - 💉 Zabiegi: {zabiegi}
                - 📚 Informacyjne: {info}
                - 📐 Wymiar: {embeddings_matrix.shape[1]}
                """)
        
        st.markdown("---")
        
        # Parametry wyszukiwania
        st.header("🎛️ Parametry")
        similarity_threshold = st.slider(
            "📏 Próg podobieństwa", 
            min_value=0.5, max_value=0.9, value=0.75, step=0.05,
            help="Minimalny poziom podobieństwa dla wyników"
        )
        
        total_results = st.selectbox(
            "🔢 Liczba wyników", 
            options=[6, 8, 10, 12, 16, 20], 
            index=2,
            help="Łączna liczba wyników (połowa zabiegi, połowa informacyjne)"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Jak używać:")
        st.markdown("""
        1. Wprowadź klucz API Gemini
        2. Wgraj plik PKL z embeddingami  
        3. Wklej tekst do analizy
        4. Otrzymasz równomierny podział wyników
        """)
    
    # Main content
    if not api_key:
        st.warning("⚠️ Wprowadź klucz API w sidebarze")
        return
    
    if 'df_embeddings' not in locals() or df_embeddings is None:
        st.warning("⚠️ Wgraj plik z embeddingami w sidebarze")
        return
    
    # Formularz wyszukiwania
    st.header("🔍 Wyszukiwanie")
    
    with st.form("search_form"):
        query_text = st.text_area(
            "📝 Wklej tekst do analizy:",
            height=200,
            placeholder="Wklej tutaj artykuł, opis zabiegu lub pytanie...",
            help="System znajdzie podobne fragmenty z równomiernym podziałem na zabiegi i artykuły informacyjne"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.form_submit_button("🚀 Szukaj", use_container_width=True)
        with col2:
            if query_text:
                st.info(f"📊 Długość tekstu: {len(query_text)} znaków")
    
    # Wyszukiwanie
    if search_button and query_text.strip():
        if len(query_text.strip()) < 10:
            st.error("⚠️ Tekst jest za krótki (min. 10 znaków)")
            return
        
        start_time = time.time()
        
        results = find_similar_articles_balanced(
            query_text, 
            df_embeddings, 
            embeddings_matrix, 
            api_key,
            similarity_threshold, 
            total_results
        )
        
        search_time = time.time() - start_time
        
        if results is not None:
            st.success(f"✅ Wyszukiwanie zakończone w {search_time:.1f}s")
            
            # Wyświetl wyniki
            st.header("🎉 Wyniki")
            display_results(results)
            
            # Opcja eksportu
            if len(results) > 0:
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Przygotuj dane do eksportu
                    export_df = results.copy()
                    if 'embedding' in export_df.columns:
                        export_df = export_df.drop('embedding', axis=1)
                    
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Pobierz CSV",
                        data=csv_data,
                        file_name=f"crazy_embeddings_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if st.button("🔄 Nowe wyszukiwanie"):
                        st.rerun()
        else:
            st.error("❌ Błąd podczas wyszukiwania")
    
    elif search_button:
        st.error("⚠️ Wprowadź tekst do wyszukiwania")

# ================================
# URUCHOMIENIE
# ================================

if __name__ == "__main__":
    main()

# ================================
# INSTRUKCJE URUCHOMIENIA
# ================================

"""
INSTRUKCJE URUCHOMIENIA:

1. Zapisz ten kod jako: crazy_embeddings.py

2. Zainstaluj wymagane biblioteki:
   pip install streamlit pandas numpy scikit-learn google-generativeai

3. Uruchom aplikację:
   streamlit run crazy_embeddings.py

4. Otwórz w przeglądarce: http://localhost:8501

5. W sidebarze:
   - Wprowadź klucz API Gemini
   - Wgraj plik PKL z embeddingami

6. Wklej tekst i ciesz się równomiernym wyszukiwaniem! 🚀

FUNKCJE:
✅ Równomierne wyniki (50% zabiegi, 50% informacyjne)
✅ Intuicyjny interfejs
✅ Statystyki w czasie rzeczywistym  
✅ Kolorowe karty wyników
✅ Eksport do CSV
✅ Responsywny design
✅ Wskaźniki podobieństwa
✅ Podgląd fragmentów
"""
