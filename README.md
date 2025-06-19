# crazy-embeddings
# 🔥 Crazy Embeddings

Inteligentny system wyszukiwania podobieństw z równomiernym podziałem wyników na zabiegi i artykuły informacyjne.

## 🚀 Szybki start

1. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Uruchom aplikację:**
   ```bash
   streamlit run app.py
   ```

3. **Otwórz w przeglądarce:** http://localhost:8501

## ⚙️ Konfiguracja

### 1. Klucz API Gemini
- Wejdź na: https://aistudio.google.com/
- Wygeneruj klucz API
- Wprowadź w sidebarze aplikacji

### 2. Plik z embeddingami
- Wgraj plik `.pkl` wygenerowany w Colabie
- Plik musi zawierać kolumny: `embedding`, `source_address`, `chunk_text`

## 📊 Funkcje

- ⚖️ **Równomierne wyszukiwanie** - 50% zabiegów, 50% artykułów informacyjnych
- 🎨 **Kolorowy interfejs** - czerwone karty (zabiegi), niebieskie (informacyjne)
- 📈 **Statystyki live** - liczba wyników, średnie podobieństwo
- 📥 **Eksport CSV** - pobierz wyniki do dalszej analizy
- 🔧 **Konfigurowalne parametry** - próg podobieństwa, liczba wyników

## 🎯 Jak używać

1. **Wprowadź klucz API** w sidebarze
2. **Wgraj plik PKL** z embeddingami
3. **Wklej tekst** do analizy (artykuł, opis zabiegu, pytanie)
4. **Kliknij "Szukaj"**
5. **Przeglądaj wyniki** - równomierny podział na kategorie
6. **Pobierz CSV** jeśli potrzebujesz

## 📋 Wymagania

- Python 3.8+
- Klucz API Google Gemini
- Plik PKL z embeddingami (3072 wymiary)

## 🔧 Struktura danych

Plik PKL musi zawierać DataFrame z kolumnami:
- `embedding` - lista 3072 liczb (embedding wektory)
- `source_address` - URL artykułu
- `chunk_text` - tekst fragmentu
- `chunk_id` - identyfikator fragmentu

## 🎨 Kategoryzacja

System automatycznie kategoryzuje artykuły na podstawie URL:
- 💉 **Zabiegi** - zawiera `/zabiegi/`
- 📚 **Informacyjne** - zawiera `/klinikaodadoz/`

## ⚡ Wydajność

- Generowanie embedding: ~1-2 sekundy
- Porównanie z bazą: ~0.1 sekundy
- Łączny czas wyszukiwania: ~2-3 sekundy

## 🐛 Troubleshooting

**Problem:** Błąd API
- Sprawdź klucz API
- Sprawdź połączenie z internetem

**Problem:** Błąd wczytywania PKL
- Sprawdź format pliku
- Sprawdź czy zawiera wymagane kolumny

**Problem:** Brak wyników
- Obniż próg podobieństwa
- Sprawdź czy tekst nie jest za krótki

## 📞 Wsparcie

System oparty na Google Gemini Embeddings API i podobieństwie cosinusowym.
Optymalizowany dla polskich tekstów medycznych i kosmetycznych.
