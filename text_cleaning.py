import pandas as pd
import re
import unicodedata
from collections import Counter
from num2words import num2words
import spacy
from nltk.stem.snowball import SnowballStemmer

class TextCleaning:
    def __init__(self, text_column: str, df: pd.DataFrame):
        # Custom list of Spanish stop words
        self.spanish_stop_words = [
        "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante", "en",
        "hacia", "hasta", "para", "por", "segun", "so", "sobre", "tras", "ser",
        "del", "al","la", "lo", "las", "los", "el", "las", "un", "una", "unos", "unas",
        "habilidad", "habilidades", "ambiente", "ambito", "laboral", "y", "o", "yo", "e",
        "capaz", "capacidad", "capacidades", "conocimiento", "conocimientos", "uso",
        "utilizar", "gusto", "mentalidad", "promover", "realizar",  "aptitud",
        "enfoque", "realizacion", "tener", "competente", "su", "como",
        "dominio", "relacionado", "manera", "forma", "deseable", "hacia",
        "saber", "excelente", "excelentes", "buen","bueno","buena",
        "buenos", "buenas","avanzado", "avanzada","avanzados", "avanzadas",
        "perfecto", "perfecta","perfectos","perfectas", "basico", "basica",
        "basicas", "basicos",  "implicito", "implicita", "adecuado", "adecuados",
        "adecuada", "adecuadas"
    ]
        # Load the spanish model for spaCy
        self.nlp = spacy.load('es_core_news_lg')
        self.text_column= text_column
        self.df= df

        # Initialize Spanish stemmer
        self.stemmer = SnowballStemmer("spanish")
        
    def remove_accents(self, text: str) -> str:
        # Define a mapping of accented vowels to their non-accented equivalents
        accents_mapping = {
            'á': 'a',
            'é': 'e',
            'í': 'i',
            'ó': 'o',
            'ú': 'u',
            'Á': 'A',
            'É': 'E',
            'Í': 'I',
            'Ó': 'O',
            'Ú': 'U'
        }
        # Replace accented vowels using the mapping
        pattern = re.compile('|'.join(accents_mapping.keys()))
        return pattern.sub(lambda x: accents_mapping[x.group()], text)

    # Clean corrupted characters
    def clean_text(self, text: str) -> str:
        text = re.sub(r'ñ', 'n', text)  # Replace corrupted 'ñ' with 'n'
        # Fix the corrupted sequence - it's Â­ between i and t, not iÂ­
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'i¬∫', 'u', text)
        text = re.sub(r'\u00C2\u00AD', '', text)
        # Clean common encoding issues like '_x0081_' or 'Ã±'
        text = re.sub(r'_x[0-9A-Fa-f]{4}_', '', text)  # Removes codes like '_x0081_'
        text = re.sub(r'Ã±', 'n', text)  # Replace corrupted 'Ã±' with 'n'
        text = re.sub(r'ã±', 'n', text)  # Replace corrupted 'ã±' with 'n'
        text = re.sub(r'Ã³', 'o', text)  # Replace corrupted 'Ã³' with 'ó'
        text = re.sub(r'ã³', 'o', text)
        text = re.sub(r'Ã¡', 'a', text)  # Replace corrupted 'Ã¡' with 'á'
        text = re.sub(r'ã¡', 'a', text)  # Replace corrupted 'ã¡' with 'á'
        text = re.sub(r'Ã©', 'e', text)  # Replace corrupted 'Ã©' with 'é'
        text = re.sub(r'ã©', 'e', text)  # Replace corrupted 'Ã©' with 'é'
        text = re.sub(r'Ã¢', 'a', text)  # Replace corrupted 'Ã¢' with 'á'
        text = re.sub(r'ã¢', 'a', text)  # Replace corrupted 'Ã¢' with 'á'
        text = re.sub(r'iÂ­', 'i', text)  # Replace corrupted 'iÂ' with 'í'
        text = re.sub(r'iâ­', 'i', text)  # Replace corrupted 'iâ' with 'í'
        text = re.sub(r'i\u00ad', 'i', text)  # Replace corrupted 'i\u00ad' with 'í'
        text = re.sub(r'i\xC2\xAD', 'i', text)  # Replace corrupted 'i\u00ad' with 'í'
        text = re.sub(r'i\u00C2\u00AD', 'i', text)
        text = re.sub(r'â€œ', 'o', text)  # Replace corrupted 'â€œ' with 'ó'
        text = re.sub(r'a x0081_n', 'on', text)  # Replace corrupted 'oÃŒ _x0081_' with 'ó'
        text = re.sub(r'a x0081_o', 'o', text)  # Replace corrupted 'oÃŒ _x0081_' with 'ó'
        text = re.sub(r'i º', 'u', text)  # Replace corrupted 'i º' with 'ú'
        text = re.sub(r'Â¼', 'u', text)  # Replace corrupted '' with 'ú'
        text = re.sub(r'â¼', 'u', text)  # Replace corrupted '' with 'ú'
        text = re.sub(r" -culos", 'culos', text)
        text = re.sub(r" -culo", 'culo', text)
        text = re.sub(r'vehi[^a-zA-Z]*culos', 'vehiculos', text, flags=re.IGNORECASE)
        text = re.sub(r'vehi[^a-zA-Z]*culo', 'vehiculo', text, flags=re.IGNORECASE)
        text = re.sub(r'mi ltiples', 'multiples', text, flags=re.IGNORECASE)
        # To join thousand numbers separated by a comma
        text = re.sub(r'(?<=\d),(?=\d{3})', '', text)
        # To separate numbers joined with string
        text = re.sub(r'(\D+)(\d+)', r'\1 \2', text)
        text = re.sub(r'\(', '', text)
        text = re.sub(r'\)', '', text)
        text = re.sub(r'\[', '', text)
        text = re.sub(r'\]', '', text)
        text = re.sub(r',', ' ', text)
        text = re.sub(r'\+', ' mas ', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub(r"'", ' ', text)
        text = re.sub(r"\.", ' ', text)
        text = re.sub(r'"', ' ', text)
        text = re.sub(r':', ' ', text)
        text = re.sub(r"\\", ' ', text)
        text = re.sub(r"/", ' ', text)
        text = re.sub(r"%", ' porcentaje ', text)
        text = re.sub(r"\$", ' dinero ', text)
        text = re.sub(r" culos", 'culos', text)
        text = re.sub(r" culo", 'culo', text)
        text = re.sub(r"objetivoshabilidad", 'objetivos', text)
        text = re.sub(r"objetivoscapacidad", 'objetivos', text)
        text = re.sub(r"flexibilidadadaptabilidad", 'flexibilidad adaptabilidad', text)
        text = re.sub(r"auto motivado", "automotivado", text)
        text = re.sub(r"auto dirigida", "autodirigida", text)
        text = re.sub(r"auto confianza", "autoconfianza", text)
        text = re.sub(r"auto motivacion", "automotivacion", text)
        text = re.sub(r"implcito", 'implicito', text)
        text = re.sub(r"mecnica", 'mecanica', text)
        text = re.sub(r"boasica", 'basico', text)
        text = re.sub(r"ensenanzaaprendizaje", "ensenanza aprendizaje", text)
        text = re.sub(r"conoci miento", 'conocimiento', text)
        text = re.sub(r"acute", '', text)
        text = re.sub(r'fi\s*\u00ADsica', 'fisica', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        # Normalize Unicode (fixes issues like 'comunicaciã³n' -> 'comunicación')
        text = unicodedata.normalize('NFKD', text)
        # Remove non-ASCII characters (optional: you can replace or remove them)
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Removes non-ASCII characters
        # Strip extra spaces and some symbols
        text = text.strip()
        return text


    # Function to remove custom stopwords from a text
    def remove_stopwords(self, text: str, stopwords) -> str:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return ' '.join(filtered_words)

    def convert_numbers_to_words(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        # Match standalone numbers only
        pattern = r'\b\d+\b'

        def replace_match(match):
            number = int(match.group(0))
            try:
                return num2words(number, lang='es')
            except ValueError:
                return match.group(0)

        return re.sub(pattern, replace_match, text)

    def data_cleaning(self):
        # Pre proccesing text
        self.df['processed_text']=  self.df[self.text_column].astype(str)
        # Converting to lower case
        self.df['processed_text']=  self.df['processed_text'].str.lower()
        # Removing accents
        self.df['processed_text'] = self.df['processed_text'].apply(self.remove_accents)
        self.df['processed_text']  = self.df['processed_text'] .apply(self.clean_text)
        # Converting numbers to words
        self.df['processed_text'] = self.df['processed_text'].apply(lambda x: self.convert_numbers_to_words(x))
        self.df['processed_text'] = self.df['processed_text'].apply(self.remove_accents)
        # Removing stop words
        self.df['processed_text']  = self.df['processed_text'] .apply(lambda x: self.remove_stopwords(x, self.spanish_stop_words))
    
    # Function to lemmatize Spanish text
    def lemmatize_spacy(self, text: str) -> str:
        doc = self.nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct])
        return lemmatized_text
    
    def stem_snowball_spanish(self,text):
        doc = self.nlp(text)
        stemmed_tokens = [
            self.stemmer.stem(token.text.lower()) for token in doc
            if not token.is_punct and not token.is_space
        ]
        return ' '.join(stemmed_tokens)
    
    # Normalize the original column based on mapping
    def normalize_text_by_stem(self, original, stemmed, stem_to_orig_map):
        orig_tokens = original.split()
        stem_tokens = stemmed.split()

        if len(orig_tokens) != len(stem_tokens):
            return original 

        normalized = []
        for orig_word, stem_word in zip(orig_tokens, stem_tokens):
            if stem_word in stem_to_orig_map:
                normalized.append(stem_to_orig_map[stem_word])
            else:
                normalized.append(orig_word)
        return ' '.join(normalized)
    
    def normalize(self):
        self.df["lemmatized"]= self.df['processed_text'].apply(self.lemmatize_spacy)
        self.df["lemmatized"] = self.df["lemmatized"].apply(self.remove_accents)
        self.df["lemmatized"]  = self.df["lemmatized"].apply(self.clean_text)
        self.df["lemmatized"] = self.df["lemmatized"].apply(lambda x: self.remove_stopwords(x, self.spanish_stop_words))
        self.df['stemming']= self.df['processed_text'].apply(self.stem_snowball_spanish)
        # Tokenize and build word-level mappings
        word_pairs = []

        for orig_text, stem_text in zip(self.df['lemmatized'], self.df['stemming']):
            orig_words = orig_text.split()
            stem_words = stem_text.split()

            if len(orig_words) == len(stem_words):
                word_pairs.extend(zip(stem_words, orig_words))

        # Count all stem occurrences
        stem_counts = Counter(stem for stem, _ in word_pairs)
        n = 50
        top_stems = [stem for stem, _ in stem_counts.most_common(n)]

        # For each top stem, find most common original word
        stem_to_common_original = {}

        for stem in top_stems:
            origs = [orig for s, orig in word_pairs if s == stem]
            most_common_orig = Counter(origs).most_common(1)[0][0]
            stem_to_common_original[stem] = most_common_orig

        # Apply normalization
        self.df['normalized'] = self.df.apply(
            lambda row: self.normalize_text_by_stem(row['processed_text'], row['stemming'], stem_to_common_original),
            axis=1
        )

        self.df = self.df.drop(["lemmatized", "stemming"], axis=1)
        self.df.to_csv('text_preprocessing.csv', index= False)


