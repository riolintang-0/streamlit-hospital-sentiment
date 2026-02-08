import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    
    def __init__(self, slang_path=None):
        print("🤖 Menyalakan Robot Preprocessor...")
        
        # 1. LOAD STOPWORDS
        self.stop_factory = StopWordRemoverFactory()
        self.stopwords = self.stop_factory.create_stop_word_remover()
        
        # 2. LOAD STEMMER
        self.stem_factory = StemmerFactory()
        self.stemmer = self.stem_factory.create_stemmer()
        
        # 3. LOAD KAMUS ALAY
        self.slang_dict = {}
        if slang_path:
            self._load_slang_dictionary(slang_path)
        
        print("✅ Robot Siap! Semua kamus sudah dimuat.")
    
    def _load_slang_dictionary(self, path):
        try:
            # Pastikan delimiter sesuai dengan CSV anda (titik koma atau koma)
            df_slang = pd.read_csv(path, encoding='latin-1', sep=';', header=None)
            self.slang_dict = dict(zip(df_slang[0], df_slang[1]))
        except Exception as e:
            print(f"⚠️ Gagal load kamus normalisasi: {e}")
            
    def normalized_slang(self, tokens):
        """
        Mengubah kata alay menjadi baku.
        Input: LIST of strings (token)
        Output: LIST of strings
        """
        if not self.slang_dict:
            return tokens
        
        # Karena inputnya sudah berupa List (tokens), kita bisa langsung loop
        return [self.slang_dict.get(word, word) for word in tokens]
        
    def clean_text(self, text, use_stemming=True):
        """
        Proses Cleaning Text
        """
        if pd.isna(text):
            return ""
        
        # 1. Lowercase
        text = str(text).lower()
        
        # 2. Hapus Karakter non-Alphabet
        # Tips: Ganti dengan ' ' (spasi), bukan '' (kosong) agar kata tidak menempel
        text = re.sub(r'[^a-z\s]+', ' ', text)
        
        # 3. Hapus Spasi berlebih
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # 4. Tokenisasi (Mengubah String jadi LIST)
        # Output: ['saya', 'sakit', 'bgt']
        tokens = word_tokenize(text)
        
        # 5. Normalisasi Alay (Input List -> Output List)
        # Output: ['saya', 'sakit', 'banget']
        tokens = self.normalized_slang(tokens)
        
        # --- PERBAIKAN UTAMA DISINI ---
        
        # 6. Gabungkan kembali List menjadi String (Wajib untuk Sastrawi)
        text = ' '.join(tokens)
        
        # 7. Stopword Removal (Sastrawi butuh String)
        text = self.stopwords.remove(text)
        
        # 8. Stemming (Sastrawi butuh String)
        if use_stemming:
            text = self.stemmer.stem(text) 
            
        return text
        
        
        