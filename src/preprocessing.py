import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import unicodedata

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:

    def __init__(self, remove_stopwords=False, lowercase=True):
        
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
    def clean_text(self, text: str) -> str:
        
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        
        if not isinstance(text, str):
            return ""
        
        text = unicodedata.normalize('NFKD', text)
        
        if self.lowercase:
            text = text.lower()
        
        # Expand contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [word for word in tokens if word.lower() not in self.stop_words]
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def preprocess(self, text: str) -> Dict[str, any]:
        
        cleaned = self.clean_text(text)
        normalized = self.normalize_text(cleaned)
        tokens = self.tokenize_text(normalized)
        sentences = self.tokenize_sentences(normalized)
        
        return {
            'original': text,
            'cleaned': cleaned,
            'normalized': normalized,
            'tokens': tokens,
            'sentences': sentences
        }

def load_and_clean_data(filepath: str, text_column: str = 'content', 
                       label_column: str = 'label') -> pd.DataFrame:
    
    df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")
    
    print(f"\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    df = df.dropna(subset=[text_column, label_column])
    
    original_len = len(df)
    df = df.drop_duplicates(subset=[text_column])
    print(f"\nRemoved {original_len - len(df)} duplicate samples")
    
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    df = df[(df['word_count'] >= 300) & (df['word_count'] <= 1000)]
    print(f"Samples after word count filtering (300-1000 words): {len(df)}")
    
    df = df.reset_index(drop=True)
    
    if 'text_id' not in df.columns:
        df['text_id'] = [f"text_{i:04d}" for i in range(len(df))]
    
    print(f"\nFinal dataset size: {len(df)} samples")
    print(f"Label distribution:\n{df[label_column].value_counts()}")
    
    return df

def preprocess_dataset(df: pd.DataFrame, text_column: str = 'content',
                      save_path: str = None) -> pd.DataFrame:
    
    preprocessor = TextPreprocessor(remove_stopwords=False, lowercase=True)
    
    processed_data = []
    for idx, text in enumerate(df[text_column]):
        
        result = preprocessor.preprocess(text)
        processed_data.append(result)
    
    df['cleaned_text'] = [item['cleaned'] for item in processed_data]
    df['normalized_text'] = [item['normalized'] for item in processed_data]
    df['tokens'] = [item['tokens'] for item in processed_data]
    df['sentences'] = [item['sentences'] for item in processed_data]
    
    df['num_tokens'] = df['tokens'].apply(len)
    df['num_sentences'] = df['sentences'].apply(len)
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved preprocessed data to {save_path}")
    
    return df
