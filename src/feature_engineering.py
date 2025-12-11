import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import nltk
from nltk.util import ngrams

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class FeatureExtractor:

    def __init__(self):

        self.tfidf_vectorizer = None
        self.svd_model = None

    def extract_statistical_features(self, text: str, tokens: List[str],
                                     sentences: List[str]) -> Dict[str, float]:

        features = {}

        num_chars = len(text)
        num_words = len(tokens)
        num_sentences = len(sentences)

        if num_words > 0:
            features['avg_word_length'] = sum(
                len(word) for word in tokens) / num_words
        else:
            features['avg_word_length'] = 0

        if num_sentences > 0:
            features['avg_sentence_length'] = num_words / num_sentences
        else:
            features['avg_sentence_length'] = 0

        if num_words > 0:
            unique_words = len(set(tokens))
            features['vocabulary_richness'] = unique_words / num_words
            features['unique_word_count'] = unique_words
        else:
            features['vocabulary_richness'] = 0
            features['unique_word_count'] = 0

        features['lexical_diversity'] = features['vocabulary_richness']

        punctuation_marks = '.,!?;:\'"'
        punctuation_count = sum(
            1 for char in text if char in punctuation_marks)

        if num_chars > 0:
            features['punctuation_ratio'] = punctuation_count / num_chars
        else:
            features['punctuation_ratio'] = 0

        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')

        if num_sentences > 1:
            sentence_lengths = [len(sent.split()) for sent in sentences]
            features['sentence_length_std'] = np.std(sentence_lengths)
        else:
            features['sentence_length_std'] = 0

        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)

        return features

    def extract_readability_features(self, text: str) -> Dict[str, float]:

        features = {}

        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(
                text)

            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(
                text)

            features['automated_readability_index'] = textstat.automated_readability_index(
                text)

            features['dale_chall_readability'] = textstat.dale_chall_readability_score(
                text)

            features['gunning_fog'] = textstat.gunning_fog(text)

            features['smog_index'] = textstat.smog_index(text)

        except Exception as e:
            print(f"Error calculating readability: {e}")
            for key in ['flesch_reading_ease', 'flesch_kincaid_grade',
                        'automated_readability_index', 'dale_chall_readability',
                        'gunning_fog', 'smog_index']:
                features[key] = 0

        return features

    def extract_linguistic_features(self, tokens: List[str]) -> Dict[str, float]:

        features = {}

        try:
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = Counter(tag for word, tag in pos_tags)

            # Calculate ratios for major POS categories
            total_tags = len(pos_tags)
            if total_tags > 0:
                # Nouns
                noun_tags = sum(pos_counts.get(tag, 0)
                                for tag in ['NN', 'NNS', 'NNP', 'NNPS'])
                features['noun_ratio'] = noun_tags / total_tags

                # Verbs
                verb_tags = sum(pos_counts.get(tag, 0)
                                for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
                features['verb_ratio'] = verb_tags / total_tags

                # Adjectives
                adj_tags = sum(pos_counts.get(tag, 0)
                               for tag in ['JJ', 'JJR', 'JJS'])
                features['adjective_ratio'] = adj_tags / total_tags

                # Adverbs
                adv_tags = sum(pos_counts.get(tag, 0)
                               for tag in ['RB', 'RBR', 'RBS'])
                features['adverb_ratio'] = adv_tags / total_tags

                # Pronouns
                pronoun_tags = sum(pos_counts.get(tag, 0)
                                   for tag in ['PRP', 'PRP$', 'WP', 'WP$'])
                features['pronoun_ratio'] = pronoun_tags / total_tags
            else:
                features['noun_ratio'] = 0
                features['verb_ratio'] = 0
                features['adjective_ratio'] = 0
                features['adverb_ratio'] = 0
                features['pronoun_ratio'] = 0

        except Exception as e:
            print(f"Error in POS tagging: {e}")
            features['noun_ratio'] = 0
            features['verb_ratio'] = 0
            features['adjective_ratio'] = 0
            features['adverb_ratio'] = 0
            features['pronoun_ratio'] = 0

        # N-gram features
        if len(tokens) >= 2:
            bigrams = list(ngrams(tokens, 2))
            unique_bigrams = len(set(bigrams))
            features['bigram_diversity'] = unique_bigrams / \
                len(bigrams) if bigrams else 0
        else:
            features['bigram_diversity'] = 0

        if len(tokens) >= 3:
            trigrams = list(ngrams(tokens, 3))
            unique_trigrams = len(set(trigrams))
            features['trigram_diversity'] = unique_trigrams / \
                len(trigrams) if trigrams else 0
        else:
            features['trigram_diversity'] = 0

        return features

    def extract_perplexity_features(self, text: str, tokens: List[str]) -> Dict[str, float]:

        features = {}

        # Word frequency distribution
        if len(tokens) > 0:
            word_freq = Counter(tokens)

            # Entropy of word distribution
            total_words = len(tokens)
            probabilities = [
                count / total_words for count in word_freq.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            features['word_entropy'] = entropy

            # Proportion of unique words
            features['unique_word_ratio'] = len(word_freq) / total_words
        else:
            features['word_entropy'] = 0
            features['unique_word_ratio'] = 0

        return features

    def create_tfidf_features(self, texts: List[str], max_features: int = 1000,
                              ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, TfidfVectorizer]:

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")

        return tfidf_matrix.toarray(), self.tfidf_vectorizer

    def apply_svd_reduction(self, tfidf_matrix: np.ndarray,
                            n_components: int = 100) -> Tuple[np.ndarray, TruncatedSVD]:

        self.svd_model = TruncatedSVD(
            n_components=n_components, random_state=42)
        reduced_matrix = self.svd_model.fit_transform(tfidf_matrix)

        # Explained variance
        explained_var = self.svd_model.explained_variance_ratio_.sum()
        print(f"SVD reduced dimensions to {n_components}")
        print(f"Explained variance: {explained_var:.4f}")

        return reduced_matrix, self.svd_model

    def extract_all_features(self, df: pd.DataFrame,
                             text_column: str = 'normalized_text') -> pd.DataFrame:

        all_features = []

        for idx, row in df.iterrows():

            text = row[text_column]
            tokens = row['tokens']
            sentences = row['sentences']

            stat_features = self.extract_statistical_features(
                text, tokens, sentences)
            read_features = self.extract_readability_features(text)
            ling_features = self.extract_linguistic_features(tokens)
            perp_features = self.extract_perplexity_features(text, tokens)

            combined_features = {
                **stat_features,
                **read_features,
                **ling_features,
                **perp_features
            }

            all_features.append(combined_features)

        features_df = pd.DataFrame(all_features)

        result_df = pd.concat([df, features_df], axis=1)

        print(f"\nExtracted {len(features_df.columns)} numerical features")
        print("Feature names:", features_df.columns.tolist())

        return result_df


def create_feature_matrix(df: pd.DataFrame,
                          feature_columns: List[str] = None) -> np.ndarray:

    if feature_columns is None:
        # Use all numeric columns except label
        feature_columns = df.select_dtypes(
            include=[np.number]).columns.tolist()

        # Remove label columns
        exclude_cols = ['label', 'text_id',
                        'word_count', 'num_tokens', 'num_sentences']
        feature_columns = [
            col for col in feature_columns if col not in exclude_cols]

    X = df[feature_columns].values

    # Handle any NaN or inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    print(f"Feature matrix shape: {X.shape}")

    return X, feature_columns
