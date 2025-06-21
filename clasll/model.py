# model.py
from collections import Counter
import re
import math

class KeywordExtractor:
    def __init__(self):
        self.stopwords = set([
            "the", "is", "in", "and", "a", "an", "of", "to", "with", "on", "for", "this", "that", "it", "at"
        ])

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def extract_keywords(self, text):
        words = self.tokenize(text)
        words = [w for w in words if w not in self.stopwords]
        return Counter(words).items()

    def get_top_keywords_from_list(self, word_freqs, text, top_n=5):
        words = self.tokenize(text)
        word_scores = []

        for word, freq in word_freqs:
            if word in words:
                tf = freq / len(words)
                idf = math.log(1 + len(words) / (1 + words.count(word)))
                score = tf * idf
                word_scores.append((word, score))

        return sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]
