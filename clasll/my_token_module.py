# second.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('stopwords')

class KeywordExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_keywords(self, text):
        words = word_tokenize(text)
        keywords = [word.lower() for word in words if word.isalnum() and word.lower() not in self.stop_words]
        return list(dict.fromkeys(keywords))  # remove duplicates, preserve order

    def get_top_keywords_from_list(self, word_list, sentence, top_n=6, threshold=0.4):
        if not word_list:
            return []

        word_embeddings = self.model.encode(word_list, convert_to_tensor=True)
        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)

        similarities = util.cos_sim(word_embeddings, sentence_embedding).squeeze()
        scored = list(zip(word_list, similarities.tolist()))

        filtered_sorted = sorted(
            [(word, score) for word, score in scored if score >= threshold],
            key=lambda x: x[1],
            reverse=True
        )

        return [(word, round(score * 100, 2)) for word, score in filtered_sorted[:top_n]]


if __name__ == "__main__":
    extractor = KeywordExtractor()

    sentence = input("\n🔹 Enter your sentence: ").strip()
    if not sentence:
        print("⚠️ Please enter a valid sentence.")
    else:
        raw_keywords = extractor.extract_keywords(sentence)
        print("🔍 Extracted Keywords:", raw_keywords)

        top_keywords = extractor.get_top_keywords_from_list(raw_keywords, sentence)

        if top_keywords:
            keyword_list = [word for word, _ in top_keywords]
            print("\n🏆 Top 6 Important Keywords List:", keyword_list)
        else:
            print("⚠️ No important keywords found.")
