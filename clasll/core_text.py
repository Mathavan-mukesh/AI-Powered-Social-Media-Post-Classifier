# first.py
from sentence_transformers import SentenceTransformer, util
from model import get_core_context, get_final_tags, PRIORITY_CATEGORIES

class ContextAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_most_related_word(self, list1, list2, threshold=0.55):
        list1_clean = [item.strip().lower() for item in list1]
        list2_clean = [item.strip().lower() for item in list2]

        emb_list1 = self.model.encode(list1_clean, convert_to_tensor=True)
        emb_list2 = self.model.encode(list2_clean, convert_to_tensor=True)

        best_word, best_score, best_match = None, 0.0, ""

        for idx, word2 in enumerate(list2_clean):
            if word2 in list1_clean:
                continue
            sim_scores = util.cos_sim(emb_list2[idx], emb_list1)[0]
            max_idx = int(sim_scores.argmax())
            max_score = float(sim_scores[max_idx])

            if max_score > best_score and max_score >= threshold:
                best_score = max_score
                best_word = list2[idx]
                best_match = list1[max_idx]

        if best_word:
            list1.append(best_word)

        return list1, best_word, best_match, round(best_score, 3)

    def analyze(self, sentence):
        core_categories = get_core_context(sentence)
        final_tags = get_final_tags(core_categories, PRIORITY_CATEGORIES)
        updated_tags, added, matched, score = self.add_most_related_word(final_tags, PRIORITY_CATEGORIES)

        return {
            "core_categories": core_categories,
            "final_tags": final_tags,
            "enhanced_tags": updated_tags,
            "added": added,
            "matched": matched,
            "score": score
        }


if __name__ == "__main__":
    analyzer = ContextAnalyzer()

    while True:
        post = input("\n📝 Enter a post: ").strip()
        if not post:
            continue

        result = analyzer.analyze(post)

        print("✅ Core Categories:", result["core_categories"])
        print("🏷️  Final Tags:", result["final_tags"])
        print("📌 Enhanced Tags:", result["enhanced_tags"])

        if result["added"]:
            print(f"➕ Added tag: '{result['added']}' (similar to '{result['matched']}', score: {result['score']})")
