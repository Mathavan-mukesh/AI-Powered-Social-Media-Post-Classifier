# main.py
from flask import Flask, request, jsonify
from llama_cpp import Llama
from difflib import SequenceMatcher
from model import KeywordExtractor  # Assumes your keyword logic is here
import string

app = Flask(__name__)

# Load the LLaMA model once globally
llm = Llama(
    model_path="/home/mathavan/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=1024,
    temperature=0.0,
    verbose=False
)

# Predefined categories
PRIORITY_CATEGORIES = [
    "public speaking", "travel", "spirituality", "food", "games", "yoga"
]

# Prompt template
FEW_SHOT_PROMPT = """
You are an expert assistant that analyzes user posts and extracts only the 3 most relevant high-level core categories for the content.

Examples:

Post: "I want to buy a formula 1 race car"
Core Categories: sports, car race, statement

Post: "Had an amazing trekking experience in the Himalayas"
Core Categories: travel, adventure, nature

Post: "Struggling to stay productive while working from home"
Core Categories: productivity, work, lifestyle

Now analyze the following:

Post: "{post}"
Core Categories:"""

# Utility functions
def clean_text_list(text_list):
    return [x.strip().lower() for x in text_list if x.strip()]

def capitalize_properly(text):
    return " ".join(word.capitalize() for word in text.split())

def is_similar(a: str, b: str, threshold=0.8) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def get_core_context(post: str) -> list:
    prompt = FEW_SHOT_PROMPT.format(post=post)
    llm.reset()
    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.0,
        stop=["\nPost:", "\npost:", "\nNow analyze", "\nExamples"]
    )
    response_text = output["choices"][0]["text"].strip()
    raw_categories = clean_text_list(response_text.split(","))
    prioritized = [cat for cat in raw_categories if cat in PRIORITY_CATEGORIES]
    others = [cat for cat in raw_categories if cat not in prioritized]
    return (prioritized + others)[:3]

def get_final_tags(core_categories: list, priority_tags: list) -> list:
    core_set = set([c.lower().strip() for c in core_categories])
    cleaned_tags = clean_text_list(priority_tags)

    filtered_tags = []
    for tag in cleaned_tags:
        tag_lower = tag.lower()
        if tag_lower in core_set:
            continue
        for core in core_set:
            if is_similar(tag_lower, core, threshold=0.6):
                filtered_tags.append(tag_lower)
                break

    if not filtered_tags:
        return [capitalize_properly(tag) for tag in core_categories]

    final = core_categories + filtered_tags
    final_unique = []
    for tag in final:
        tag_lower = tag.lower()
        if tag_lower not in final_unique:
            final_unique.append(tag_lower)

    return [capitalize_properly(tag) for tag in final_unique[:4]]

def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Initialize keyword extractor
keyword_extractor = KeywordExtractor()

@app.route("/analyze", methods=["POST"])
def analyze_post():
    data = request.get_json()
    sentence = data.get("sentence", "").strip()

    if not sentence:
        return jsonify({"error": "Please provide a valid sentence."}), 400

    # Core category tagging
    core_categories = get_core_context(sentence)
    enhanced_tags = get_final_tags(core_categories, core_categories)

    # Keyword extraction
    raw_keywords = keyword_extractor.extract_keywords(sentence)
    top_keywords = keyword_extractor.get_top_keywords_from_list(raw_keywords, sentence)
    keyword_list = [word for word, _ in top_keywords] if top_keywords else []

    # Combine and deduplicate
    all_tags = enhanced_tags + keyword_list
    seen = set()
    unique_tags = []
    for tag in all_tags:
        norm_tag = normalize(tag)
        if norm_tag not in seen:
            seen.add(norm_tag)
            unique_tags.append(tag)

    return jsonify({
        "final_combined_tags": unique_tags
    })

if __name__ == "__main__":
    app.run(debug=True)
