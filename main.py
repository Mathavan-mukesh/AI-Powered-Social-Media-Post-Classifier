import json
from llama_cpp import Llama

# Initialize LLaMA model
llm = Llama(
    model_path="/home/mathavan/my_models/tinyllama.gguf",
    n_ctx=512,
    verbose=False
)

# Priority tags
priority_tags = ["spirituality", "travel", "yoga", "food", "games", "public speaking"]

# Few-shot prompt
few_shot_prompt = """
You are an assistant that extracts relevant tags from user posts. Respond with lowercase, comma-separated tags.

Examples:
Post: Just made homemade biryani – turned out better than I expected!
Tags: food, homemade, biryani

Post: I went hiking in the Himalayas and meditated every morning.
Tags: travel, meditation, hiking, spirituality

Post: Just installed Android 14 – the AI camera features are amazing!
Tags: technology, android, ai

Post: I'm reading the Bhagavad Gita and practicing silence every morning.
Tags: spirituality, meditation, books
"""

# Posts
posts = [
    "Just made homemade biryani for the first time – turned out better than I expected! 🍛",
    "Woke up at 5 AM today to meditate as the sun rose over the quiet fields. There's something deeply healing about sitting in stillness...",
    "Excited to be starting my MBA at IIM Ahmedabad.",
    "Visited the Taj Mahal for the first time – breathtaking!",
    "Trying out the new Android 14 beta – some cool features here.",
    "Caught up on the latest K-drama – totally hooked!",
    "I'm a digital artist from Mumbai.",
    "Just voted in the local elections. Every voice matters!",
    "After months of planning, I finally embarked on my solo trip to the Himalayas. The breathtaking views, the peaceful monasteries...",
    "Went on a trekking trip to Himachal – unforgettable experience!",
    "rajini next flim name is coolie",
    "next tamil nadu chief minister is vijay"
]

output_data = []

for post in posts:
    full_prompt = f"{few_shot_prompt}\n\nPost: {post}\nTags:"

    response = llm.create_completion(
        prompt=full_prompt,
        max_tokens=50,
        temperature=0.3,
        stop=["\nPost:"],  
    )
    raw_tags = response["choices"][0]["text"].strip()
    model_tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


    priority_included = [
        tag for tag in priority_tags
        if tag in post.lower() or tag in model_tags
    ]

    final_tags = list(dict.fromkeys(priority_included + model_tags))

    output_data.append({
        "post": post,
        "tags": final_tags
    })


with open("tagged_posts.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("✅ Tagged posts saved to 'tagged_posts.json'")
