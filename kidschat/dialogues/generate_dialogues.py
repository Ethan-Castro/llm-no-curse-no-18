"""
generate_dialogues.py — Generate kid-safe tutoring dialogues using Gemma 4 31B via OpenRouter.

Produces 1,000 multi-turn conversations between a curious child (user) and a
kind, patient educational assistant. Output matches nanochat's CustomJSON
format for SFT training.

Usage:
    python -m kidschat.dialogues.generate_dialogues --output dialogues/kidschat_dialogues.jsonl

Requires OPENROUTER_API_KEY environment variable.
"""

import argparse
import json
import os
import random
import time

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Topic templates
# ---------------------------------------------------------------------------

TOPICS = [
    # Math
    "adding two-digit numbers",
    "subtracting with borrowing",
    "what multiplication means",
    "understanding fractions (like half and quarter)",
    "basic division with sharing equally",
    "counting by 5s and 10s",
    "what even and odd numbers are",
    "shapes and their properties (triangles, squares, circles)",
    "telling time on an analog clock",
    "understanding money (coins and dollars)",
    "word problems about buying things at a store",
    "measuring length with a ruler",
    "patterns in numbers",
    "greater than and less than",
    "place value (ones, tens, hundreds)",

    # Science
    "why the sky is blue",
    "how plants grow from seeds",
    "the water cycle (rain, rivers, oceans, clouds)",
    "what dinosaurs were like",
    "how magnets work",
    "why we have seasons (summer, winter, spring, fall)",
    "the solar system and the planets",
    "why the moon changes shape",
    "what makes a rainbow",
    "how volcanoes erupt",
    "the life cycle of a butterfly",
    "what germs are and why we wash our hands",
    "how sound travels through the air",
    "why things float or sink in water",
    "what fossils are and how they form",
    "the food chain in nature",
    "how earthquakes happen",
    "what clouds are made of",
    "why leaves change color in autumn",
    "how bees make honey",

    # Animals
    "how dolphins communicate",
    "why birds migrate in winter",
    "how chameleons change color",
    "the difference between reptiles and amphibians",
    "how penguins survive in the cold",
    "why cats purr",
    "how spiders make webs",
    "what makes owls good hunters at night",
    "how elephants use their trunks",
    "the life of ants in a colony",

    # History
    "what life was like in ancient Egypt",
    "how the pyramids were built",
    "who the first explorers of the ocean were",
    "what medieval castles were like",
    "the invention of the printing press",
    "how the first airplane flew",
    "what life was like for kids 100 years ago",
    "the ancient Olympic games",
    "how people discovered fire",
    "the first trip to the moon",

    # Nature and Earth
    "how mountains form",
    "what makes the ocean salty",
    "how rivers are made",
    "why deserts are so dry",
    "what causes thunder and lightning",
    "how coral reefs form",
    "why the Earth spins",
    "what glaciers are",
    "how caves are made",
    "the layers of the Earth",

    # How things work
    "how a bicycle works",
    "how electricity gets to our houses",
    "how bridges stay up",
    "how a refrigerator keeps food cold",
    "how cameras take pictures",
    "how boats float",
    "how airplanes fly",
    "how computers work (very simply)",
    "how a compass tells direction",
    "how recycling works",

    # Life skills
    "why sleeping is important for your body",
    "how to be a good friend",
    "why exercise is good for you",
    "how to stay safe near water",
    "why eating vegetables helps you grow",
    "how to take care of a pet",
    "why reading is a superpower",
    "how to set a goal and work toward it",
    "what to do if you feel worried",
    "why it is important to be kind to everyone",

    # Space
    "what stars are made of",
    "how astronauts live in space",
    "what a black hole is (simply explained)",
    "the difference between a planet and a star",
    "what comets and asteroids are",
    "why Pluto is called a dwarf planet",
    "how telescopes help us see far away",
    "what the International Space Station is",
    "the rings of Saturn",
    "whether there could be life on other planets",
]

SYSTEM_PROMPT = """\
You are generating a tutoring conversation between a curious child (age 8-12) \
and a friendly, patient educational assistant. The conversation should feel \
natural and engaging.

Rules:
- The child asks a question and the assistant explains clearly and simply
- The assistant uses warm, encouraging language ("That's a great question!", \
"Let me break that down...", "You're really thinking like a scientist!")
- The assistant asks at least one follow-up question to check understanding \
or spark curiosity
- Use simple vocabulary appropriate for ages 8-12
- The conversation should be 4-8 messages total (alternating user/assistant)
- NEVER use profanity, crude language, slang, or reference any adult content
- Keep explanations accurate but age-appropriate
- Use fun analogies and real-world examples kids can relate to

Output ONLY a JSON array of message objects, no other text. Example format:
[
  {"role": "user", "content": "Why is the sky blue?"},
  {"role": "assistant", "content": "That's a wonderful question! ..."},
  {"role": "user", "content": "Oh cool! ..."},
  {"role": "assistant", "content": "Exactly right! ..."}
]"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_dialogue(client: OpenAI, topic: str, variation: int) -> list[dict] | None:
    """Generate a single tutoring dialogue about a topic."""
    user_prompt = (
        f"Generate a tutoring conversation where a child asks about: {topic}\n"
        f"Variation {variation}: Make it feel unique and natural. "
        f"The child should be genuinely curious and the assistant should be "
        f"encouraging and educational."
    )

    try:
        response = client.chat.completions.create(
            model="google/gemma-4-31b-it",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=1500,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        messages = json.loads(content)

        # Validate format
        if not isinstance(messages, list) or len(messages) < 4:
            return None
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return None
            if msg["role"] not in ("user", "assistant"):
                return None
        # Must start with user
        if messages[0]["role"] != "user":
            return None

        return messages

    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    except Exception as e:
        print(f"    API error: {e}")
        time.sleep(2)
        return None


def safety_check_dialogue(messages: list[dict]) -> bool:
    """Run three-layer safety check on all message content.

    Pass 0: Hardcoded blocklist (failsafe)
    Pass 1: profanity-check (ML-based)
    Pass 2: detoxify (BERT-based toxicity)
    """
    from kidschat.data.blocklist import contains_blocked_content

    try:
        from alt_profanity_check import predict_prob
    except ImportError:
        raise ImportError(
            "profanity-check is required for safety filtering. "
            "Install it: pip install profanity-check"
        )

    try:
        from detoxify import Detoxify
    except ImportError:
        raise ImportError(
            "detoxify is required for safety filtering. "
            "Install it: pip install detoxify"
        )

    texts = [msg["content"] for msg in messages]

    # Pass 0: hardcoded blocklist
    for text in texts:
        is_blocked, _ = contains_blocked_content(text)
        if is_blocked:
            return False

    # Pass 1: profanity-check
    probs = predict_prob(texts)
    if any(p > 0.3 for p in probs):
        return False

    # Pass 2: detoxify
    model = Detoxify("original", device="cpu")
    predictions = model.predict(texts)
    toxicity_labels = ["toxic", "severe_toxic", "obscene", "threat",
                       "insult", "identity_hate", "sexual_explicit"]
    for label in toxicity_labels:
        if label in predictions:
            if any(score > 0.3 for score in predictions[label]):
                return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate tutoring dialogues")
    parser.add_argument(
        "--output",
        type=str,
        default="dialogues/kidschat_dialogues.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-dialogues",
        type=int,
        default=5000,
        help="Target number of dialogues to generate",
    )
    parser.add_argument(
        "--variations-per-topic",
        type=int,
        default=50,
        help="Number of variations to generate per topic",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  Get a key at https://openrouter.ai/keys")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing dialogues for resumability
    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_count = sum(1 for _ in f)
        print(f"Resuming: {existing_count} dialogues already generated")

    target = args.num_dialogues
    generated = existing_count
    failed = 0
    safety_rejected = 0

    # Build (topic, variation) pairs and shuffle for diversity
    pairs = []
    for topic in TOPICS:
        for v in range(1, args.variations_per_topic + 1):
            pairs.append((topic, v))
    random.seed(42)
    random.shuffle(pairs)

    # Skip already-generated pairs
    pairs = pairs[existing_count:]

    print(f"\nGenerating {target - generated} more dialogues (target: {target})")
    print(f"Topics: {len(TOPICS)}, Variations per topic: {args.variations_per_topic}")
    print(f"Output: {output_path}\n")

    with open(output_path, "a") as out_f:
        for topic, variation in tqdm(pairs, desc="Generating", unit=" dialogues"):
            if generated >= target:
                break

            messages = generate_dialogue(client, topic, variation)

            if messages is None:
                failed += 1
                continue

            if not safety_check_dialogue(messages):
                safety_rejected += 1
                continue

            # Write in nanochat CustomJSON format: each line is a JSON array
            out_f.write(json.dumps(messages) + "\n")
            out_f.flush()
            generated += 1

            # Rate limiting
            time.sleep(0.1)

    print(f"\n{'=' * 60}")
    print(f"DIALOGUE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total generated:     {generated:,}")
    print(f"Failed (bad format): {failed:,}")
    print(f"Safety rejected:     {safety_rejected:,}")
    print(f"Output file:         {output_path}")
    print(f"\nEstimated API cost:  <$1.00")


if __name__ == "__main__":
    main()
