"""
generate_dialogues.py — Generate kid-safe tutoring dialogues using Gemma 4 31B via OpenRouter.

Safety: Llama Guard 4 12B via OpenRouter (parallel API calls).
Generation: 50 concurrent Gemma 4 requests, 20 concurrent Llama Guard checks.

Usage:
    python -m kidschat.dialogues.generate_dialogues --output dialogues/kidschat_dialogues.jsonl

Requires OPENROUTER_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import os
import random
import re

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Topic templates
# ---------------------------------------------------------------------------

TOPICS = [
    "adding two-digit numbers", "subtracting with borrowing", "what multiplication means",
    "understanding fractions (like half and quarter)", "basic division with sharing equally",
    "counting by 5s and 10s", "what even and odd numbers are",
    "shapes and their properties (triangles, squares, circles)",
    "telling time on an analog clock", "understanding money (coins and dollars)",
    "word problems about buying things at a store", "measuring length with a ruler",
    "patterns in numbers", "greater than and less than", "place value (ones, tens, hundreds)",
    "why the sky is blue", "how plants grow from seeds",
    "the water cycle (rain, rivers, oceans, clouds)", "what dinosaurs were like",
    "how magnets work", "why we have seasons (summer, winter, spring, fall)",
    "the solar system and the planets", "why the moon changes shape",
    "what makes a rainbow", "how volcanoes erupt", "the life cycle of a butterfly",
    "what germs are and why we wash our hands", "how sound travels through the air",
    "why things float or sink in water", "what fossils are and how they form",
    "the food chain in nature", "how earthquakes happen", "what clouds are made of",
    "why leaves change color in autumn", "how bees make honey",
    "how dolphins communicate", "why birds migrate in winter",
    "how chameleons change color", "the difference between reptiles and amphibians",
    "how penguins survive in the cold", "why cats purr", "how spiders make webs",
    "what makes owls good hunters at night", "how elephants use their trunks",
    "the life of ants in a colony", "what life was like in ancient Egypt",
    "how the pyramids were built", "who the first explorers of the ocean were",
    "what medieval castles were like", "the invention of the printing press",
    "how the first airplane flew", "what life was like for kids 100 years ago",
    "the ancient Olympic games", "how people discovered fire", "the first trip to the moon",
    "how mountains form", "what makes the ocean salty", "how rivers are made",
    "why deserts are so dry", "what causes thunder and lightning",
    "how coral reefs form", "why the Earth spins", "what glaciers are",
    "how caves are made", "the layers of the Earth",
    "how a bicycle works", "how electricity gets to our houses", "how bridges stay up",
    "how a refrigerator keeps food cold", "how cameras take pictures",
    "how boats float", "how airplanes fly", "how computers work (very simply)",
    "how a compass tells direction", "how recycling works",
    "why sleeping is important for your body", "how to be a good friend",
    "why exercise is good for you", "how to stay safe near water",
    "why eating vegetables helps you grow", "how to take care of a pet",
    "why reading is a superpower", "how to set a goal and work toward it",
    "what to do if you feel worried", "why it is important to be kind to everyone",
    "what stars are made of", "how astronauts live in space",
    "what a black hole is (simply explained)", "the difference between a planet and a star",
    "what comets and asteroids are", "why Pluto is called a dwarf planet",
    "how telescopes help us see far away", "what the International Space Station is",
    "the rings of Saturn", "whether there could be life on other planets",
]

SYSTEM_PROMPT = """\
You are generating a tutoring conversation between a curious child (age 8-12) \
and a friendly, patient educational assistant. The conversation should feel \
natural and engaging.

Rules:
- The child asks a question and the assistant explains clearly and simply
- The assistant uses warm, encouraging language ("That's a great question!", \
"Let me break that down...", "You're really thinking like a scientist!")
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

GEN_CONCURRENCY = 50   # concurrent Gemma 4 generation requests
GUARD_CONCURRENCY = 30  # concurrent Llama Guard safety checks


# ---------------------------------------------------------------------------
# Llama Guard 4 safety check
# ---------------------------------------------------------------------------

async def llama_guard_check(client: OpenAI, text: str, semaphore: asyncio.Semaphore) -> bool:
    """Check text with Llama Guard 4 12B. Returns True if safe."""
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="meta-llama/llama-guard-4-12b",
                messages=[{"role": "user", "content": text}],
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip().lower()
            return result == "safe"
        except Exception:
            return False  # reject on error — safety first


async def safety_check_dialogue(client: OpenAI, messages: list[dict], guard_sem: asyncio.Semaphore) -> bool:
    """Run blocklist + Llama Guard 4 on all messages."""
    from kidschat.data.blocklist import contains_blocked_content

    # Fast blocklist pre-filter
    for msg in messages:
        is_blocked, _ = contains_blocked_content(msg["content"])
        if is_blocked:
            return False

    # Llama Guard 4 on the full conversation
    full_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    return await llama_guard_check(client, full_text, guard_sem)


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def generate_one(client: OpenAI, topic: str, variation: int, gen_sem: asyncio.Semaphore) -> list[dict] | None:
    """Generate a single dialogue with concurrency limiting."""
    user_prompt = (
        f"Generate a tutoring conversation where a child asks about: {topic}\n"
        f"Variation {variation}: Make it feel unique and natural."
    )

    async with gen_sem:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="google/gemma-4-31b-it",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
                max_tokens=1500,
            )
            content = response.choices[0].message.content.strip()

            # Robustly extract a JSON array from anywhere in the response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                return None
            content = match.group(0)

            messages = json.loads(content)

            if not isinstance(messages, list) or len(messages) < 4:
                return None
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    return None
                if msg["role"] not in ("user", "assistant"):
                    return None
            if messages[0]["role"] != "user":
                return None

            return messages

        except (json.JSONDecodeError, KeyError, IndexError):
            return None
        except Exception as e:
            if "429" in str(e):
                await asyncio.sleep(5)
            return None


async def generate_and_check(client, topic, var, gen_sem, guard_sem):
    """Generate one dialogue and safety-check it."""
    messages = await generate_one(client, topic, var, gen_sem)
    if messages is None:
        return None, "failed"

    is_safe = await safety_check_dialogue(client, messages, guard_sem)
    if not is_safe:
        return None, "rejected"

    return messages, "ok"


async def run_generation(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"Resuming: {existing_count} dialogues already generated")

    target = args.num_dialogues

    pairs = []
    for topic in TOPICS:
        for v in range(1, args.variations_per_topic + 1):
            pairs.append((topic, v))
    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[existing_count:]

    remaining = min(target - existing_count, len(pairs))
    if remaining <= 0:
        print("Already at target. Done.")
        return

    pairs = pairs[:remaining]
    print(f"\nGenerating {remaining} dialogues")
    print(f"  Gemma 4 concurrency: {GEN_CONCURRENCY}")
    print(f"  Llama Guard concurrency: {GUARD_CONCURRENCY}")
    print(f"  Output: {output_path}\n")

    gen_sem = asyncio.Semaphore(GEN_CONCURRENCY)
    guard_sem = asyncio.Semaphore(GUARD_CONCURRENCY)

    generated = existing_count
    failed = 0
    rejected = 0
    pbar = tqdm(total=remaining, desc="Generating", unit=" dialogues")

    out_f = open(output_path, "a")

    # Fire ALL tasks at once, stream results as they complete
    all_tasks = [
        asyncio.ensure_future(generate_and_check(client, t, v, gen_sem, guard_sem))
        for t, v in pairs
    ]

    for coro in asyncio.as_completed(all_tasks):
        messages, status = await coro
        pbar.update(1)
        if status == "failed":
            failed += 1
        elif status == "rejected":
            rejected += 1
        else:
            out_f.write(json.dumps(messages) + "\n")
            out_f.flush()
            generated += 1

        if generated >= target:
            # Cancel remaining
            for t in all_tasks:
                t.cancel()
            break

    out_f.close()
    pbar.close()

    print(f"\n{'=' * 60}")
    print(f"DIALOGUE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total generated:     {generated:,}")
    print(f"Failed (bad format): {failed:,}")
    print(f"Safety rejected:     {rejected:,}")
    print(f"Output file:         {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tutoring dialogues")
    parser.add_argument("--output", type=str, default="dialogues/kidschat_dialogues.jsonl")
    parser.add_argument("--num-dialogues", type=int, default=5000)
    parser.add_argument("--variations-per-topic", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
