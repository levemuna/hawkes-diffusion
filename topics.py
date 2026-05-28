"""Nutrition topic taxonomy + keyword-based classifier.

A post's topic is the subtopic whose keywords overlap most with the post text.
Returns "other" if no overlap. Light-weight; production would use a small
embedding model or supervised classifier.
"""
from __future__ import annotations

from collections import Counter

# Subtopic -> list of keywords / hashtags (lowercased).
NUTRITION_TOPICS: dict[str, list[str]] = {
    "keto_carnivore": [
        "keto", "ketogenic", "carnivore", "low carb", "lchf",
        "#keto", "#carnivore", "#lowcarb", "ketosis", "bhb",
    ],
    "intermittent_fasting": [
        "intermittent fasting", "if", "16:8", "omad", "extended fast",
        "#intermittentfasting", "#fasting", "#omad", "autophagy",
    ],
    "plant_based_vegan": [
        "vegan", "plant based", "plant-based", "whole food plant",
        "#vegan", "#plantbased", "#wfpb", "veganism", "raw vegan",
    ],
    "supplements_detox": [
        "supplement", "supplements", "detox", "cleanse", "liver flush",
        "heavy metals", "parasite cleanse", "binders", "#detox", "#cleanse",
    ],
    "weight_loss_pills": [
        "weight loss", "fat burner", "appetite suppressant", "thermogenic",
        "semaglutide", "ozempic", "mounjaro", "berberine", "ozempic alternative",
        "#weightloss", "#fatloss", "#ozempic",
    ],
    "anti_seed_oil": [
        "seed oil", "seed oils", "vegetable oil", "soybean oil", "canola",
        "#seedoils", "#beeftallow", "tallow", "lard",
    ],
    "gut_microbiome": [
        "gut health", "microbiome", "probiotics", "prebiotic", "leaky gut",
        "candida", "fodmap", "#guthealth", "#microbiome",
    ],
    "raw_milk_traditional": [
        "raw milk", "raw dairy", "ancestral diet", "traditional foods",
        "weston price", "#rawmilk", "#ancestral",
    ],
    "anti_sugar": [
        "sugar free", "no sugar", "sugar addiction", "insulin resistance",
        "blood sugar", "#sugarfree", "#nosugar",
    ],
}


def classify_text(text: str | None) -> str:
    """Return the best-matching topic key, or 'other' if no overlap."""
    if not text:
        return "other"
    t = text.lower()
    scores: Counter[str] = Counter()
    for topic, keywords in NUTRITION_TOPICS.items():
        for kw in keywords:
            if kw in t:
                scores[topic] += len(kw.split())  # multi-word matches weigh more
    if not scores:
        return "other"
    return scores.most_common(1)[0][0]


def all_topic_keys() -> list[str]:
    return list(NUTRITION_TOPICS.keys()) + ["other"]


def default_targets() -> list[tuple[str, str, str]]:
    """Curated seed list: (kind, value, topic) — one hashtag per topic for the demo."""
    return [
        ("hashtag", "#keto", "keto_carnivore"),
        ("hashtag", "#carnivore", "keto_carnivore"),
        ("hashtag", "#intermittentfasting", "intermittent_fasting"),
        ("hashtag", "#omad", "intermittent_fasting"),
        ("hashtag", "#plantbased", "plant_based_vegan"),
        ("hashtag", "#vegan", "plant_based_vegan"),
        ("keyword", "detox cleanse", "supplements_detox"),
        ("keyword", "parasite cleanse", "supplements_detox"),
        ("hashtag", "#ozempic", "weight_loss_pills"),
        ("keyword", "berberine ozempic", "weight_loss_pills"),
        ("hashtag", "#seedoils", "anti_seed_oil"),
        ("hashtag", "#beeftallow", "anti_seed_oil"),
        ("hashtag", "#guthealth", "gut_microbiome"),
        ("hashtag", "#rawmilk", "raw_milk_traditional"),
        ("hashtag", "#sugarfree", "anti_sugar"),
    ]
