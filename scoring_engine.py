import json
import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")
    
# ------------ Load rubric & NLP models -----------------

RUBRIC_PATH = Path(__file__).parent / "rubric.json"

with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
    RUBRIC = json.load(f)["rubric"]

# Small, fast sentence-embedding model
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sentiment Analyzer
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


# ------------ Utility helpers -----------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def count_keywords(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def any_keyword(text: str, keywords: List[str]) -> bool:
    return count_keywords(text, keywords) > 0


def get_score_from_range_rules(value: float, rules: Dict[str, int]) -> int:
    for rng, score in rules.items():
        rng = rng.strip()
        if "+" in rng:
            min_v = float(rng.replace("+", ""))
            if value >= min_v:
                return score
        elif rng.startswith(">="):
            if value >= float(rng[2:]):
                return score
        elif rng.startswith(">"):
            if value > float(rng[1:]):
                return score
        elif rng.startswith("<="):
            if value <= float(rng[2:]):
                return score
        elif rng.startswith("<"):
            if value < float(rng[1:]):
                return score
        elif "-" in rng:
            lo, hi = rng.split("-")
            if float(lo) <= value <= float(hi):
                return score

    return min(rules.values())


def semantic_similarity(text: str, description: str) -> float:
    embeddings = EMBED_MODEL.encode([text, description])
    sim = cosine_similarity(
        embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
    )[0][0]
    return float(max(0.0, min(1.0, (sim + 1) / 2)))


# ------------ NO JAVA: Simple Grammar Error Approximation -------------

def simple_grammar_error_estimate(text: str) -> int:
    errors = 0

    errors += len(re.findall(r"  +", text))

    sentences = re.split(r"[.!?]", text)
    for s in sentences:
        s = s.strip()
        if s and not s[0].isupper():
            errors += 1

    for s in sentences:
        if s.strip() and len(s.split()) < 3:
            errors += 1

    common_mistakes = ["teh", "recieve", "definately", "wierd"]
    for m in common_mistakes:
        errors += text.lower().count(m)

    return errors


def get_grammar_quality(text: str, total_words: int) -> float:
    errors = simple_grammar_error_estimate(text)
    if total_words == 0:
        return 0.0

    errors_per_100 = errors / total_words * 100
    quality = 1 - min(errors_per_100 / 10, 1)

    return max(0.0, min(1.0, quality))


def get_ttr(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def get_filler_rate(tokens: List[str], filler_words: List[str]) -> float:
    if not tokens:
        return 0.0
    fillers = sum(1 for t in tokens if t.lower() in filler_words)
    return fillers / len(tokens) * 100


def get_sentiment_pos_score(text: str) -> float:
    vs = SENTIMENT_ANALYZER.polarity_scores(text)
    compound = vs["compound"]
    return (compound + 1) / 2


# ------------ Rule-based scoring for each metric -------------

def score_salutation(text: str, metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rules = metric_cfg["rules"]
    kw = metric_cfg["keywords"]
    text_l = text.lower()

    level = "no_salutation"
    score = rules["no_salutation"]
    found_in = []

    if any(k in text_l for k in kw["excellent"]):
        level = "excellent"
        score = rules["excellent"]
        found_in = kw["excellent"]
    elif any(k in text_l for k in kw["good"]):
        level = "good"
        score = rules["good"]
        found_in = kw["good"]
    elif any(k in text_l for k in kw["normal"]):
        level = "normal"
        score = rules["normal"]
        found_in = kw["normal"]

    feedback = f"Salutation level: {level.replace('_',' ')}."
    if found_in:
        feedback += f" Found: {', '.join([f for f in found_in if f in text_l])}."
    else:
        feedback += " No greeting detected."

    return {
        "rule_score": score,
        "details": {"level": level},
        "feedback": feedback,
    }


def score_keywords(text: str, metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rules = metric_cfg["rules"]
    must_cfg = rules["must_have_each_4"]
    good_cfg = rules["good_to_have_each_2"]

    text_l = text.lower()

    must_present = [kw for kw in must_cfg["items"] if kw.lower() in text_l]
    good_present = [kw for kw in good_cfg["items"] if kw.lower() in text_l]

    score = (
        len(must_present) * must_cfg["score_per_item"]
        + len(good_present) * good_cfg["score_per_item"]
    )
    score = min(score, metric_cfg["weight"])

    feedback = (
        f"Must-have keywords ({len(must_present)}): {', '.join(must_present)}. "
        f"Good-to-have ({len(good_present)}): {', '.join(good_present)}."
    )

    return {
        "rule_score": score,
        "details": {
            "must_present": must_present,
            "good_present": good_present,
        },
        "feedback": feedback,
    }


def score_flow(text: str, metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    text_l = text.lower()

    salutation_present = any(
        s in text_l for s in ["hi", "hello", "good morning", "good afternoon", "good evening"]
    )
    name_present = any(s in text_l for s in ["my name is", "i am ", "this is "])
    closing_present = any(
        s in text_l for s in ["thank you", "thanks for listening", "nice to meet you"]
    )

    if salutation_present and name_present and closing_present:
        score = metric_cfg["rules"]["order_followed"]
        order_ok = True
        feedback = "Good flow (greeting → details → closing)."
    else:
        score = metric_cfg["rules"]["order_not_followed"]
        order_ok = False
        feedback = "Flow incorrect. Expected: greeting → details → closing."

    return {
        "rule_score": score,
        "details": {"order_followed": order_ok},
        "feedback": feedback,
    }


def score_speech_rate(total_words: int, duration_seconds: float, metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if duration_seconds <= 0:
        duration_seconds = 60.0

    wpm = total_words / (duration_seconds / 60.0)

    rules = metric_cfg["rules"]
    score = None
    label = None

    for name, cfg in rules.items():
        rng = cfg["range"]
        s = cfg["score"]
        if rng.startswith(">") and wpm > float(rng[1:]):
            score, label = s, name
        elif rng.startswith("<") and wpm < float(rng[1:]):
            score, label = s, name
        elif "-" in rng:
            lo, hi = rng.split("-")
            if float(lo) <= wpm <= float(hi):
                score, label = s, name

    if score is None:
        score = 0
        label = "unknown"

    feedback = f"Speech rate: {wpm:.1f} WPM ({label})."

    return {
        "rule_score": score,
        "details": {"wpm": wpm, "category": label},
        "feedback": feedback,
    }


def score_grammar(text: str, total_words: int, metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    quality = get_grammar_quality(text, total_words)
    score = get_score_from_range_rules(quality, metric_cfg["rules"])
    feedback = f"Grammar quality: {quality:.2f}."
    return {
        "rule_score": score,
        "details": {"grammar_quality": quality},
        "feedback": feedback,
    }


def score_vocab(tokens: List[str], metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ttr = get_ttr(tokens)
    score = get_score_from_range_rules(ttr, metric_cfg["rules"])
    feedback = f"TTR vocabulary richness: {ttr:.2f}."
    return {
        "rule_score": score,
        "details": {"ttr": ttr},
        "feedback": feedback,
    }


def score_filler(tokens: List[str], metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    filler_words = metric_cfg["filler_words"]
    rate = get_filler_rate(tokens, filler_words)
    score = get_score_from_range_rules(rate, metric_cfg["rules"])
    feedback = f"Filler rate: {rate:.1f}%."
    return {
        "rule_score": score,
        "details": {"filler_rate": rate},
        "feedback": feedback,
    }


def score_sentiment(text: str, metric_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pos_score = get_sentiment_pos_score(text)
    score = get_score_from_range_rules(pos_score, metric_cfg["rules"])
    feedback = f"Positivity score: {pos_score:.2f}."
    return {
        "rule_score": score,
        "details": {"positivity_score": pos_score},
        "feedback": feedback,
    }


# ------------- Semantic Adjustment -------------

SEMANTIC_DESCRIPTIONS = {
    "Salutation Level": "How well the speaker greets the audience at the beginning.",
    "Keyword Presence": "Whether the speaker includes name, age, school, family, hobbies, goals and unique details.",
    "Flow": "Whether the introduction flows smoothly in correct order.",
    "Speech Rate (WPM)": "Whether the speaking speed is clear and comfortable.",
    "Grammar Error Count": "How correct the grammar and structure are.",
    "Vocabulary Richness (TTR)": "How diverse the vocabulary is.",
    "Filler Word Rate": "Frequency of filler words like um, uh, like.",
    "Sentiment / Positivity Score": "How positive, confident, or enthusiastic the tone is."
}


def apply_semantic_adjustment(text: str, metric_name: str, rule_score: float, max_score: float):
    description = SEMANTIC_DESCRIPTIONS.get(metric_name, metric_name)
    sim = semantic_similarity(text, description)
    alpha = 0.7
    final_score = alpha * rule_score + (1 - alpha) * (sim * max_score)
    return {"semantic_similarity": sim, "final_score": final_score}


# ------------ Main Scoring Function -------------

def analyze_transcript(transcript: str, speech_duration_seconds: float = 60.0):
    transcript = transcript.strip()
    tokens = tokenize(transcript)
    total_words = len(tokens)

    overall_score = 0.0
    criteria_results = []

    for crit_cfg in RUBRIC:
        crit_name = crit_cfg["criterion"]
        crit_metrics = crit_cfg["metrics"]
        crit_max = sum(m["weight"] for m in crit_metrics)

        crit_score = 0.0
        metric_results = []

        for metric_cfg in crit_metrics:
            name = metric_cfg["metric"]
            weight = metric_cfg["weight"]

            if name == "Salutation Level":
                result = score_salutation(transcript, metric_cfg)
            elif name == "Keyword Presence":
                result = score_keywords(transcript, metric_cfg)
            elif name == "Flow":
                result = score_flow(transcript, metric_cfg)
            elif name == "Speech Rate (WPM)":
                result = score_speech_rate(total_words, speech_duration_seconds, metric_cfg)
            elif name == "Grammar Error Count":
                result = score_grammar(transcript, total_words, metric_cfg)
            elif name == "Vocabulary Richness (TTR)":
                result = score_vocab(tokens, metric_cfg)
            elif name == "Filler Word Rate":
                result = score_filler(tokens, metric_cfg)
            elif name == "Sentiment / Positivity Score":
                result = score_sentiment(transcript, metric_cfg)
            else:
                continue

            rule_score = result["rule_score"]

            sem = apply_semantic_adjustment(transcript, name, rule_score, weight)
            final_score = sem["final_score"]
            semantic_sim = sem["semantic_similarity"]

            crit_score += final_score

            metric_results.append({
                "metric": name,
                "weight": weight,
                "rule_score": rule_score,
                "semantic_similarity": round(semantic_sim, 3),
                "final_score": round(final_score, 2),
                "feedback": result["feedback"],
                "details": result["details"]
            })

        criteria_results.append({
            "criterion": crit_name,
            "max_score": crit_max,
            "score": round(crit_score, 2),
            "metrics": metric_results
        })

        overall_score += crit_score

    overall_score = max(0.0, min(100.0, overall_score))

    return {
        "overall_score": round(overall_score, 2),
        "total_words": total_words,
        "criteria": criteria_results
    }


# ------------ CLI USAGE  -------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Score a transcript using Nirmaan rubric.")
    parser.add_argument("--file", type=str, help="Path to .txt file")
    parser.add_argument("--duration", type=float, default=60.0, help="Speech duration seconds")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Paste transcript (Ctrl+Z to finish):")
        text = ""
        try:
            while True:
                line = input()
                text += line + "\n"
        except EOFError:
            pass

    result = analyze_transcript(text, args.duration)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

