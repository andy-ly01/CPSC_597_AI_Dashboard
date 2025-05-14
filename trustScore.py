import re

#keywords for trustworthiness(will add more later, maybe txt file instead?)
POS_TRUST_WORDS = {
    "trustworthy", "trustable", "reliable", "dependable", "authentic",
    "genuine", "honest", "transparent", "secure", "quality", "legit",
    "credible", "professional", "responsible", "sincere", "safe",
    "excellent", "good", "great", "amazing", "fantastic", "superb",
    "wonderful", "outstanding", "impressive", "top-notch",
    "exceptional", "first-rate", "superior", "premium", "high quality"
    "inexpensive", "value for money", "worth the price"
}

NEG_TRUST_WORDS = {
    "scam", "fraud", "fraudulent", "fake", "counterfeit", "deceptive",
    "dishonest", "shady", "unreliable", "untrustworthy", "misleading",
    "bogus", "rip‑off", "poor quality", "breaks", "defective", "cheap",
    "low quality", "damaged", "faulty", "problematic", "inferior",
    "annoying", "disappointing", "terrible", "awful", "horrible",
    "disgusting", "useless", "waste of money", "waste of time"
    "crappy", "bad", "worst", "unpleasant", "wasteful"
}

# pre‑compile a regex pattern for speed
_POS_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in POS_TRUST_WORDS) + r")\b", re.I)
_NEG_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in NEG_TRUST_WORDS) + r")\b", re.I)

#scoring algorithm
def trustScore(text: str,
               sentiment: str,
               conf: float | None = None,
               *,
               base: int = 50,
               pos_weight: int = 8,
               neg_weight: int = 10,
               sent_bonus: int = 10,
               max_score: int = 100) -> int:
    #count keywords
    pos_hits = len(_POS_PATTERN.findall(text))
    neg_hits = len(_NEG_PATTERN.findall(text))
    score = base + pos_weight * pos_hits - neg_weight * neg_hits

    #sentiment adjustment
    sentiment = sentiment.lower()
    if sentiment == "positive":
        score += sent_bonus
    elif sentiment == "negative":
        score -= sent_bonus

    #adjusts score on confidence
    if conf is not None:
        score += int(5 * (conf - 0.5) * 2)   # conf 0.0 →‑5, 1.0 →+5

    return max(0, min(max_score, score))


#standalone Demo
if __name__ == "__main__":
    demo_texts = [
        "Absolutely trustworthy seller, quick delivery and reliable charger.",
        "Total scam cracked after two uses. Fake product!",
        "Average. Works okay but feels cheap.",
    ]
    demo_sents = ["positive", "negative", "neutral"]
    for txt, lab in zip(demo_texts, demo_sents):
        print(f"{lab:8}  {trustScore(txt, lab, conf=0.9):3d}  |  {txt}")
