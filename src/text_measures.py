"""
Five text representations of FOMC post-meeting statements.

The representations are ordered by expressiveness, and the point of applying all
five to the same test is to separate two explanations for a null result: either
statement content genuinely carries no price-relevant information beyond the rate
surprise, or the measure is too coarse to detect it.

    1. bag-of-words hawkishness    lexical counts, transparent, context-blind
    2. FinBERT                     transformer fine-tuned on financial news
    3. CentralBankRoBERTa          transformer fine-tuned on central-bank text
    4. LDA topic intensities       subject-matter composition, not tone
    5. RoBERTa embeddings          full contextual meaning, no supervision

Measures 1 to 3 collapse a statement onto a single axis. Measure 4 describes what
a statement is *about* rather than how it reads. Measure 5 imposes no axis at all:
it uses the model's internal representation directly, which is the most expressive
option and the least interpretable.
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Administrative passages that recur in every post-2008 statement. Left in, they
# form their own topics and crowd out economic content: an early LDA run returned
# a topic consisting of "action, board, approved, discount, basis", which is the
# implementation note rather than monetary policy.
ADMIN_PATTERN = re.compile(
    r"\b(in a related action|board of governors|boards of directors|"
    r"federal reserve banks?|requests submitted|discount rate|primary credit rate|"
    r"reverse repurchase|reserve balances|interest on reserve|voting (for|against)|"
    r"news\s*&?\s*events|media inquiries|last update|recent postings|"
    r"implementation note)\b",
    re.I,
)

# Boilerplate that appears in nearly every statement and so cannot distinguish
# one from another.
FOMC_BOILERPLATE = {
    "committee", "federal", "reserve", "monetary", "policy", "decided", "decision",
    "meeting", "target", "rate", "rates", "longer", "run", "term", "appropriate",
    "remain", "remains", "continue", "continued", "expects", "anticipates", "likely",
    "conditions", "economic", "activity", "level", "levels", "pace", "support",
    "stance", "range", "percent", "action", "board", "approved", "basis", "available",
    "news", "events", "media", "release", "consistent", "mandate", "account",
    "chairman", "announced", "announce", "today", "believes",
}

# Multi-word terms held together so the topic model treats them as one concept.
PHRASE_MAP = {
    r"\bfederal funds\b": "federal_funds",
    r"\blabor market\b": "labor_market",
    r"\bunemployment rate\b": "unemployment",
    r"\bprice pressures?\b": "price_pressure",
    r"\bjob gains\b": "job_gains",
    r"\bmortgage[- ]backed securities\b": "mbs",
    r"\btreasury securities\b": "treasury_securities",
    r"\basset purchases?\b": "asset_purchases",
    r"\bbalance sheet\b": "balance_sheet",
}


def strip_boilerplate(text):
    """Drop the administrative implementation note and voting-record sentences."""
    head = re.split(r"Decisions Regarding Monetary Policy Implementation", str(text))[0]
    sentences = re.split(r"(?<=[.!?])\s+", head)
    return " ".join(s for s in sentences if not ADMIN_PATTERN.search(s))


def normalize(text):
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text, min_tokens=4):
    """Sentence split for the transformer measures, which run per sentence."""
    text = re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= min_tokens]


# ---------------------------------------------------------------------------
# 1. Bag-of-words hawkishness index
# ---------------------------------------------------------------------------

def bow_score(text, hawkish_terms, dovish_terms):
    """
    Normalised hawkish-minus-dovish score in [-1, 1].

    Matching is longest-first with masking, so that a multi-word hawkish phrase
    such as "removing accommodation" fires once and does not also register
    "accommodation" as a dovish hit.
    """
    t = normalize(text)
    hits = {"h": 0, "d": 0}
    for term in sorted(hawkish_terms, key=len, reverse=True):
        n = len(re.findall(rf"\b{re.escape(term)}\b", t))
        hits["h"] += n
        t = re.sub(rf"\b{re.escape(term)}\b", " ", t)
    for term in sorted(dovish_terms, key=len, reverse=True):
        n = len(re.findall(rf"\b{re.escape(term)}\b", t))
        hits["d"] += n
        t = re.sub(rf"\b{re.escape(term)}\b", " ", t)
    return (hits["h"] - hits["d"]) / (hits["h"] + hits["d"] + 1)


# ---------------------------------------------------------------------------
# 4. LDA topic intensities
# ---------------------------------------------------------------------------

def lda_tokenize(text):
    """Strip administrative text, glue key phrases, drop stopwords."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    text = normalize(strip_boilerplate(text))
    for pattern, token in PHRASE_MAP.items():
        text = re.sub(pattern, token, text)
    words = re.findall(r"[a-z_]+", text)
    stop = set(ENGLISH_STOP_WORDS) | FOMC_BOILERPLATE
    return [w for w in words if w not in stop and len(w) > 2]


def fit_lda(statements, k=5, min_df=5, max_df=0.4, max_iter=50, seed=42):
    """
    Estimate topic intensities. Returns (theta, model, vectorizer, vocab).

    theta has one row per statement and K columns summing to one. Because the
    intensities lie on the unit simplex they are collinear with an intercept, so
    one topic is dropped as reference in the regressions.

    K is a researcher choice the model cannot determine. Results in the paper are
    reported for K = 5 and shown to be unchanged for K = 3 to 7.
    """
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from sklearn.decomposition import LatentDirichletAllocation

    docs = [" ".join(lda_tokenize(s)) for s in statements]
    vec = CountVectorizer(
        ngram_range=(1, 2), min_df=min_df, max_df=max_df,
        token_pattern=r"(?u)\b[a-z_]{3,}\b",
        stop_words=list(set(ENGLISH_STOP_WORDS) | FOMC_BOILERPLATE),
    )
    dtm = vec.fit_transform(docs)
    model = LatentDirichletAllocation(
        n_components=k, learning_method="batch", max_iter=max_iter, random_state=seed
    ).fit(dtm)

    theta = model.transform(dtm)
    theta = theta / theta.sum(axis=1, keepdims=True)
    return theta, model, vec, vec.get_feature_names_out()


def umass_coherence(components, dtm, topn=10):
    """
    Self-contained UMass topic coherence, avoiding a gensim dependency.

    Bounded above by zero, so values closer to zero indicate topics whose leading
    words co-occur more often. Note that on a corpus of a couple of hundred short
    documents this measure is noisy and does not reliably identify a single
    optimal K, which is why the paper reports robustness across K instead.
    """
    B = (dtm > 0).astype(int).tocsc()
    df = np.asarray(B.sum(axis=0)).ravel()
    scores = []
    for comp in components:
        top = comp.argsort()[::-1][:topn]
        total, pairs = 0.0, 0
        for a in range(1, len(top)):
            for b in range(a):
                wi, wj = top[a], top[b]
                co = int(B[:, wi].multiply(B[:, wj]).sum())
                total += np.log((co + 1.0) / (df[wj] if df[wj] > 0 else 1.0))
                pairs += 1
        scores.append(total / max(pairs, 1))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 2, 3, 5. Transformer measures
# ---------------------------------------------------------------------------

def sentiment_score(texts, model_name, positive_key="pos", negative_key="neg",
                    max_length=256):
    """
    Sentence-level signed sentiment, averaged over the statement.

    Works for both FinBERT and CentralBankRoBERTa. Class indices are read from the
    model config rather than hard-coded, so a model that ships generic LABEL_0/1/2
    names does not silently produce a sign-flipped score.

    Note that for both models the positive pole records *economic* optimism, not
    dovishness. A sentence describing strong growth scores positive while carrying
    hawkish policy implications, which is why these measures are not substitutes
    for the hawkish-dovish dictionary.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    id2label = {i: l.lower() for i, l in mdl.config.id2label.items()}
    pos = next((i for i, l in id2label.items() if positive_key in l), None)
    neg = next((i for i, l in id2label.items() if negative_key in l), None)
    if pos is None or neg is None:
        raise ValueError(f"could not locate sentiment classes in {id2label}")

    scores = []
    for text in texts:
        sents = split_sentences(text)
        if not sents:
            scores.append(np.nan)
            continue
        net = []
        for s in sents:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=max_length)
            with torch.no_grad():
                p = torch.softmax(mdl(**enc).logits, dim=1).cpu().numpy()[0]
            net.append(float(p[pos] - p[neg]))
        scores.append(float(np.mean(net)))
    return np.array(scores)


def embed_statements(texts, model_name="roberta-base", n_components=5,
                     max_length=256, seed=42):
    """
    Contextual embeddings reduced to their leading principal components.

    Each sentence is encoded and mean-pooled over its tokens with attention
    masking; the statement embedding is the mean over its sentences. The result is
    768-dimensional, which cannot enter a regression with roughly two hundred
    observations, so PCA retains the directions along which statements differ most.

    Interpretation caveat, and it matters: the leading component is strongly
    correlated with the calendar (-0.72 with year in this sample), because the
    largest source of variation in FOMC language across three decades is simply
    when it was written. Embedding *levels* therefore encode the policy regime,
    which is why the paper uses first differences throughout.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.decomposition import PCA

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()

    vectors = []
    for text in texts:
        sents = split_sentences(text) or [str(text)]
        per_sentence = []
        for s in sents:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=max_length)
            with torch.no_grad():
                out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            per_sentence.append(pooled.cpu().numpy()[0])
        vectors.append(np.mean(per_sentence, axis=0))

    mat = np.vstack(vectors)
    std = (mat - mat.mean(0)) / (mat.std(0) + 1e-9)
    return PCA(n_components=n_components, random_state=seed).fit_transform(std)
