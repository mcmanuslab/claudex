"""Text data for the language experiments: Alice in Wonderland (public domain) and a
10k common-English word list (used only to judge whether generated words are real).
Downloaded on first use into lang/data/ (not committed)."""
import os, re, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = {
    "alice.txt": "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt",
    "words10k.txt": "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt",
}
ALPHABET = " abcdefghijklmnopqrstuvwxyz."   # 28 symbols: space, a-z, sentence end
STOI = {c: i for i, c in enumerate(ALPHABET)}


def _get(name):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        urllib.request.urlretrieve(SRC[name], p)
    return open(p, encoding="utf-8", errors="ignore").read()


def corpus():
    t = _get("alice.txt")
    a, b = t.find("CHAPTER I"), t.find("End of Project Gutenberg")
    t = t[a:b if b > 0 else None].lower()
    t = re.sub(r"[!?;:]", ".", t)
    t = re.sub(r"[^a-z. ]+", " ", t.replace("\n", " "))
    t = re.sub(r"\s*\.\s*", ". ", t)
    t = re.sub(r"(\. )+", ". ", t)
    t = re.sub(r" +", " ", t).strip()
    ids = np.array([STOI[c] for c in t], np.int64)
    n = int(len(ids) * 0.9)
    return t, ids[:n], ids[n:]


def dictionary():
    words = set(_get("words10k.txt").split())
    t, _, _ = corpus()
    words |= set(w.strip(".") for w in t.split())   # words that actually occur in the corpus
    return {w for w in words if w}


def decode(ids):
    return "".join(ALPHABET[i] for i in ids)
