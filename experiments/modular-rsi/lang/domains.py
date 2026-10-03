"""Pre-registered text domains for the compounding continual-acquisition test.

Eight domains, fixed before any experiment ran. Raw files come from GitHub raw; their
SHA-256 hashes are recorded in domains_prereg.json and checked on load. Text is mapped
to a 96-symbol alphabet (newline + printable ASCII 32..126; accents stripped via NFKD;
anything else dropped), Gutenberg headers/footers removed, and each domain is cut to
its first 150,000 characters, split into train/selection/test = 80/10/10.
"""
import hashlib, json, os, re, unicodedata, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "domains")
PREREG = os.path.join(HERE, "domains_prereg.json")
LIC = "https://raw.githubusercontent.com/spdx/license-list-data/main/text/{}.txt"
DOMAINS = {
    "shakespeare": ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"],
    "alice": ["https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt"],
    "python": ["https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py",
               "https://raw.githubusercontent.com/python/cpython/main/Lib/typing.py"],
    "javascript": ["https://raw.githubusercontent.com/lodash/lodash/4.17.21/lodash.js"],
    "latex": ["https://raw.githubusercontent.com/latex3/latex2e/develop/required/amsmath/amsmath.dtx"],
    "markdown": ["https://raw.githubusercontent.com/facebook/react/main/CHANGELOG.md"],
    "legal": [LIC.format(l) for l in ("GPL-3.0-only", "AGPL-3.0-only", "LGPL-2.1-only", "MPL-2.0",
                                      "Apache-2.0", "EPL-2.0", "CC-BY-SA-4.0", "GPL-2.0-only")],
    "french": ["https://raw.githubusercontent.com/GITenberg/Candide_4650/master/4650-8.txt"],
}
ALPHABET = "\n" + "".join(chr(c) for c in range(32, 127))   # 96 symbols
STOI = {c: i for i, c in enumerate(ALPHABET)}
N_CHARS = 150_000


def _fetch(url):
    os.makedirs(RAW, exist_ok=True)
    p = os.path.join(RAW, hashlib.md5(url.encode()).hexdigest())
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    return open(p, "rb").read()


def _clean(raw, url):
    t = raw.decode("utf-8") if b"\xc3" in raw[:200000] else raw.decode("latin-1")
    if "GITenberg" in url:
        a = max(t.find("*** START"), t.find("CHAPTER I"), 0)
        a = t.find("\n", a) + 1 if a else 0
        b = t.find("End of Project Gutenberg"); b = t.find("*** END") if b < 0 else b
        t = t[a:b if b > 0 else None]
    t = unicodedata.normalize("NFKD", t).replace("\r\n", "\n").replace("\t", "    ")
    return "".join(c for c in t if c in STOI)


def build(register=False):
    reg = json.load(open(PREREG)) if os.path.exists(PREREG) and not register else {}
    out, meta = {}, {}
    for name, urls in DOMAINS.items():
        parts = []
        for u in urls:
            raw = _fetch(u)
            h = hashlib.sha256(raw).hexdigest()
            if reg and reg[name]["sha256"][u] != h:
                raise RuntimeError(f"{name}: {u} changed since pre-registration")
            parts.append(_clean(raw, u))
            meta.setdefault(name, {"urls": urls, "sha256": {}})["sha256"][u] = h
        t = re.sub(r"\n{3,}", "\n\n", "\n".join(parts))[:N_CHARS]
        ids = np.array([STOI[c] for c in t], np.int64)
        a, b = int(len(ids) * 0.8), int(len(ids) * 0.9)
        out[name] = (ids[:a], ids[a:b], ids[b:])
        meta[name]["n_chars"] = len(ids)
    if register:
        json.dump(meta, open(PREREG, "w"), indent=1)
    return out


if __name__ == "__main__":
    d = build(register=not os.path.exists(PREREG))
    for k, (tr, se, te) in d.items():
        print(f"{k:12s} train={len(tr):6d} sel={len(se):5d} test={len(te):5d} | "
              + "".join(ALPHABET[i] for i in tr[2000:2080]).replace("\n", "\\n"))
