"""Text normalization and near-duplicate detection.

Two texts are near-duplicates only if they contain the SAME numbers and their content words
overlap (Jaccard) >= threshold. Keeps "pH of 0.01 M HCl" vs "0.02 M HCl" apart, catches rewordings."""
import re
from collections import defaultdict

NEAR_DEFAULT = 0.70

_WS = re.compile(r"\s+")

_PUNCT = re.compile(r"(?<!\d)\.|\.(?!\d)|[^\w\s.]")   # drop punctuation, keep decimal points

def normalize(text):
    """lowercase, drop punctuation, collapse whitespace. Numbers and units are kept."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()



_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")

STOPWORDS = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose how why when where can could would should will shall may might must i you
he she it we they me my your our their this that these those there here please tell explain give find
calculate determine compute s t""".split())

def signature(norm_text):
    """(numbers, content-word set) for a normalized text. Words get a crude plural strip."""
    toks = _TOKEN.findall(norm_text)
    nums = tuple(sorted(t for t in toks if t[0].isdigit()))
    words = {(t[:-1] if len(t) > 3 and t.endswith("s") else t) for t in toks if t not in STOPWORDS}
    return nums, frozenset(words)

def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0

class NearIndex:
    """Near-duplicate lookup. Only texts with identical numbers are compared, so this stays fast."""
    def __init__(self, threshold):
        self.threshold, self.buckets = threshold, defaultdict(list)
    def add(self, norm_text):
        nums, words = signature(norm_text)
        self.buckets[nums].append((words, norm_text))
    def find(self, norm_text):
        nums, words = signature(norm_text)
        for other_words, other_text in self.buckets.get(nums, ()):
            if _jaccard(words, other_words) >= self.threshold:
                return other_text
        return None

def is_near(a, b, threshold=NEAR_DEFAULT):
    idx = NearIndex(threshold); idx.add(a)
    return idx.find(b) is not None

