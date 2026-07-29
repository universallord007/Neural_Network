import regex as re
import unicodedata
# s = "Konnichiwa (こんにちは)"
# # print([ord(x) for x in s])
# print(list("Konnichiwa (こんにちは)".encode("utf-8")))

text = "The unexceptionable synchronization of their ultra-modernized cryptocurrency algorithms unexpectedly failed during the multi-dimensional simulation, causing 10,450 automated supercomputers to repeatedly recalculate the unquantifiable data—unbelievable, right? 🤔 Failure-driven experimentation ultimately builds unstoppable, unshakeable stability."

tokens = text.encode("utf-8")
tokens = list(map(int,tokens))
# print()
# print(text)
# print(len(text))
# print()
# print(tokens)
# print(len(tokens))
# def get_stats(ids):
#     counts = {}
#     for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
#         counts[pair] = counts.get(pair, 0) + 1
#     return counts

# stats = get_stats(tokens)
# # print(stats)
# print(tokens)
# print(sorted(((k,v) for v,k in stats.items()),reverse=True))


# This calculates the number of times the pair has occcured and keeps the count as a record
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# Replaces the most occuring or the repetitive pair with some other value hence the no of tokens get increased by that factor
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# ----------------------------------------

vocab_size = 276          # desired final vocabulary size
num_merges = vocab_size - 256

ids = list(tokens)        # copy so we don't modify original

merges = {}               # (int, int) -> int

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i

    print(f"merging {pair} into a new token {idx}")

    ids = merge(ids, pair, idx)
    merges[pair] = idx

# print(f"length of tokens is {len(tokens)}")
# print(f"length of ids is {len(ids)}")
# print(f"The comparison ratio is {len(tokens)/len(ids)}")


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))

    while len(tokens)>=2:
        stats = get_stats(tokens)

        pair = min(stats, key=lambda p: merges.get(p, float("inf")))

        if pair not in merges:
            break

        idx = merges[pair]
        tokens = merge(tokens, pair, idx)

    return tokens


# print(encode("h"))
# print(decode([104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]))

gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d
    | ?\p{L}+
    | ?\p{N}+
    | ?[^\s\p{L}\p{N}]+
    | \s+(?!\S)
    | \s+"""
)

print(re.findall(gpt2pat, "Hello world"))
