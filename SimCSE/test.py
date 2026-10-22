"""SimCSE feature tour.  Run from inside the SimCSE folder with:  python test.py

Tip: send the INFO log spam to /dev/null if it's still noisy:
    python test.py 2>/dev/null
"""
import logging
logging.disable(logging.INFO)  # hide the HTTP request lines

import numpy as np
from simcse import SimCSE

MODEL_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"
RUN_UNSUP_COMPARISON = False  # True downloads a second ~440 MB model

model = SimCSE(MODEL_NAME)


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# ---------------------------------------------------------------- 1
section("1. Similarity of a single pair (returns a float)")
a = "A man is playing a guitar."
b = "Someone is strumming an acoustic guitar."
print(f"{a!r}\n{b!r}\n-> {model.similarity(a, b):.3f}")


# ---------------------------------------------------------------- 2
section("2. One query vs many candidates, ranked")
query = "How do I reset my password?"
candidates = [
    "I forgot my login credentials and need to change them.",
    "What is the best pizza in New York?",
    "Steps to recover access to your account.",
    "The stock market closed higher today.",
    "Click 'Forgot password' to receive a reset link.",
]
scores = np.asarray(model.similarity(query, candidates)).ravel()
for idx in np.argsort(-scores):
    print(f"{scores[idx]:.3f}  {candidates[idx]}")


# ---------------------------------------------------------------- 3
section("3. Many-vs-many similarity matrix with labels")
sents_a = ["A woman is reading.", "A man is playing a guitar.", "Kids are playing soccer."]
sents_b = ["He plays guitar.", "A woman is making a photo.", "Children kick a ball outside."]
matrix = np.asarray(model.similarity(sents_a, sents_b))
print("rows = sents_a, cols = sents_b")
for i, row in enumerate(matrix):
    best = int(np.argmax(row))
    print(f"{sents_a[i]!r:35} best match -> {sents_b[best]!r} ({row[best]:.3f})")


# ---------------------------------------------------------------- 4
section("4. Raw embeddings with encode()")
texts = ["The cat sits on the mat.", "A feline rests on a rug."]
emb_t = model.encode(texts)                      # torch tensor by default
emb_np = model.encode(texts, return_numpy=True)  # numpy array
print("type (default):", type(emb_t).__name__, "| shape:", tuple(emb_t.shape))
print("type (numpy)  :", type(emb_np).__name__, "| shape:", emb_np.shape)
print("vector norms (unit-normalized by default):", np.linalg.norm(emb_np, axis=1).round(3))

raw = model.encode(texts, return_numpy=True, normalize_to_unit=False)
print("norms with normalize_to_unit=False:", np.linalg.norm(raw, axis=1).round(2))

# similarity() is just cosine similarity of these vectors
manual = float(emb_np[0] @ emb_np[1])
print(f"manual cosine: {manual:.4f}  vs  model.similarity: {model.similarity(*texts):.4f}")


# ---------------------------------------------------------------- 5
section("5. Semantic search: build_index() + search()")
corpus = [
    "A man is playing a guitar.",
    "A woman is slicing an onion.",
    "Two dogs are running through a field.",
    "A chef is chopping vegetables in a kitchen.",
    "A musician performs on stage with an electric guitar.",
    "A puppy chases a ball in the park.",
    "The stock market fell sharply on Monday.",
    "Investors worry about rising interest rates.",
    "A child is riding a bicycle down the street.",
    "Someone is cutting up an onion.",
]
model.build_index(corpus)  # uses faiss automatically if it's installed; brute force otherwise

for q in ["Someone is making dinner", "Pets playing outside", "Financial news"]:
    print(f"\nQuery: {q!r}")
    for sent, score in model.search(q, top_k=3, threshold=0.3):
        print(f"  {score:.3f}  {sent}")

print("\nBatch search (list of queries -> list of result lists):")
batch = model.search(["live music", "bike"], top_k=2, threshold=0.0)
for q, hits in zip(["live music", "bike"], batch):
    print(f"  {q!r}: {[(s, round(float(sc), 3)) for s, sc in hits]}")

print("\nAdd new documents to the index without rebuilding:")
model.add_to_index(["A cyclist races uphill during a competition."])
print(model.search("bike race", top_k=2, threshold=0.0))


# ---------------------------------------------------------------- 6
section("6. Near-duplicate detection")
reviews = [
    "The battery life on this phone is amazing.",
    "This phone has amazing battery life.",
    "Shipping took two weeks, which was too slow.",
    "The delivery was very slow, it took 14 days.",
    "The screen is bright and sharp.",
    "Great phone, the battery lasts forever.",
]
sim = np.asarray(model.similarity(reviews, reviews))
DUP_THRESHOLD = 0.75
print(f"pairs with similarity >= {DUP_THRESHOLD}:")
for i in range(len(reviews)):
    for j in range(i + 1, len(reviews)):
        if sim[i, j] >= DUP_THRESHOLD:
            print(f"  {sim[i, j]:.3f}  {reviews[i]!r}  <->  {reviews[j]!r}")


# ---------------------------------------------------------------- 7
section("7. Clustering embeddings with scikit-learn KMeans")
from sklearn.cluster import KMeans

docs = [
    "The team won the championship game last night.",
    "The striker scored twice in the second half.",
    "Her new album tops the music charts.",
    "The band announced a world tour this summer.",
    "The senate passed the budget bill on Tuesday.",
    "The president signed the new trade agreement.",
    "The goalkeeper made a spectacular save.",
    "The singer released a surprise single.",
    "Lawmakers debated the proposed tax reform.",
]
vecs = model.encode(docs, return_numpy=True)
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(vecs)
for cluster in sorted(set(labels)):
    print(f"\nCluster {cluster}:")
    for d, l in zip(docs, labels):
        if l == cluster:
            print(f"  - {d}")


# ---------------------------------------------------------------- 8
section("8. Edge cases: negation, word order, unrelated text")
pairs = [
    ("The movie was great.", "The movie was terrible."),          # antonyms
    ("The dog bit the man.", "The man bit the dog."),             # word order
    ("I love pizza.", "Quantum chromodynamics is hard."),         # unrelated
    ("A big dog.", "A large canine."),                            # paraphrase
    ("The bank approved my loan.", "We sat on the river bank."),  # polysemy
]
for x, y in pairs:
    print(f"{model.similarity(x, y):.3f}  {x!r} vs {y!r}")


# ---------------------------------------------------------------- 9
section("9. Batch size and speed")
import time

many = [f"This is test sentence number {i} about topic {i % 7}." for i in range(500)]
for bs in (8, 64):
    t0 = time.time()
    model.encode(many, batch_size=bs)
    print(f"batch_size={bs:3d}: {time.time() - t0:.2f}s for {len(many)} sentences")


# ---------------------------------------------------------------- 10
if RUN_UNSUP_COMPARISON:
    section("10. Supervised vs unsupervised SimCSE")
    unsup = SimCSE("princeton-nlp/unsup-simcse-bert-base-uncased")
    for x, y in pairs[:4]:
        print(f"sup={model.similarity(x, y):.3f}  unsup={unsup.similarity(x, y):.3f}  {x!r} vs {y!r}")

print("\nDone.")