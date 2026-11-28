from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "You can reset your password through account settings.",
    "Our company provides cloud computing services.",
    "To recover your account, request a password reset link.",
    "You can track your order using the tracking number.",
    "Update your billing information in the payment section."
]

document_embeddings = model.encode(documents)

query = "I forgot my password. How can I access my account?"

query_embedding = model.encode([query])
similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]
ranked_indices = similarities.argsort()[::-1]

for index in ranked_indices:
    print(f"Score: {similarities[index]:.3f}")
    print(documents[index])
    print()

