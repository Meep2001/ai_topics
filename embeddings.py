import numpy as np

king = np.array([0.8, 0.9, 0.2])
queen = np.array([0.8, 0.85, 0.25])
apple = np.array([0.1, 0.2, 0.9])

def cosine_similarity(a,b):
    dot_product = np.dot(a,b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return (dot_product/ (magnitude_a*magnitude_b))

print(cosine_similarity(king,queen))
print(cosine_similarity(apple,queen))
