import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def decode(tokens):
    print()
    for token in tokens:
        print(f"{token} --> {encoding.decode([token])}")
        
sentences = [
    "I love Python",
    "I love programming",
    "Artificial Intelligence",
    "Internationalization",
    "I love programming with Python!"
]

for s in sentences:
    decode(encoding.encode(s))



    