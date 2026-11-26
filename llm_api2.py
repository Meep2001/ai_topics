from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

result = generator(
    'Artificial Intelligence is',
    max_new_tokens = 50,
    do_sample = True,
    temperature = 0.7
)

print(result)