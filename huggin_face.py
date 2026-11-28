from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)
print('Ask something')
question = input()
result = generator(
    question,
    max_new_tokens = 150,
    do_sample = True,
    temperature = 0.7
)
print(result)