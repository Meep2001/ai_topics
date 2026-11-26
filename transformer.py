import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math 

text='I love python'
encoding = tiktoken.get_encoding("cl100k_base")
token_ids = encoding.encode(text)

print(f'Token Ids: {token_ids}')
print(f'Number of token Ids: {len(token_ids)}')

token_ids = torch.tensor(token_ids, dtype=torch.long)
embedding_dim = 8
vocab_size = encoding.n_vocab

print(f'Vocab szie in given encoding: {vocab_size}')

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
x = embedding_layer(token_ids)
print(f'embedding output: {x.shape}')

position_layer = nn.Embedding(
    100,
    embedding_dim
)

positions = torch.arange(len(token_ids))
print(f'Positions: {positions}')

position_embedding = position_layer(positions)
print(f'Position embedding size: {position_embedding.shape}')

x = x + position_embedding
print(f'After adding positional Information: {x.shape}')

query_layer = nn.Linear(8,4,bias=False)
key_layer = nn.Linear(8,4, bias=False)
value_layer = nn.Linear(8,4, bias = False)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

scores = Q @ K.T / math.sqrt(K.shape[-1])
print(f'Scores: {scores}')
seq_len = len(token_ids)

mask = torch.triu(torch.ones(seq_len,seq_len, dtype=torch.bool))

scores = scores.masked_fill(mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
print(f'Proababilties : {weights}')

output = weights @ V

print("\n--- FINAL SHAPES ---")

print("Token IDs:", token_ids.shape)
print("Input embeddings:", x.shape)
print("Query:", Q.shape)
print("Key:", K.shape)
print("Value:", V.shape)
print("Attention scores:", scores.shape)
print("Attention weights:", weights.shape)
print("Output:", output.shape)


#Note:
# Multi head transformer we just use 2 layer of head,Key,
# value and then concatenate the output