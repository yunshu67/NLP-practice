from toolkit import *

embed_layer, weights_matrix, word2idx, embed_dim = word_embedding('./glove.6B.50d.txt', True, freeze=True)
word = 'hello'
idx = word2idx[word]
embedding = weights_matrix[idx]

print(idx)
print(embedding)
print(type(embedding))


data = torch.tensor([[1,2],[6,7]])
output= embed_layer(data)
print(output)