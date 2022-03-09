import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10,hidden_size=20,num_layers=2,batch_first=True)
inputs = torch.randn(3, 5, 10)  # batch_size,seq_length,embedding_dim
h0 = torch.randn(2, 3, 20)      # D*num_layers, batch_size, hidden_size
output, hn = rnn(inputs, h0)

print(output)                   # batch_size,seq_length,D*hidden_size
print(hn)                       # D*num_layers,batch_size,hidden_size
print(output.size())
print(hn.size())