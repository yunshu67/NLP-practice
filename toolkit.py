import os
import bcolz
import numpy as np
import pickle
import torch

def word_embedding(embed_path:str,over_writte:bool,special_tk:bool=True,freeze:bool=False):
    ''' return a torch.nn.Embedding layer, utilizing the pre-trained word vector (e.g., Glove), add 'unk' and 'pad'.

    :param embed_path: the path where pre-trained matrix cached (e.g., './glove.6B.300d.txt').
    :param over_writte: force to rewritte the existing matrix.
    :param special_tk: whether adding special token -- 'unk' and 'pad', at position 1 and 0 by default.
    :param freeze: whether trainable.
    :return: embed -> nn.Embedding, weights_matrix -> np.array, word2idx -> dict, embed_dim -> int
    '''
    root_dir = embed_path.rsplit(".",1)[0]+".dat"
    out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
    out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
    if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
        ## process and cache glove ===========================================
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
        with open(os.path.join(embed_path),"rb") as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(float)
                vectors.append(vect)
        vectors = bcolz.carray(vectors[1:].reshape((idx, vect.shape[0])), rootdir=root_dir, mode='w')
        vectors.flush()
        pickle.dump(words, open(out_dir_word, 'wb'))
        pickle.dump(word2idx, open(out_dir_idx, 'wb'))
        print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
        ## =======================================================
    ## load glove
    vectors = bcolz.open(root_dir)[:]
    words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
    word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))
    print("Successfully load Golve from {}, the shape of cached matrix: {}".format(embed_path.rsplit("/",1)[0],vectors.shape))

    word_num, embed_dim = vectors.shape
    word_num += 2  if special_tk else 0  ## e.g., 400002
    embedding_matrix = np.zeros((word_num, embed_dim))
    if special_tk:
        embedding_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim, ))
        embedding_matrix[2:,:] = vectors
        weights_matrix_tensor = torch.FloatTensor(embedding_matrix)
        pad_idx,unk_idx = 0,1
        embed_layer = torch.nn.Embedding(word_num, embed_dim,padding_idx=pad_idx)
        embed_layer.from_pretrained(weights_matrix_tensor,freeze=freeze,padding_idx=pad_idx)
        word2idx = dict([(k,v+2) for k,v in word2idx.items()])
        assert len(word2idx) + 2 == embedding_matrix.shape[0]
    else:
        embedding_matrix[:,:] = vectors
        weights_matrix_tensor = torch.FloatTensor(embedding_matrix)
        embed_layer = torch.nn.Embedding(word_num, embed_dim)
        embed_layer.from_pretrained(weights_matrix_tensor,freeze=freeze)
        assert len(word2idx) == embedding_matrix.shape[0]

    return embed_layer, embedding_matrix, word2idx, embed_dim


