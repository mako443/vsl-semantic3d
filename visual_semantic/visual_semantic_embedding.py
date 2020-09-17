import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
import pickle

#TODO/CARE: do a sanity check that the Text-LSTM can capture information

'''
A Visual Semantic Embedding with a pre-trained feature-extractor + NetVLAD as the image-model (https://arxiv.org/pdf/1411.2539.pdf)
'''
class VisualSemanticEmbedding(torch.nn.Module):
    def __init__(self,image_model, known_words, embedding_dim):
        super(VisualSemanticEmbedding, self).__init__()

        self.embedding_dim=embedding_dim

        self.padding_idx=0 #Padding idx is also used for unkown words (there shouldn't be any)
        self.word_dictionary={} #dictionary {word: embedding_index}
        for i_word, word in enumerate(known_words):
            self.word_dictionary[word]=i_word+1 #Index 0 is the padding/unknown index

        #Contrary to the paper, we use initialize the word-embedding from PyTorch
        self.word_embedding=nn.Embedding(len(known_words)+1, self.embedding_dim, padding_idx=self.padding_idx) 
        self.word_embedding.weight.requires_grad_(False) #TODO: train the embedding?

        self.lstm=nn.LSTM(self.embedding_dim,self.embedding_dim)

        #Set the image model for evaluation, no training
        self.image_model=image_model
        self.image_model.requires_grad_(False)
        self.image_model.eval()
        #self.image_dim=image_model.dim*image_model.num_clusters #Output get's flattened during NetVLAD
        self.image_dim=list(image_model.parameters())[-1].shape[0] #For VGG
        assert self.image_dim==4096

        self.W_i=nn.Linear(self.image_dim,self.embedding_dim,bias=True)        

    def forward(self,images,captions):
        if len(images.shape)==3: images=torch.unsqueeze(images,0)
        if type(captions) is str: captions=[captions,]

        assert len(images)==len(captions)
        x=self.encode_images(images)
        v=self.encode_captions(captions)
        return x,v        

    def encode_images(self,images):
        assert len(images.shape)==4 #Expect a batch of images
        q=self.image_model(images)
        x=self.W_i(q)
        return x

    def encode_captions(self,captions):
        word_indices=[ [self.word_dictionary.get(word,self.padding_idx) for word in caption.split()] for caption in captions]
        caption_lengths=[len(w) for w in word_indices]
        batch_size,max_length=len(word_indices), max(caption_lengths)
        padded_indices=np.ones((batch_size,max_length),np.int)*self.padding_idx

        for i,caption_length in enumerate(caption_lengths):
            padded_indices[i,:caption_length]=word_indices[i]
        
        padded_indices=torch.from_numpy(padded_indices)
        if self.is_cuda(): padded_indices=padded_indices.cuda()

        #Packing seems to work, sentences have the same output regardless of padding
        embedded_words=self.word_embedding(padded_indices)
        x=nn.utils.rnn.pack_padded_sequence(embedded_words, torch.tensor(caption_lengths), batch_first=True, enforce_sorted=False)

        if self.is_cuda():
            h=torch.zeros(1,batch_size,self.word_embedding.embedding_dim).cuda()
            c=torch.zeros(1,batch_size,self.word_embedding.embedding_dim).cuda()
        else:
            h=torch.zeros(1,batch_size,self.word_embedding.embedding_dim)
            c=torch.zeros(1,batch_size,self.word_embedding.embedding_dim)

        _,(h,c)=self.lstm(x,(h,c))
        return torch.squeeze(h)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s): #Norming the input (as in paper) is actually not helpful
        im=im/torch.norm(im,dim=1,keepdim=True)
        s=s/torch.norm(s,dim=1,keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im) #Take mean for batch-size stability        