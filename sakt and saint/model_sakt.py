# modified the implementation from https://github.com/hcnoh/knowledge-tracing-collection-pytorch

import os
import time

import pandas as pd
import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics

class SAKT(Module):
    '''
        This implementation has a reference from: \
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            n: the length of the sequence of the questions or responses
            d: the dimension of the hidden vectors in this model
            num_attn_heads: the number of the attention heads in the \
                multi-head attention module in this model
            dropout: the dropout rate of this model
    '''
    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        kaiming_normal_(self.P)

        self.before_attn_dropout = Dropout(self.dropout)
        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                qry: the query sequence with the size of [batch_size, m], \
                    where the query is the question(KC) what the user wants \
                    to check the knowledge level of

            Returns:
                p: the knowledge level about the query
                attn_weights: the attention weights from the multi-head \
                    attention module
        '''
        ## only predict the last question
        #qry = qry[:,-1].reshape(qry.shape[0],1)
        
        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M = M + P
        
        M = self.before_attn_dropout(M)
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        
        ## predict all
        S = S.permute(1, 0, 2)
        E = E.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        
        # The original code use S+M+E. I checked two other open-source implementations of SAKT,
        # both used S+E.
        # S = self.attn_layer_norm(S + M + E)
        S = self.attn_layer_norm(S + E)
        

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path, for_what = "train", train_last = False, do_test = True
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            tic_epoch = time.time()
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long())
                if train_last:
                    p = p[:,-1]
                    t = rshft[:,-1]
                else:
                    p = torch.masked_select(p, m)
                    t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                if do_test:
                    # only one bacth in test_loader
                    for data in test_loader:
                        q, r, qshft, rshft, m = data


                    self.eval()
                    p, _ = self(q.long(), r.long(), qshft.long())
                    if train_last:
                        # predict the last one
                        p = p[:,-1].detach().cpu()
                        t = rshft[:,-1].detach().cpu()
                    else:
                    ## predict all
                        p = torch.masked_select(p, m).detach().cpu()
                        t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
                    )
                    
                    aucs.append(auc)
                    
                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc
                        
                        
                loss_mean = np.mean(loss_mean)
                toc_epoch = time.time()
                if i % 10 == 0:
                    if do_test:
                        print(
                            "Epoch: {},   AUC: {:5f},   Loss Mean: {:5f}, Time: {:4f}"
                            .format(i, auc, loss_mean, toc_epoch - tic_epoch)
                        )
                    else:
                         print(
                            "Epoch: {},   Loss Mean: {:5f}, Time: {:4f}"
                            .format(i, loss_mean, toc_epoch - tic_epoch)
                        )                       
                
                loss_means.append(loss_mean)
                
       
        if do_test:
            return aucs, loss_means
        else: 
            torch.save(
                self.state_dict(),
                os.path.join(
                    ckpt_path, for_what + "_final_model.ckpt"
                )
            )            
            return loss_means