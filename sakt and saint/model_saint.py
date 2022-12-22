# modified the implementation from https://github.com/Nino-SEGALA/SAINT-pytorch

import os
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import pandas as pd
import numpy as np
import copy


from torch.nn.functional import binary_cross_entropy
from sklearn import metrics

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )
        

class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self , dim_model, heads_en, total_ex ,total_cat, seq_len, dropout):
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.embd_ex =   nn.Embedding( total_ex , embedding_dim = dim_model )                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_model )

        self.multi_en = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_en,  )     # multihead attention
        self.ffn_en = Feed_Forward_block( dim_model )                                            # feedforward block
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )


    def forward(self, in_ex, in_cat, first_block=True):
        
        if torch.cuda.is_available():
            device ='cuda'
        else:
            device = "cpu"  
            
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_cat = self.embd_cat( in_cat )
            in_pos = position_embedding(in_ex.shape[0], self.seq_len, self.dim_model).to(device)
            #combining the embedings
            out = in_ex + in_cat + in_pos                      # (b,n,d)
        else:
            out = in_ex
        
        out = self.dropout(out)
        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape )
        
        #Multihead attention                            
        n,_,_ = out.shape
        skip_out = out 
        out, attn_wt = self.multi_en( out , out , out ,
                                attn_mask=get_mask(seq_len=n).to(device))  # attention mask upper triangular
        out = self.dropout(out)
        out = out + skip_out                                    # skip connection
        out = self.layer_norm1( out )                           # Layer norm

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d) 
        skip_out = out
        out = self.ffn_en( out )
        out = self.dropout(out)
        out = out + skip_out                                    # skip connection
        out = self.layer_norm2( out )                           # Layer norm

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self,dim_model, total_in, heads_de, seq_len, dropout):
        super().__init__()
        self.dim_model  = dim_model
        self.seq_len    = seq_len
        self.embd_in    = nn.Embedding(  total_in , embedding_dim = dim_model )                  #interaction embedding
        self.multi_de1  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = Feed_Forward_block( dim_model )                                         # feed forward layer

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )
        self.layer_norm3 = nn.LayerNorm( dim_model )


    def forward(self, in_in, en_out,first_block=True):
        
        if torch.cuda.is_available():
            device ='cuda'
        else:
            device = "cpu"  
            
        if first_block:
            in_in = self.embd_in( in_in )
            in_pos = position_embedding(in_in.shape[0], self.seq_len, self.dim_model).to(device)
            #combining the embedings
            out = in_in + in_pos                                    # (b,n,d)
        else:
            out = in_in

        out = self.dropout(out)
        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape )

        #Multihead attention M1
        n,_,_ = out.shape
        skip_out = out
        out, attn_wt = self.multi_de1( out , out , out, 
                                     attn_mask=get_mask(seq_len=n).to(device)) # attention mask upper triangular
        out = self.dropout(out)
        out = skip_out + out                                        # skip connection
        out = self.layer_norm1( out )

        #Multihead attention M2
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        skip_out = out
        out, attn_wt = self.multi_de2( out , en_out , en_out,
                                    attn_mask=get_mask(seq_len=n).to(device))  # attention mask upper triangular
        out = out + skip_out
        en_out = self.layer_norm2( en_out )

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        skip_out = out
        out = self.ffn_en( out )                                    
        out = self.dropout(out)
        out = out + skip_out                                        # skip connection
        out = self.layer_norm3( out )                               # Layer norm 

        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    return torch.from_numpy( np.triu(np.ones((seq_len ,seq_len)), k=1).astype('bool'))

def position_encoding(pos, dim_model):
    # Encode one position with sin and cos
    # Attention Is All You Need uses positinal sines, SAINT paper does not specify
    pos_enc = np.zeros(dim_model)
    for i in range(0, dim_model, 2):
        pos_enc[i] = np.sin(pos / (10000 ** (2 * i / dim_model)))
        pos_enc[i + 1] = np.cos(pos / (10000 ** (2 * i / dim_model)))
    return pos_enc


def position_embedding(bs, seq_len, dim_model):
    # Return the position embedding for the whole sequence
    pe_array = np.array([[position_encoding(pos, dim_model) for pos in range(seq_len)]] * bs)
    return torch.from_numpy(pe_array).float()

# dim_model: int.
# Dimension of model ( embeddings, attention, linear layers).
# num_en: int.
# Number of encoder layers.
# num_de: int.
# Number of decoder layers.
# heads_en: int.
# Number of heads in multi-head attention block in each layer of encoder.
# heads_de: int.
# Number of heads in multi-head attention block in each layer of decoder.
# total_ex: int.
# Total number of unique excercise.
# total_cat: int.
# Total number of unique concept categories.
# total_in: int.
# Total number of unique interactions.

class SAINT(nn.Module):
    def __init__(self,dim_model,num_en, num_de ,heads_en, total_ex ,total_cat,total_in,heads_de,seq_len, dropout):
        super().__init__( )
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"  
        
        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones( Encoder_block(dim_model, heads_en , total_ex ,total_cat,seq_len, dropout) , num_en)
        self.decoder = get_clones( Decoder_block(dim_model ,total_in, heads_de,seq_len, dropout)             , num_de)

        self.out = nn.Linear(in_features= dim_model , out_features=1)
    
    def forward(self,in_ex, in_cat,  in_in ):
        '''
            Args:
                in_ex: the question(KC) sequence with the size of [batch_size, n]
                in_cat: the category sequence with the size of [batch_size, n]
                in_in: the response sequence with the size of [batch_size, m]

            Returns:
                in_in: the predicted response
        '''        
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x>=1:
                first_block = False
            in_ex = self.encoder[x]( in_ex, in_cat ,first_block=first_block)
            in_cat = in_ex                                  # passing same output as q,k,v to next encoder block

        
        ## pass through each decoder blocks in sequence
        first_block = True
        for x in range(self.num_de):
            if x>=1:
                first_block = False
            in_in = self.decoder[x]( in_in , en_out= in_ex, first_block=first_block )

        ## Output layer
        in_in = torch.sigmoid( self.out( in_in ) ).squeeze()
        return in_in
    
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

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"  

        for i in range(1, num_epochs + 1):
            tic_epoch = time.time()
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                # create a category sequence with only 0s
                cat = torch.zeros(q.shape, dtype=torch.int32, device = device)
                self.train()

                p = self(q.long(), cat, r.long())
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

                    # create a category sequence with only 0s
                    cat = torch.zeros(q.shape, dtype=torch.int32, device = device)

                    self.eval()
                    p = self(q.long(), cat, r.long())
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
                if i % 20 == 0:
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