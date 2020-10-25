import torch
import torch.nn as nn
import torch.nn.functional as F


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class CNN_Text(nn.Module):
    """A CNN classifier, originally from 
    https://github.com/Shawn1993/cnn-text-classification-pytorch
    """
    def __init__(self, dictionary, class_num, embed_dim=512, 
        kernel_num=100, kernel_sizes='3,4,5', dropout=0.5, 
        pretrained_embed=None,
    ):
        super(CNN_Text, self).__init__()

        V = len(dictionary)
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = [int(x) for x in kernel_sizes.split(',')]

        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed = Embedding(V, embed_dim, self.padding_idx)
        else:
            self.embed = pretrained_embed
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, net_input, split=None):
        """
        Args:
            net_input (dict): input dict with keys 'src_tokens' and 'src_lengths'


        Returns:
            logits (tensor): shape (B, class_num)
        """

        x = net_input['src_tokens']
        # (N, W, D) <-> (batch_size, seq_len, feature)
        x = self.embed(x)  # (N, W, D)
        
        # if self.args.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        # (N, Co * len(Ks)) <-> (batch_size, kernel_num * len(kernel_size))
        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
