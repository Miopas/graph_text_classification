import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphConvolution


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_out * nheads, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super(GCN, self).__init__()

        self.gc = GraphConvolution(n_in, n_out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class MultiLayerGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, layer_type):
        super(MultiLayerGNN, self).__init__()
        self.dropout = dropout

        if nlayers == 1:
            self.gnn_layers = [GraphConvolution(nfeat, nclass)]
        else:
            #self.gnn_layers = [GraphConvolution(nfeat, nhid)]
            #self.gnn_layers += [GraphConvolution(nhid, nhid) for _ in range(nlayers-2)]
            #self.gnn_layers.append(GraphConvolution(nhid, nclass))
            self.gnn_layers = [GraphConvolution(nfeat, nhid), GraphConvolution(nhid, nclass)]

        for i, gnn in enumerate(self.gnn_layers):
            self.add_module('gnn_{}'.format(i), gnn)

    def forward(self, x, adj):
        x = self.gnn_layers[0](x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gnn_layers[1](x, adj)
        #for layer in self.gnn_layers:
        #    x = layer(x, adj)
        #    x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class MultiLayerResGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, layer_type):
        super(MultiLayerResGCN, self).__init__()
        self.dropout = dropout

        self.gnn_layers = [GraphConvolution(nfeat, nhid)]
        self.gnn_layers += [GraphConvolution(nhid, nhid) for _ in range(nlayers-1)]

        self.proj = nn.Linear(nfeat, nclass)

        for i, gnn in enumerate(self.gnn_layers):
            self.add_module('gnn_{}'.format(i), gnn)

        self.classifier = GraphConvolution(nhid, nclass)

    def forward(self, x, adj):
        raw = x

        x = F.dropout(x, self.dropout, training=self.training)
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier(x, adj)
        x = torch.add(x, self.proj(raw))
        return F.log_softmax(x, dim=1)
