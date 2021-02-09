import torch
import torch.nn as nn
import dgl

def init_embedding(g):
    g.ndata['h']=g.ndata['feat']

class ATTN(nn.Module):
    def __init__(self,args,time_encoder):
        super().__init__()
        self.n_layer = args.n_layer
        self.h_dimension = args.emb_dimension
        self.attnlayers = nn.ModuleList()
        self.mergelayers=nn.ModuleList()
        self.edge_feat_dim=args.edge_feat_dim
        self.n_head = args.n_head
        self.time_dim = args.time_dimension
        self.node_feat_dim = args.node_feat_dim
        self.dropout = args.dropout
        self.args=args
        self.time_encoder=time_encoder
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'

        self.query_dim = self.node_feat_dim + self.time_dim
        self.key_dim = self.node_feat_dim + self.time_dim + self.edge_feat_dim
        for i in range(0, self.n_layer):
            self.attnlayers.append(nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=self.n_head,
                                                       dropout=self.dropout).to(self.device))
            self.mergelayers.append(MergeLayer(self.query_dim, self.node_feat_dim, self.node_feat_dim, self.h_dimension).to(self.device))

    def C_compute(self,edges):
        te_C = self.time_encoder(edges.data['timestamp'] - edges.src['last_update'])

        eid=edges.data[dgl.EID].view(-1,1).float()
        q_ij=edges.data['q_ij'].view(-1,1).float()
        C = torch.cat([edges.src['h'], edges.data['feat'], te_C,eid,q_ij], dim=1)
        #print(C.size())
        return {'C': C}

    def h_compute(self,nodes):
        C=nodes.mailbox['C'][:,:,:-2]
        eid=nodes.mailbox['C'][:,:,-2].view(-1).detach().long()
        q_ij=nodes.mailbox['C'][:,:,-1].view(-1).detach()
        #print(q_ij,eid)
        C=C.permute([1,0,2])#convert to [num_node,num_neighbor,feat_dim]
        key = C.to(self.device)

        te_q=self.time_encoder(torch.zeros(nodes.batch_size()).to(self.device))
        query = torch.cat([nodes.data['h'], te_q],dim=1).unsqueeze(dim=0)

        h_before,att= self.attnlayers[self.l](query=query, key=key, value=key)

        qij_sum=torch.sum(q_ij)
        att_sum=torch.sum(att).view(-1)
        att=qij_sum*att/att_sum
        if self.l==self.n_layer-1:
            self.attn_map[eid]=att.view(-1)
        #print(self.attn_map)
        h_before=h_before.squeeze(0)

        h= self.mergelayers[self.l](nodes.data['h'], h_before)
        return {'h':h}

    def forward(self, blocks):
        for l in range(self.n_layer):
            self.l = l
            if self.l==self.n_layer-1:
                attn_map=torch.ones((blocks[self.l].number_of_edges())).to(self.device)
                self.attn_map=-attn_map
            blocks[l].update_all(self.C_compute,self.h_compute)
            if l!=self.n_layer-1:
                blocks[l+1].srcdata['h']=blocks[l].dstdata['h']
        return blocks,self.attn_map

class MergeLayer(torch.nn.Module):
    '''(dim1+dim2)->dim3->dim4'''

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

