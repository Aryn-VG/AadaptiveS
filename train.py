from args import get_args
from dataloader import dataloader
from dgl.data.utils import load_graphs
import torch
import dgl
from embeding import init_embedding,ATTN
from time_encode import TimeEncode
from decoder import  Decoder
from val_eval import get_current_ts,eval_epoch
from bandi_sampler import bandisampler

if __name__ == '__main__':
    args = get_args()
    args.tasks = 'LP'

    #the path of data
    g = load_graphs(f"wiki_ppc.dgl")[0][0]
    efeat_dim = g.edata['feat'].shape[1]
    device='cuda:0' if torch.cuda.is_available() else 'cpu'

    #initialize
    node_feature = torch.zeros((g.number_of_nodes(), args.memory_dimension))
    g.ndata['feat'] = node_feature
    train_loader, val_loader, test_loader, val_num, test_enum = dataloader(args,g)

    t0=torch.zeros(g.number_of_nodes())
    time_encoder = TimeEncode(args.time_dimension).to(device)

    emb_updater = ATTN(args, time_encoder).to(device)
    decoder=Decoder(args,args.emb_dimension).to(device)
    loss_fcn = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(emb_updater.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
    k = torch.tensor(args.n_degree).float().to(device)
    eta = torch.tensor(args.eta).float().to(device)
    T = torch.tensor(args.T).float().to(device)
    bandi_sampler=bandisampler(eta,T,k)

    #initialize q_ij&weight
    g.ndata['indegree'] = g.in_degrees()
    def init_indegree(edges):
        in_degree=edges.dst['indegree']
        return{'in_degree':in_degree}

    def init_qij(edges):
        prob = torch.div(1.0, edges.dst['indegree'])
        return {'q_ij': prob}
    def init_weight(g):
        weight = torch.ones((g.number_of_edges()))
        g.edata['weight'] = weight
        return g

    g.apply_edges(init_indegree)
    g.edata['timestamp']+=torch.tensor(1)
    eid_of_g = torch.linspace(0, g.number_of_edges() - 1, g.number_of_edges()).long()
    g.edata['eid'] = eid_of_g

    #training
    for i in range(args.n_epoch):
        init_embedding(g)
        g.ndata['last_update'] = t0
        g.apply_edges(init_qij)
        init_weight(g)

        decoder.train()
        time_encoder.train()
        emb_updater.train()
        for batch_id, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            for j in range(args.n_layer):
                blocks[j] = blocks[j].to(device)

            current_ts, pos_ts, num_pos_nodes = get_current_ts(pos_graph, neg_graph)
            pos_graph.ndata['ts'] = current_ts

            blocks,att_map=emb_updater.forward(blocks)

            emb=blocks[-1].dstdata['h']
            if batch_id!=0:
                # backward
                logits, labels = decoder(emb, pos_graph, neg_graph)
                # pred = logits.sigmoid() > 0.5
                # ap = average_precision(logits, labels)
                loss = loss_fcn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #update bandit sampler
            blocks=bandi_sampler.weight_update(blocks,att_map)
            blocks=bandi_sampler.prob_update(blocks)

            with torch.no_grad():
                g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
                g.edata['q_ij'][blocks[-1].edata['eid']] = blocks[-1].edata['q_ij'].cpu()
                g.edata['weight'][blocks[-1].edata['eid']] = blocks[-1].edata['weight'].cpu()
                #g.ndata['h'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
        val_ap, val_auc, val_acc, val_loss ,time_c= eval_epoch(args,g, val_loader, emb_updater, decoder,
                                                        bandi_sampler,loss_fcn, device,val_num)#评估验证集
        print("epoch:%d,loss:%f,ap:%f,time_consume:%f" % (i, val_loss, val_ap,time_c))
