import dgl
import torch
import dgl.function as fn

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    def __init__(self,args,g,eids,block_sampler,g_sampling=None,exclude=None,
                reverse_eids=None,reverse_etypes=None,negative_sampler=None):
        super(TemporalEdgeCollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        self.args=args
    def collate(self,items):
        current_ts=self.g.edata['timestamp'][items[0]]     #only sample edges before current timestamp
        self.block_sampler.ts=current_ts
        neg_pair_graph=None
        if self.negative_sampler is None:
            input_nodes,pair_graph,blocks=self._collate(items)
        else:
            input_nodes,pair_graph,neg_pair_graph,blocks=self._collate_with_negative_sampling(items)
        if self.args.n_layer>1:
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[1].edges())
        frontier=dgl.reverse(self.block_sampler.frontiers[0])
        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts


class ValSampler(dgl.dataloading.BlockSampler):

    def __init__(self, args,fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(len(fanouts))]

    def sample_frontier(self, block_id, g, seed_nodes):
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])
        frontier=g
        self.frontiers[block_id] = frontier
        return frontier

class TrainSampler(dgl.dataloading.BlockSampler):
    '''
    对边采样，返回block的frontier。
    '''

    def __init__(self, args,fanouts, replace=False, return_eids=False):
        super().__init__(args.n_layer, return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(args.n_layer)]
        self.sigmoid=torch.nn.Sigmoid()

    def sample_prob(self, edges):
        timespan = edges.dst['sample_time'] - edges.data['timestamp']

        return {'timespan': timespan}

    def sample_time(self, edges):
        return {'st': edges.data['timestamp']}

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id] if self.fanouts is not None else None
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])
        if self.args.valid_path:
            if block_id != self.args.n_layer - 1:
                g.dstdata['sample_time'] = self.frontiers[block_id + 1].srcdata['sample_time']
                g.apply_edges(self.sample_prob)
                g.remove_edges(torch.where(g.edata['timespan'] < 0)[0])
            g_re=dgl.reverse(g,copy_edata=True,copy_ndata=True)
            g_re.update_all(self.sample_time,fn.max('st','sample_time'))
            g=dgl.reverse(g_re,copy_edata=True,copy_ndata=True)

        if fanout is None:
            frontier = g
        else:
            if block_id == self.args.n_layer - 1:

                if self.args.bandit:
                    frontier = dgl.sampling.sample_neighbors(g,seed_nodes,fanout,prob='q_ij')
                else:
                    frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)

            else:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)

        self.frontiers[block_id] = frontier
        return frontier


def dataloader(args,g):
    origin_num_edges = g.num_edges() // 2

    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    val_eid = torch.arange(int(0.7 * origin_num_edges), int(0.85 * origin_num_edges))
    test_eid = torch.arange(int(0.85 * origin_num_edges), origin_num_edges)
    exclude, reverse_eids = None, None

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    fan_out = [args.n_degree for _ in range(args.n_layer)]
    train_sampler = TrainSampler(args, fanouts=fan_out, return_eids=False)
    val_sampler=ValSampler(args, fanouts=fan_out, return_eids=False)
    train_collator = TemporalEdgeCollator(args,g, train_eid, train_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                          negative_sampler=negative_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_collator.dataset, collate_fn=train_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_worker)
    val_collator = TemporalEdgeCollator(args,g, val_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                        negative_sampler=negative_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_collator.dataset, collate_fn=val_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_worker)
    test_collator = TemporalEdgeCollator(args,g, test_eid, val_sampler, exclude=exclude, reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_collator.dataset, collate_fn=test_collator.collate,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_worker)
    return train_loader, val_loader, test_loader, val_eid.shape[0], test_eid.shape[0]