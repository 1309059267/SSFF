import torch
from torch import nn
import itertools
from transformers import BertModel


def get_distance(pair_indices, dep_dis_matrix, bucket_bins):
    position_dis = []
    dependency_dis = []
    for batch, quaters in enumerate(pair_indices):
        rel_dis = []
        dep_dis = []
        for pair_index in quaters:
            abs_dis = min(abs(pair_index[0]-pair_index[3]), abs(pair_index[1]-pair_index[2]))
            min_value = [ix for ix, v in enumerate(bucket_bins) if abs_dis >= v][-1]
            rel_dis.append(min_value)
            row_start = pair_index[0]
            row_end = pair_index[1]+1
            cloumn_start = pair_index[2]
            cloumn_end = pair_index[3]+1
            matrix = dep_dis_matrix[batch][row_start:row_end, cloumn_start:cloumn_end]
            min_value = min(matrix.reshape(1,-1).squeeze(0))
            dep_dis.append(min_value)
        position_dis.append(rel_dis)
        dependency_dis.append(dep_dis)
    return position_dis, dependency_dis

def get_target_opinion_pairs(span_features, span_indices, target_indices, opinion_indices):
    
    batch_size = span_features.size(0)
    device = span_features.device

    # candidate_indices :[(a,b,c,d)]
    # relation_indices :[(span1,span2)]
    candidate_indices, relation_indices = [], []
    for batch in range(batch_size):
        pairs = list(itertools.product(target_indices[batch].cpu().tolist(), opinion_indices[batch].cpu().tolist()))
        relation_indices.append(pairs)
        candidate_ind = []
        for pair in pairs:
            if len(span_indices[pair[0]]) == 2:
                a, b = span_indices[pair[0]]
            else:
                a = b = span_indices[pair[0]][0]
            if len(span_indices[pair[1]]) == 2:
                c, d = span_indices[pair[1]]
            else:
                c = d = span_indices[pair[1]][0]
            candidate_ind.append([a, b, c, d])
        candidate_indices.append(candidate_ind)

    candidate_pool = []
    for batch in range(batch_size):
        relations = [torch.cat((span_features[batch, c[0], :], span_features[batch, c[1], :]), dim=0) for c in relation_indices[batch]]
        candidate_pool.append(torch.stack(relations))

    return torch.stack(candidate_pool), candidate_indices

def SpanRepresention(args, features, pos_features, batch_max_len):
    indices = torch.arange(0, batch_max_len, dtype=int, device=args.device)
    span_indices = []
    max_window = min(args.max_span_len, batch_max_len)
    for window in range(1, max_window + 1):
        if window == 1:
            span_index = [[idx.item()] for idx in indices]
        else:
            res = indices.unfold(0, window, 1)
            span_index = [[idx[0].item(), idx[-1].item()] for idx in res]
        span_indices.extend(span_index)
    span_features = []
    span_pos = []
    for idx in span_indices:
        if len(idx) == 1:
            span_features.append(features[:, idx, :])
            span_pos.append(pos_features[:, idx, :])
        else:
            span_features.append(features[:, idx[0]:idx[1]+1, :].mean(dim=1, keepdim=True))
            span_pos.append(pos_features[:, idx[0]:idx[1]+1, :].mean(dim=1, keepdim=True))
    span_features = torch.cat(span_features, dim=1)
    span_pos = torch.cat(span_pos, dim=1)
    return span_features, span_pos, span_indices

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, text, adj):
        hidden = self.linear(text)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        output = self.dropout(output)
        return output

class PrunedTargetOpinion:
    """
    For a sentence X
    of length n, the number of enumerated spans is O(n^2), while the number of possible pairs between
    all opinion and target candidate spans is O(n^4) at the later stage (i.e., the triplet module). As such,
    it is not computationally practical to consider all possible pairwise interactions when using a span-based
    approach. Previous works (Luan et al., 2019; Wadden  et al., 2019) employ a pruning strategy to
    reduce the number of spans, but they only prune the spans to a single pool which is a mix of different
    mention types. This strategy does not fully consider
    """

    def __init__(self):
        pass

    def __call__(self, spans_probability, nz):
        target_indices = torch.topk(spans_probability[:, :, 1], nz, dim=-1).indices
        opinion_indices = torch.topk(spans_probability[:, :, 2], nz, dim=-1).indices
        return target_indices, opinion_indices

class SpanRepresentation(nn.Module):
    """
    We define each span representation si,j ∈ S as:
            si,j =   [hi; hj ; f_width(i, j)] if BiLSTM
                     [xi; xj ; f_width(i, j)] if BERT
    where f_width(i, j) produces a trainable feature embedding representing the span width (i.e., j −i+ 1)
    Besides the concatenation of the start token, end token, and width representations,the span representation si,j
    can also be formed by max-pooling or mean-pooling across all token representations of the span from position i to j.
    The experimental results can be found in the ablation study.
    """

    def __init__(self, span_width_embedding_dim, span_maximum_length):
        super(SpanRepresentation, self).__init__()
        self.span_maximum_length = span_maximum_length
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 80]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), span_width_embedding_dim)

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.span_width_embedding(torch.LongTensor([em]).to(device))

    def forward(self, features, pos_features, batch_max_seq_len):
        """
        [[2, 5], [0, 1], [1, 2], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]
        :param x: batch * len * dim
        :param term_cat:
        :return:
        """
        batch_size, sequence_length, _ = features.size()
        device = features.device

        len_arrange = torch.arange(0, batch_max_seq_len, device=device)
        span_indices = []

        max_window = min(batch_max_seq_len, self.span_maximum_length)

        for window in range(1, max_window + 1):
            if window == 1:
                indics = [(x.item(), x.item()) for x in len_arrange]
            else:
                res = len_arrange.unfold(0, window, 1)
                indics = [(idx[0].item(), idx[-1].item()) for idx in res]
            span_indices.extend(indics)

        spans_features = [torch.cat((features[:, s[0], :], features[:, s[1], :], pos_features[:, s[0], :], pos_features[:, s[1], :], 
            self.bucket_embedding(abs(s[1] - s[0] + 1), device).repeat((batch_size, 1)).to(device)), dim=1) for s in span_indices]

        return torch.stack(spans_features, dim=1), span_indices

class TargetOpinionPairRepresentation(nn.Module):
    """
    Target Opinion Pair Representation We obtain the target-opinion pair representation by coupling each target candidate representation
    St_a,b ∈ St with each opinion candidate representation So_a,b ∈ So:
        G(St_a,b,So_c,d) = [St_a,b; So_c,d; f_distance(a, b, c, d)] (5)
    where f_distance(a, b, c, d) produces a trainable feature embedding based on the distance (i.e., min(|b − c|, |a − d|)) between the target
    and opinion span
    """

    def __init__(self, dis_vocab_len, pair_width_dim, dep_dis_dim):
        super(TargetOpinionPairRepresentation, self).__init__()
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 80]
        self.rel_dis_embedding = nn.Embedding(len(self.bucket_bins), pair_width_dim)
        self.dep_dis_embedding = nn.Embedding(dis_vocab_len, dep_dis_dim)

    def dis_embedding(self, a, b, c, d, dep_dis_matrix, device):
        rel_dis = min(abs(b - c), abs(a - d))
        rel_dis_id = [ix for ix, v in enumerate(self.bucket_bins) if rel_dis >= v][-1]
        rel_dis_emd = self.rel_dis_embedding(torch.LongTensor([rel_dis_id]).to(device))
        matrix = dep_dis_matrix[a:b+1, c:d+1]
        dep_dis_id = min(matrix.reshape(1,-1).squeeze(0))
        dep_dis_emd = self.dep_dis_embedding(torch.LongTensor([dep_dis_id]).to(device))
        return rel_dis_emd.squeeze(0), dep_dis_emd.squeeze(0)

    def forward(self, spans, span_indices, target_indices, opinion_indices, dep_dis_matrix):
        """

        :param spans:
        :param span_indices:
        :param target_indices:
        :type
        :param opinion_indices:
        :return:
            candidate_indices :
                List[List[Tuple(a,b,c,d)]]
            relation_indices :
                List[List[Tuple(span1,span2)]]
        """
        batch_size = spans.size(0)
        device = spans.device

        # candidate_indices :[(a,b,c,d)]
        # relation_indices :[(span1,span2)]
        candidate_indices, relation_indices = [], []
        for batch in range(batch_size):
            pairs = list(itertools.product(target_indices[batch].cpu().tolist(), opinion_indices[batch].cpu().tolist()))
            relation_indices.append(pairs)
            candidate_ind = []
            for pair in pairs:
                a, b = span_indices[pair[0]]
                c, d = span_indices[pair[1]]
                candidate_ind.append([a, b, c, d])
            candidate_indices.append(candidate_ind)

        candidate_pool = []
        for batch in range(batch_size):
            relations = [torch.cat((spans[batch, c[0], :], spans[batch, c[1], :],
            *self.dis_embedding(*span_indices[c[0]], *span_indices[c[1]], dep_dis_matrix[batch], device)), dim=0) for c in relation_indices[batch]]
            candidate_pool.append(torch.stack(relations))
        return torch.stack(candidate_pool), candidate_indices

class STABSA(nn.Module):
    
    def __init__(
        self,
        args,
        pos_vocab_len,
        dis_vocab_len,
        span_pruned_threshold: "int"=0.5
    ) -> None:
        super(STABSA, self).__init__()
        self.args = args
        self.span_pruned_threshold = span_pruned_threshold
        self.bert = BertModel.from_pretrained(args.bert_model_dir)
        self.pos_embed = nn.Embedding(pos_vocab_len, args.pos_dim)
        self.span_representation = SpanRepresentation(args.span_width_dim, args.max_span_len)
        self.pruned_target_opinion = PrunedTargetOpinion()
        self.target_opinion_pair_representation = TargetOpinionPairRepresentation(dis_vocab_len, args.pair_width_dim, args.dep_dis_dim)
        self.gcn1 = GraphConvolution(args.bert_dim, args.bert_dim)
        self.gcn2 = GraphConvolution(args.bert_dim, args.bert_dim)
        span_dim = args.bert_dim*2 +args.pos_dim*2 + args.span_width_dim
        self.span_ffnn = torch.nn.Sequential(
            nn.Linear(span_dim, args.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.hidden_dim, args.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.hidden_dim, args.span_label_class, bias=True),
        )
        pair_dim = span_dim*2 + args.pair_width_dim + args.dep_dis_dim
        self.pair_ffnn = torch.nn.Sequential(
            nn.Linear(pair_dim, args.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.hidden_dim, args.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(args.hidden_dim, args.senti_label_class, bias=True),
        )
        

    def forward(self, bert_input_id, attention_mask, token_type_ids, \
                bert_subword_mask_matrix, first_subword_seq, pos_id, sentence_len, word_len, dep_dis):

        batch_max_len = max(sentence_len)
        bert_output = self.bert(bert_input_id, attention_mask, token_type_ids)
        pool_output = bert_output.pooler_output.unsqueeze(1)
        # 将bert切分的多个子词平均池化，重构特征矩阵
        features = bert_output.last_hidden_state
        features = torch.matmul(bert_subword_mask_matrix, features)
        features = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(features, first_subword_seq)])
        features = features / word_len.unsqueeze(2)
        
        # 利用GNN重编码features
        # gcn_features = self.relu(self.gcn1(features, adj))
        # gcn_features = self.relu(self.gcn2(gcn_features, adj))


        # 根据span的特征及词性预测span类别
        pos_features = self.pos_embed(pos_id)
        span_features, span_indices = self.span_representation(features, pos_features, batch_max_len)
        span_probability = self.span_ffnn(span_features)
        
        # 选取预测概率最大的nz个方面词和观点词
        nz = int(batch_max_len * self.span_pruned_threshold)
        target_indices, opinion_indices = self.pruned_target_opinion(span_probability, nz)
        
        # 将预测的target和opinion进行配对,根据语义特征预测情感极性概率分布
        pair_features, pair_indices = self.target_opinion_pair_representation(span_features, span_indices, target_indices, opinion_indices, dep_dis)
        pair_probability = self.pair_ffnn(pair_features)

        return span_probability, span_indices, pair_probability, pair_indices
