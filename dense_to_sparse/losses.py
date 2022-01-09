#FROM Sentence-BERT: (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MultipleNegativesRankingLoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MarginMSELoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/util.py) with minimal changes.
#Original License APACHE2

import torch
from torch import nn, Tensor
from typing import Iterable, Dict

def pairwise_dot_score(a: Tensor, b: Tensor):
    """
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class Dense2SparseLoss(nn.Module):
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_rank=1, lambda_sparse_doc=0.001, lambda_sparse_query = 0.01):
        """
        :param model: a dense model to sparsify 
        :param similarity_fct:  Which similarity function to use
        """
        super(Dense2SparseLoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.lambda_rank = lambda_rank
        self.lambda_sparse = lambda_sparse
        self.flops = FLOPS()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        
        # use 'type' to identify query and doc 
        sentence_features[0]["type"] = "query"
        sentence_features[1]["type"] = "doc"
        sentence_features[2]["type"] = "doc"
        
        reps = [self.model(sentence_feature) for sentence_feature in sentence_features]

        dense_reps = [rep["dense_embedding"] for rep in reps]
        dense_query = dense_reps[0]
        dense_pos = dense_reps[1]
        dense_neg = dense_reps[2]

        sparse_from_dense = [rep["sparse_from_dense"] for rep in reps]
        sparse_from_dense_query = sparse_from_dense[0]
        sparse_from_dense_pos = sparse_from_dense[1]
        sparse_from_dense_neg = sparse_from_dense[2]
        
        # dense margin 
        dense_scores_pos = self.similarity_fct(dense_query, dense_pos)
        dense_scores_neg = self.similarity_fct(dense_query, dense_neg)
        dense_margin_pred = dense_scores_pos - dense_scores_neg

        # appromiated sparse margin 
        sparse_from_dense_scores_pos = self.similarity_fct(sparse_from_dense_query, sparse_from_dense_pos)
        sparse_from_dense_scores_neg = self.similarity_fct(sparse_from_dense_query, sparse_from_dense_neg)
        sparse_from_dense_margin_pred = sparse_from_dense_scores_pos - sparse_from_dense_scores_neg
        ranking_loss = self.loss_fct(sparse_from_dense_margin_pred, dense_margin_pred)  

        # sparsity 
        sparsity_query = self.flops(sparse_from_dense_query) 
        sparsity_doc = (self.flops(sparse_from_dense_pos) + self.flops(sparse_from_dense_neg))/2
        sparsity = lambda_sparse_query*sparsity_query + lambda_sparse_doc*expressionsparsity_doc

        print(f"ranking loss {ranking_loss} sparsity_query {sparsity_query} sparsity_doc {sparsity_doc}")
        return self.lambda_rank*ranking_loss + sparsity