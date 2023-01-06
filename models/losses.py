import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F

def cos_dist(x, y):
    """cosine distance function"""
    return 1-F.cosine_similarity(x, y)

def cos_cdist(x, y):
    """all pairs cosine distance function
    x: batch_size x emb_dim
    y: batch_size x emb_dim"""
    x /= torch.sqrt((x*x).sum(axis=-1)).unsqueeze(axis=-1)
    y /= torch.sqrt((y*y).sum(axis=-1)).unsqueeze(axis=-1)
    return 1 - (x @ y.T) # all pairs cosine-distance

def cos_csim(x, y):
    """all pairs cosine similarity function
    x: batch_size x emb_dim
    y: batch_size x emb_dim"""
    x_norm = x.norm(dim=-1, p=2).unsqueeze(dim=-1)
    y_norm = y.norm(dim=-1, p=2).unsqueeze(dim=-1)
    return (x/x_norm @ (y/y_norm).T)

def triplet_margin_with_distance_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, *,
                                      distance_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                                      margin: float = 1.0, swap: bool = False, reduction: str = "mean") -> torch.Tensor:
    """See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details."""
    # if torch.jit.is_scripting():
    #     raise NotImplementedError(
    #         "F.triplet_margin_with_distance_loss does not support JIT scripting: "
    #         "functions requiring Callables cannot be scripted."
    #     )
    # if F.has_torch_function_variadic(anchor, positive, negative):
    #     return F.handle_torch_function(
    #         triplet_margin_with_distance_loss,
    #         (anchor, positive, negative),
    #         anchor,
    #         positive,
    #         negative,
    #         distance_function=distance_function,
    #         margin=margin,
    #         swap=swap,
    #         reduction=reduction,
    #     )
    distance_function = distance_function if distance_function is not None else pairwise_distance
    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)
    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)
    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)
    if reduction == "mean":
        return output.mean()
    elif reduction == "none":
        return output.sum()
    else: return output

# base loss function class
class _Loss(nn.Module):
    reduction: str
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = nn._Reduction.legacy_get_string(size_average, reduce)
        else: self.reduction = reduction
            
# triplet margin loss with custom distance function
class TripletMarginWithDistanceLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given input
    tensors :math:`a`, :math:`p`, and :math:`n` (representing anchor,
    positive, and negative examples, respectively), and a nonnegative,
    real-valued function ("distance function") used to compute the relationship
    between the anchor and positive example ("positive distance") and the
    anchor and negative example ("negative distance").

    The unreduced loss (i.e., with :attr:`reduction` set to ``'none'``)
    can be described as:

    .. math::
        \ell(a, p, n) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_i = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    where :math:`N` is the batch size; :math:`d` is a nonnegative, real-valued function
    quantifying the closeness of two tensors, referred to as the :attr:`distance_function`;
    and :math:`margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to
    be 0.  The input tensors have :math:`N` elements each and can be of any shape
    that the distance function can handle.

    If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    See also :class:`~torch.nn.TripletMarginLoss`, which computes the triplet
    loss for input tensors using the :math:`l_p` distance as the distance function.

    Args:
        distance_function (Callable, optional): A nonnegative, real-valued function that
            quantifies the closeness of two tensors. If not specified,
            `nn.PairwiseDistance` will be used.  Default: ``None``
        margin (float, optional): A nonnegative margin representing the minimum difference
            between the positive and negative distances required for the loss to be 0. Larger
            margins penalize cases where the negative examples are not distant enough from the
            anchors, relative to the positives. Default: :math:`1`.
        swap (bool, optional): Whether to use the distance swap described in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. If True, and if the positive example is closer to the
            negative example than the anchor is, swaps the positive example and the anchor in
            the loss computation. Default: ``False``.
        reduction (str, optional): Specifies the (optional) reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``


    Shape:
        - Input: :math:`(N, *)` where :math:`*` represents any number of additional dimensions
          as supported by the distance function.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar
          otherwise.

    Examples::

    >>> # Initialize embeddings
    >>> embedding = nn.Embedding(1000, 128)
    >>> anchor_ids = torch.randint(0, 1000, (1,))
    >>> positive_ids = torch.randint(0, 1000, (1,))
    >>> negative_ids = torch.randint(0, 1000, (1,))
    >>> anchor = embedding(anchor_ids)
    >>> positive = embedding(positive_ids)
    >>> negative = embedding(negative_ids)
    >>>
    >>> # Built-in Distance Function
    >>> triplet_loss = \
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # Custom Distance Function
    >>> def l_infinity(x1, x2):
    >>>     return torch.max(torch.abs(x1 - x2), dim=1).values
    >>>
    >>> # xdoctest: +SKIP("FIXME: Would call backwards a second time")
    >>> triplet_loss = (
    >>>     nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # Custom Distance Function (Lambda)
    >>> triplet_loss = (
    >>>     nn.TripletMarginWithDistanceLoss(
    >>>         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    Reference:
        V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'swap', 'reduction']
    margin: float
    swap: bool

    def __init__(self, *, distance_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 margin: float = 1.0, swap: bool = False, reduction: str = 'mean'):
        super(TripletMarginWithDistanceLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.distance_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = \
            distance_function if distance_function is not None else nn.PairwiseDistance()
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return triplet_margin_with_distance_loss(anchor, positive, negative,
                                                 distance_function=self.distance_function,
                                                 margin=self.margin, swap=self.swap, reduction=self.reduction)
# def scl_loss(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor, 
#              device: str="cuda:0", lamb: float=1, margin=1):
#     """selectively contrastive triplet loss:
#     lamb: like in original paper is 1"""
#     S_ap = F.cosine_similarity(a, p)
#     S_an = F.cosine_similarity(a, n)
#     s_ap = torch.diag(a @ p.T)
#     s_an = torch.diag(a @ n.T)
#     mask = (s_ap < s_an).to(device) # print(mask)
#     soft_neg_loss = (~mask)*torch.clamp(s_an-s_ap+margin, min=0)
#     hard_neg_loss = ((mask)*(S_ap+1))
#     # print(soft_neg_loss.mean())
#     # print(hard_neg_loss.mean())
#     return soft_neg_loss + lamb*hard_neg_loss
def scl_loss(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor, 
             loss_fn, device: str="cuda:0", lamb: float=1):
    """selectively contrastive triplet loss:
    lamb: like in original paper is 1"""
    # d_ap = F.pairwise_distance(a, p)
    # d_an = F.pairwise_distance(a, n)
    # S_an = F.cosine_similarity(a, n)
    # mask = (d_ap < d_an).to(device) # print(mask)
    # soft_neg_loss = ((~mask)*loss_fn(a, p, n))
    # hard_neg_loss = ((mask)*(S_an+1))
    # print(soft_neg_loss.mean())
    # print(hard_neg_loss.mean())
    return loss_fn(a, p, n) # soft_neg_loss + lamb*hard_neg_loss
  
# test main method. 
if __name__ == "__main__":
    tml = nn.TripletMarginLoss(reduction='none')
    pdist = nn.PairwiseDistance()
    a = torch.rand(48, 768)
    p = torch.rand(48, 768)
    n = torch.rand(48, 768)
    sct_loss(a, p, n, pdist, tml)