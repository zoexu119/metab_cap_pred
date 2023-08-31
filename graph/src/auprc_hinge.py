import torch
import torch.nn as nn
import numpy as np



CUDA_ENABLED = True
DISTRIBUTED_WORLD_SIZE = 1


def Variable(data, *args, **kwargs):
    if CUDA_ENABLED:
        return torch.autograd.Variable(data.cuda(), *args, **kwargs)
    else:
        return torch.autograd.Variable(data, *args, **kwargs)


def var_to_numpy(v):
    return (v.cpu() if CUDA_ENABLED else v).data.numpy()


def zerovar(*size):
    return Variable(torch.zeros(*size))


def FloatTensor(*args):
    if CUDA_ENABLED:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


def LongTensor(*args):
    if CUDA_ENABLED:
        return torch.cuda.LongTensor(*args)
    else:
        return torch.LongTensor(*args)


def GetTensor(tensor):
    if CUDA_ENABLED:
        return tensor.cuda()
    else:
        return tensor


def tensor(data, dtype):
    return torch.tensor(data, dtype=dtype, device=device())


def device():
    return "cuda:{}".format(torch.cuda.current_device()) if CUDA_ENABLED else "cpu"


def range_to_anchors_and_delta(precision_range, num_anchors):
    """Calculates anchor points from precision range.
        Args:
            precision_range: an interval (a, b), where 0.0 <= a <= b <= 1.0
            num_anchors: int, number of equally spaced anchor points.
        Returns:
            precision_values: A `Tensor` of [num_anchors] equally spaced values
                in the interval precision_range.
            delta: The spacing between the values in precision_values.
        Raises:
            ValueError: If precision_range is invalid.
    """
    # Validate precision_range.
    if len(precision_range) != 2:
        raise ValueError(
            "length of precision_range (%d) must be 2" % len(precision_range)
        )
    if not 0 <= precision_range[0] <= precision_range[1] <= 1:
        raise ValueError(
            "precision values must follow 0 <= %f <= %f <= 1"
            % (precision_range[0], precision_range[1])
        )

    # Sets precision_values uniformly between min_precision and max_precision.
    precision_values = np.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return FloatTensor(precision_values), delta


def build_class_priors(
    labels,
    class_priors=None,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0,
):
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.
    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors


def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    """
    Args:
        labels: one-hot representation `Tensor` of shape broadcastable to logits
        logits: A `Tensor` of shape [N, C] or [N, C, K]
        positive_weights: Scalar or Tensor
        negative_weights: same shape as positive_weights
    Returns:
        3D Tensor of shape [N, C, K], where K is length of positive weights
        or 2D Tensor of shape [N, C]
    """
    positive_weights_is_tensor = torch.is_tensor(positive_weights)
    negative_weights_is_tensor = torch.is_tensor(negative_weights)

    # Validate positive_weights and negative_weights
    if positive_weights_is_tensor ^ negative_weights_is_tensor:
        raise ValueError(
            "positive_weights and negative_weights must be same shape Tensor "
            "or both be scalars. But positive_weight_is_tensor: %r, while "
            "negative_weight_is_tensor: %r"
            % (positive_weights_is_tensor, negative_weights_is_tensor)
        )

    if positive_weights_is_tensor and (
        positive_weights.size() != negative_weights.size()
    ):
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            % (positive_weights.size(), negative_weights.size())
        )

    # positive_term: Tensor [N, C] or [N, C, K]
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)

    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
            positive_term.unsqueeze(-1) * positive_weights
            + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


def true_positives_lower_bound(labels, logits, weights):
    """
    true_positives_lower_bound defined in paper:
    "Scalable Learning of Non-Decomposable Objectives"
    Args:
        labels: A `Tensor` of shape broadcastable to logits.
        logits: A `Tensor` of shape [N, C] or [N, C, K].
            If the third dimension is present,
            the lower bound is computed on each slice [:, :, k] independently.
        weights: Per-example loss coefficients, with shape [N, 1] or [N, C]
    Returns:
        A `Tensor` of shape [C] or [C, K].
    """
    # A `Tensor` of shape [N, C] or [N, C, K]
    loss_on_positives = weighted_hinge_loss(labels, logits, negative_weights=0.0)

    weighted_loss_on_positives = (
        weights.unsqueeze(-1) * (labels - loss_on_positives)
        if loss_on_positives.dim() > weights.dim()
        else weights * (labels - loss_on_positives)
    )
    return weighted_loss_on_positives.sum(0)


def false_postives_upper_bound(labels, logits, weights):
    """
    false_positives_upper_bound defined in paper:
    "Scalable Learning of Non-Decomposable Objectives"
    Args:
        labels: A `Tensor` of shape broadcastable to logits.
        logits: A `Tensor` of shape [N, C] or [N, C, K].
            If the third dimension is present,
            the lower bound is computed on each slice [:, :, k] independently.
        weights: Per-example loss coefficients, with shape broadcast-compatible with
            that of `labels`. i.e. [N, 1] or [N, C]
    Returns:
        A `Tensor` of shape [C] or [C, K].
    """
    loss_on_negatives = weighted_hinge_loss(labels, logits, positive_weights=0)

    weighted_loss_on_negatives = (
        weights.unsqueeze(-1) * loss_on_negatives
        if loss_on_negatives.dim() > weights.dim()
        else weights * loss_on_negatives
    )
    return weighted_loss_on_negatives.sum(0)


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)




class AUCPRHingeLoss(nn.Module):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """

    class Config():
        """
        Attributes:
            precision_range_lower (float): the lower range of precision values over
                which to compute AUC. Must be nonnegative, `\leq precision_range_upper`,
                and `leq 1.0`.
            precision_range_upper (float): the upper range of precision values over
                which to compute AUC. Must be nonnegative, `\geq precision_range_lower`,
                and `leq 1.0`.
            num_classes (int): number of classes(aka labels)
            num_anchors (int): The number of grid points used to approximate the
                Riemann sum.
        """

        precision_range_lower: float = 0.0
        precision_range_upper: float = 1.0
        num_classes: int = 2
        num_anchors: int = 20

    def __init__(self, weights=None, *args, **kwargs):
        """Args:
            config: Config containing `precision_range_lower`, `precision_range_upper`,
                `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)

        self.num_classes = 2 #self.config.num_classes
        self.num_anchors = 100 #self.config.num_anchors
        self.precision_range = (
            0.0, #self.config.precision_range_lower,
            1.0, #self.config.precision_range_upper,
        )

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.precision_values, self.delta = range_to_anchors_and_delta(#loss_utils.
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            FloatTensor(self.num_classes, self.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.num_classes, self.num_anchors).data.fill_(
                1.0
            )
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        logits = torch.cat((1-logits,logits),1)
        C = 1 if logits.dim() == 1 else logits.size(1)

        #print(torch.cat((1-logits,logits),1))
        

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(
            logits, targets, weights=weights
        )

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = lagrange_multiplier(self.lambdas)#loss_utils.
        # print("lambdas: {}".format(lambdas))

        # A `Tensor` of Shape [N, C, K]
        hinge_loss = weighted_hinge_loss(#loss_utils.
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=(1.0 + lambdas) * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values,
        )

        # 1D tensor of shape [C]
        class_priors = build_class_priors(labels, weights=weights)#loss_utils.

        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values)
        )

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        loss = per_anchor_loss.sum(2) * self.delta #per_anchor_loss.sum(2) * self.delta / (1.0 - self.precision_values)
        loss /= self.precision_range[1] - self.precision_range[0]
        #print(loss.mean())

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        # print(logits.size())
        # print(targets.size())
        # print(targets)
        # Converts targets to one-hot representation. Dim: [N, C]
        targets = targets.long()
        # print(targets.unsqueeze(1).data)
        labels = FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)

        if weights is None:
            weights = FloatTensor(N).fill_(1.0)
            #print(weights)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        return labels, weights