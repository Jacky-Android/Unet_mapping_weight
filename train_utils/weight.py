from torch.autograd import Variable
import torch
def get_weights(target, w0, sigma, imsize):
    '''
    w1 is temporarily torch.ones - it should handle class imbalance for the whole dataset
    '''
    w0, sigma, C = _get_loss_variables(w0, sigma, imsize)
    distances = target
    sizes = target
    #print(distances.size())
    w1 = Variable(torch.ones(distances.size()), requires_grad=False)  # TODO: fix it to handle class imbalance
    if torch.cuda.is_available():
        w1 = w1.cuda()
    size_weights = _get_size_weights(sizes, C)

    distance_weights = _get_distance_weights(distances, w1, w0, sigma)

    weights = distance_weights * size_weights

    return weights


def _get_distance_weights(d, w1, w0, sigma):
    d = d.cuda()
    weights = w1 + w0 * torch.exp(-(d ** 2) / (sigma ** 2))
    weights[d == 0] = 1
    return weights


def _get_size_weights(sizes, C):
    sizes_ = sizes.clone()
    sizes_[sizes == 0] = 1
    sizes_ = sizes_.cuda()
    size_weights =  C/ sizes_
    size_weights[sizes_ == 1] = 1
    return size_weights


def _get_loss_variables(w0, sigma, imsize):
    w0 = Variable(torch.Tensor([w0]), requires_grad=False)
    sigma = Variable(torch.Tensor([sigma]), requires_grad=False)
    C = Variable(torch.sqrt(torch.Tensor([imsize[0] * imsize[1]])) / 2, requires_grad=False)
    if torch.cuda.is_available():
        w0 = w0.cuda()
        sigma = sigma.cuda()
        C = C.cuda()
    return w0, sigma, C