import  random
import  numpy as np
import torch

def init_weights(distribution_limit): #0.08 volt a "sequence to sequence learning with neural networks" cikkben
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -distribution_limit, distribution_limit)


# model.apply(init_weights)

def determine_randomization(seed):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True