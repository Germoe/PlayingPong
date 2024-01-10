import numpy as np
import torch

if __name__ == "__main__":
    a = torch.FloatTensor(3, 2)
    print(a)

    b = np.zeros(shape=(3, 2))
    b = torch.tensor(b)
    print(b)

    c = torch.tensor([1, 2, 3])
    c = c.sum()
    print(c.item(), c)
