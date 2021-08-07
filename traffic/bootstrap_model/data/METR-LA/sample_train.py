import numpy as np
import pandas as pd
import os
data = np.load("data/METR-LA/train.npz")
for i in range (1):
    index = np.random.choice(23974, 11987, replace=False)
    print(index)
    np.savez_compressed(
                os.path.join("data/METR-LA/train_"+str(i)+".npz"),
                x=data['x'][index],
                y=data['y'][index],
                x_offsets=data['x_offsets'],
                y_offsets=data['y_offsets'],
            )
