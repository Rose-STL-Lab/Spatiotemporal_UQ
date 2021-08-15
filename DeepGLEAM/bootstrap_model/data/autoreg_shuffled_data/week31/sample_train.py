import numpy as np
import pandas as pd
import os
data = np.load("data/autoreg_shuffled_data/week31/train.npz")
for i in range (1):
    index = np.random.choice(25, 23, replace=True)
    print(index)
    np.savez_compressed(
                os.path.join("data/autoreg_shuffled_data/week31/train_"+str(i)+".npz"),
                x=data['x'][index],
                y=data['y'][index],
                x_offsets=data['x_offsets'],
                y_offsets=data['y_offsets'],
            )
