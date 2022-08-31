import os, torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


## NOTE: cd to custom/data/mfcc to run

M030_MFCC_DIR = os.path.join(os.getcwd(), 'M030')

for sub_dir in os.listdir(M030_MFCC_DIR):
    sub_dir_path = os.listdir(os.path.join(M030_MFCC_DIR, sub_dir))
    fig, axs = plt.subplots(len(sub_dir_path)//5, 5)

    for (idx, mfcc) in enumerate(sub_dir_path):
        mfcc = np.load(os.path.join(M030_MFCC_DIR, sub_dir, mfcc))
        mfcc_plt_data= np.swapaxes(mfcc, 0 ,1)

        axs[idx//5 - 1, idx%5].imshow(mfcc_plt_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
        axs[idx//5 - 1, idx%5].set_title(f'{sub_dir} {idx}')

        # mfcc = torch.FloatTensor(mfcc)
        # mfcc=torch.unsqueeze(mfcc, 0)
        # if mfcc.shape != torch.Size([1, 28, 13]):
        #     print(mfcc.shape)
    plt.show()

