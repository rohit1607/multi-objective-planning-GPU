import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


all_Yi = np.load('all_Yi.npy')
print(all_Yi.shape)

for t in [1, 60, 120 , 180]:
    coeffs = all_Yi[t,:,0:4]
    df = pd.DataFrame(coeffs, columns = ['Mode 1','Mode 2','Mode 3','Mode 4'])
    sns.set_context("talk", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":25})   

    g= sns.pairplot(df, corner=True, diag_kind="kde", plot_kws=dict(s=20, edgecolor= 'b', linewidth=0.5, alpha=0.1))
    # g.map_lower(sns.kdeplot, levels=4, color=".2")
    # g.set(xlim=(-3, 3))
    # g.set(ylim=(-3, 3))
    for ax in g.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.2,0.5)
        ax.set_xlim((-3,3))
        ax.set_ylim((-3,3))
    for ax in g.axes[3,:]:
        # ax.get_yaxis().set_label_coords(-0.5,0.5)
        ax.set_xlim((-3,3))
        ax.set_ylim((-3,3))

    # g.axes.set_title("Title",fontsize=50)
    # g.set_xlabel("X Label",fontsize=30)
    # g.set_ylabel("Y Label",fontsize=20)
    # g.tick_params(labelsize=5)

    fname = "coeff_pairplots@" + str(t)
    plt.savefig(fname, dpi = 300)
    plt.show()
    plt.clf()
    plt.close()
