from cProfile import label
from tempfile import NamedTemporaryFile
 
import numpy as np
import matplotlib.pyplot as plt

def get_avg_coefs(ft, max_card, flag_rescale=True):
    cardinalites = ft.freqs.sum(axis =1)
    metric = lambda x: np.linalg.norm(x)**2
    avg_values = []
    #print(ft.coefs.shape)
    for i in range(max_card+1):
        card_i = cardinalites == np.full(ft.freqs.shape[0],i)
        if(flag_rescale):
            avg = metric(card_i*ft.coefs)/metric(ft.coefs)
            avg_values +=  [avg]
        else:
            avg = metric(card_i*ft.coefs)
            avg_values += [avg]
    return avg_values
   
def plot_avg_coefs(ft,max_card,flag_rescale = True):
    cardinalites = ft.freqs.sum(axis =1)
    metric = lambda x: np.linalg.norm(x)**2
    avg_values = []
    for i in range(max_card+1):
        card_i = cardinalites == np.full(ft.freqs.shape[0],i)
        if(flag_rescale):
            avg = metric(card_i*ft.coefs)/metric(ft.coefs)
            avg_values +=  [avg]
        else:
            avg = metric(card_i*ft.coefs)
            avg_values += [avg]
   
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        plt.plot(avg_values,label = 'setfunction')
        plt.legend()
        plt.xlabel('Cardinality of Frequency')
        plt.ylabel('Avg. Coefficient')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()