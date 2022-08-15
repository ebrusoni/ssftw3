import matplotlib.pyplot as plt
import pandas as pd


path_to_parent = 'C:/Users/henry/OneDrive/Desktop/BachelorThesis/'
sensor_results_paths = ['DSFT4/rain', 'DSFT3/rain', 'WDSFT3/rain']
sensor_results_description = ['DSFT4', 'DSFT3', 'WDSFT3']
assert(len(sensor_results_paths)>0)
assert(len(sensor_results_description)==len(sensor_results_paths))
df_results = []
for path in sensor_results_paths:
    df_results.append(pd.read_csv(path_to_parent+'res/'+path+'/log_det.csv'))

gt = df_results[0].to_numpy()[:,1]
random = df_results[0].to_numpy()[:,3]

fts = [df.to_numpy()[:,2] for df in df_results]

plt.plot(gt, label='gt')
for i in range(len(fts)):
    plt.plot(fts[i], label=sensor_results_description[i])
plt.plot(random, label='random')
plt.legend()
plt.xlabel('cardinality constraint')
plt.ylabel('information gain')
plt.xlim(0, gt.shape[0])
plt.ylim(bottom=0)
plt.savefig('../res/plot_RAIN2.pdf', format='pdf')
plt.close()


