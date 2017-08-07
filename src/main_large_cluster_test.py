from experiment_runner import *
from gradient_viewer import *



K = 10
d = 2
lr = 0.2
epochs = 200
n_per_cluster = 5
num_runs = 30

all_grads = run_and_report_grads(K, d, n_per_cluster, lr, epochs, num_runs)
display_grads(all_grads, epochs)






