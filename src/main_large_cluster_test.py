from experiment_runner import *
from gradient_viewer import *



K = 4
d = 100
lr = 0.1
epochs = 400
n_per_cluster = 5
num_runs = 100
m = 5

total_grad_stats = run_and_report_grads(K, d, n_per_cluster, lr, epochs, num_runs, m=m)
#display_grads(total_grad_stats, epochs)






