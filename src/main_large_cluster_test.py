from experiment_runner import *
from gradient_viewer import *



K = 64
d = 100
lr = float(d) ** 0.5
epochs = 400
n_per_cluster = 10
num_runs = 10
m = 320

total_grad_stats = run_and_report_grads(K, d, n_per_cluster, lr, epochs, num_runs, m=m)
#display_grads(total_grad_stats, epochs)






