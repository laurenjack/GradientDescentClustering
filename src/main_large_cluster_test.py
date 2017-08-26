from experiment_runner import *
from gradient_viewer import *



K = 10
d = 1000
lr = 0.15
epochs = 500
n_per_cluster = 5
num_runs = 100

total_grad_stats = run_and_report_grads(K, d, n_per_cluster, lr, epochs, num_runs)
display_grads(total_grad_stats, epochs)






