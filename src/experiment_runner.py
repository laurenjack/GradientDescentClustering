from cluster_gen import *
from grad_desc_clustering import GDC
from grad_desc_clustering import GradStats
import cluster_utils

"""Responsible for running multiple clustering runs, and reporting the results,
as well as aspets of the training procedure such as the gradient over time."""

def run_and_report_grads(K, d, n_per_cluster, lr, epochs, num_runs, m=None):
    """Run the GDC algorithm n times and report the
    result, as well as the gradient over time"""
    total_opt = 0
    total_gdc = 0
    gdc_under = 0

    total_grad_stats = np.zeros((5, epochs))

    for i in xrange(num_runs):
        # Generate the clusters
        clusters, actual_centres, global_opt = k_centres_d_space(K, d, n_per_cluster)
        X = np.concatenate(clusters, axis=0)

        # Choose K for GDC
        gdc = GDC()
        # co = ConvexOptimizer(gdc)
        # tp = TrainingParams(X, lr, epochs)
        # K, _ = co.bin_search(tp, 0, 10)

        # Create same set of starting centres
        X_bar = np.random.uniform(low=-5.0, high=5.0, size=(K, d))

        # Train the clustering algorithms
        #W, X_bar_gdc, all_prev_gdc, grad_stats = gdc.train(X, K, lr, epochs, X_bar)
        _, X_bar_gdc, all_prev_gdc = gdc.train_sgd(X, K, lr, epochs, m, X_bar)


        # Compute total cost of each clustering alg
        gdc_C = cluster_utils.cost_of_closest_to(X, X_bar_gdc)

        #total_grad_stats += grad_stats.as_np()

        # Update agregates
        total_opt += global_opt
        total_gdc += gdc_C
        if gdc_C < global_opt: #+ 0.1:
            gdc_under += 1
        else:
            print "Global Optimum: " + str(global_opt) + "   GDC: " + str(gdc_C)

    print "Global Optimum: " + str(total_opt / 100.0)
    print "GDC: " + str(total_gdc / 100.0) + "  Percent Under: " + str(gdc_under)
    #return total_grad_stats/float(num_runs)