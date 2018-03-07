import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

CENT_COLOURS = ['k', '#ffa500']


def animate(clusters, all_prev_lists):
    fig, ax = plt.subplots()
    _draw(clusters)
    epochs = len(all_prev_lists[0])
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-7.0, 7.0)
    # ax.set_xlim(-14.0, 14.0)
    # ax.set_ylim(-14.0, 14.0)

    #Plot the inital centres
    num_algs = len(all_prev_lists)
    scats = []
    for i in xrange(num_algs):
        all_prev = all_prev_lists[i]
        initial_centres = all_prev[0]
        colour = CENT_COLOURS[i % 2]
        scat = _plot_scatter(initial_centres, colour)
        scats.append(scat)

    #Delegates required for the animator
    def init():
        return scats

    def update(frame):
        for scat, all_prev in zip(scats, all_prev_lists):
            current_centres = all_prev[frame+1]
            K, _ = current_centres.shape
            as_list = np.split(current_centres, K, axis=0)
            scat.set_offsets(as_list)

    #Run the animation
    ani = FuncAnimation(fig, update, frames=epochs - 1, init_func=init, interval=50, repeat=False)
    plt.show()

def plot_z(all_prev_z):
    epochs = all_prev_z.shape[0]
    x = np.arange(epochs)
    plt.plot(x, all_prev_z)
    plt.show()



def _draw(clusters):
    K = len(clusters)
    #Get even spread of colours for scatter plot
    color_func = _get_cmap(K)
    for i in xrange(K):
        cluster = clusters[i]
        _plot_scatter(cluster, color_func(i))


def _plot_scatter(clust_or_centres, colour):
    xs = clust_or_centres[:, 0]
    ys = clust_or_centres[:, 1]
    return plt.scatter(xs, ys, color=colour)


def _get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='gist_rainbow')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

# all_prev = [np.array([[-1.0, -2.0],[1.0, 1.0],[0.5, -1.0]]), np.array([[-0.5, -1.5],[1.5, 1.5],[1.0, -0.5]]),
#             np.array([[0.0, -1.0],[2.0, 2.0],[1.5, 0.0]]), np.array([[0.5, -0.5],[2.5, 2.5],[2.0, 0.5]])]
# from cluster_gen import k_gaussian_clusters
# clusters, _ = k_gaussian_clusters(4)
# animate(clusters, all_prev)
