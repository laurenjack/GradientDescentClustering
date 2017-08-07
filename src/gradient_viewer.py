import matplotlib.pyplot as plt
import numpy as np

"""File responsible for displaying the average gradient over a set of
training runs, for both the centres and the weights"""

def display_grads(all_grads, epochs):
    """Display dW over time on the left hand side, and dX_bar over
    time on the right hand side"""
    #extract x and y
    all_dW, all_dX_bar = all_grads
    all_epochs = np.arange(epochs)

    #Create side by side subplots
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Average gradients over time')
    ax1.plot(all_epochs, all_dW)
    ax2.plot(all_epochs, all_dX_bar)
    plt.show()

