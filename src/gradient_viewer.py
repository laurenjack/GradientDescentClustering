import matplotlib.pyplot as plt
import numpy as np

"""File responsible for displaying the average gradient over a set of
training runs, for both the centres and the weights"""

def display_grads(tgs, epochs):
    """Display dW over time on the left hand side, and dX_bar over
    time on the right hand side"""
    #extract x and y
    avg_dW, avg_dX_bar, max_dW, max_dXbar, max_p_sum = tgs[0], tgs[1], tgs[2], tgs[3], tgs[4]
    all_epochs = np.arange(epochs)

    #Create side by side subplots, for average and max grads
    #plt.figure(1)
    #plt.title('Average gradients over time')
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(all_epochs, avg_dW)
    ax2.plot(all_epochs, avg_dX_bar)

    #plt.figure(2)
    #plt.title('Max gradients over time')
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(all_epochs, max_dW)
    ax2.plot(all_epochs, max_dXbar)

    plt.figure(3)
    plt.plot(all_epochs, max_p_sum)

    plt.show()



