import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function

        # TODO: define layers and initialize weights
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input to hidden layer: shape (input_dim, hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros(hidden_dim)

        # Hidden to output layer: shape (hidden_dim, output_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def _activate(self, X):
        if self.activation_fn == 'tanh':
            return np.tanh(X)
        elif self.activation_fn == 'relu':
            return np.maximum(0, X)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # forward pass: input layer to hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._activate(Z1)

        # forward pass: hidden layer to output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = Z2 #1 / (1 + np.exp(-Z2))

        # TODO: store activations for visualization
        self.A1 = A1
        self.A2 = A2
        self.Z1 = Z1
        self.Z2 = Z2

        # returning the forward pass
        out = self.A2
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]
        y = y.reshape(-1, 1)
        
        # Compute the gradient of the loss with respect to the output layer
        dZ2 = self.forward(X) - y  # Gradient of loss w.r.t output (Sigmoid gradient already included)

        # Gradients for W2 and b2
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m

        # Gradient of the loss w.r.t the hidden layer
        dA1 = np.dot(dZ2, self.W2.T)

        # Depending on the activation function used, compute the gradient
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - self.A1 ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.A1 > 0)
        elif self.activation_fn == 'sigmoid':
            dZ1 = dA1 * self.A1 * (1 - self.A1)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

        # Gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        

        # TODO: update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # TODO: store gradients for visualization
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Space at Step {frame*10}')

    # TODO: Hyperplane visualization in the hidden space
    x_vals = np.linspace(hidden_features[:, 0].min()-0.5, hidden_features[:, 0].max()+0.5, 50)
    y_vals = np.linspace(hidden_features[:, 1].min()-0.5, hidden_features[:, 1].max()+0.5, 50)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
    # Calculate the Z values for the hyperplane (third hidden dimension) based on the weights
    Z_grid = - (mlp.W2[0, 0] * X_grid + mlp.W2[1, 0] * Y_grid + mlp.b2[0]) / mlp.W2[2, 0]
    ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, color='r', alpha=0.3)

    # TODO: Distorted input space transformed by the hidden layer
    Z1 = np.dot(grid_points, mlp.W1) + mlp.b1  # Linear transformation
    hidden_grid = mlp._activate(Z1)

    # Reshape transformed grid for plotting
    H1 = hidden_grid[:, 0].reshape(X_grid.shape)
    H2 = hidden_grid[:, 1].reshape(X_grid.shape)
    H3 = hidden_grid[:, 2].reshape(X_grid.shape)

    # Plot the distorted grid in the hidden space
    ax_hidden.plot_surface(H1, H2, H3, color='lightblue', alpha=0.3, rstride=1, cstride=1)
    

    x_ticks = np.arange(np.floor(hidden_features[:, 0].min()), np.ceil(hidden_features[:, 0].max()) + 0.5, 0.5)
    y_ticks = np.arange(np.floor(hidden_features[:, 1].min()), np.ceil(hidden_features[:, 1].max()) + 0.5, 0.5)
    z_ticks = np.arange(np.floor(hidden_features[:, 2].min()), np.ceil(hidden_features[:, 2].max()) + 0.5, 0.5)
    ax_hidden.set_xlim([np.floor(hidden_features[:, 0].min())-0.5, np.ceil(hidden_features[:, 0].max())+0.5])
    ax_hidden.set_ylim([np.floor(hidden_features[:, 1].min())-0.5, np.ceil(hidden_features[:, 1].max())+0.5])
    ax_hidden.set_zlim([np.floor(hidden_features[:, 2].min())-0.5, np.ceil(hidden_features[:, 2].max())+0.5])
    ax_hidden.set_xticks(x_ticks)
    ax_hidden.set_yticks(y_ticks)
    ax_hidden.set_zticks(z_ticks)

    # TODO: Plot input layer decision boundary
    ax_input.set_title(f'Input Space at Step {frame*10}')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions for each point in the grid
    Z = mlp.forward(grid)
    Z = Z.reshape(xx.shape)
    Z_class = (Z >= 0.5).astype(int)

    # Plot the decision boundary for the input layer
    ax_input.contourf(xx, yy, Z_class, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', marker='o')
    

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    ax_gradient.set_title(f'Gradients at Step {frame*10}')
    positions = {
        'x1': (0, 0), 'x2': (0, 1),
        'h1': (0.5, 0), 'h2': (0.5, 0.5), 'h3': (0.5, 1),
        'y': (1, 0)
    }

    # Plot nodes
    for layer, (x, y) in positions.items():
        ax_gradient.add_patch(Circle((x, y), 0.05, color='blue'))
        ax_gradient.text(x, y, layer, fontsize=10, ha='center', va='center', color='white')

    # Plot edges with gradient magnitudes
    for i in range(mlp.input_dim):  # Input to hidden
        for j in range(mlp.hidden_dim):
            start = positions[f'x{i + 1}']
            end = positions[f'h{j + 1}']
            grad_magnitude = np.abs(mlp.dW1[i, j])
            ax_gradient.plot([start[0], end[0]], [start[1], end[1]], 'purple', linewidth=grad_magnitude * 20)

    for i in range(mlp.hidden_dim):  # Hidden to output
        start = positions[f'h{i + 1}']
        end = positions['y']
        grad_magnitude = np.abs(mlp.dW2[i, 0])
        ax_gradient.plot([start[0], end[0]], [start[1], end[1]], 'purple', linewidth=grad_magnitude * 20)

    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)

    


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)