class NeuralNet:
    def __init__(self, n_layers, n_units, epochs, lr, momentum, activation, val_percent):
        self.L = n_layers
        self.n = n_units
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.fact = activation
        self.val_percent = val_percent
        # Inicializar arrays: h, xi, w, theta, delta, d_w, d_theta, d_w_prev, d_theta_prev