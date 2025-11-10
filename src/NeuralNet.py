import numpy as np

class NeuralNet:
    def __init__(self, n_layers, n_units, epochs, lr, momentum, activation='sigmoid', val_percent=0.2):
        self.L = n_layers
        self.n = n_units
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.fact = activation
        self.val_percent = val_percent

        # Inicializar arrays internos
        self.h = [np.zeros(n) for n in n_units]            # campos h
        self.xi = [np.zeros(n) for n in n_units]           # activaciones
        self.delta = [np.zeros(n) for n in n_units]        # errores delta
        self.w = [np.random.randn(n_units[i], n_units[i-1])*0.1 for i in range(1, n_layers)]  # pesos
        self.theta = [np.zeros(n_units[i]) for i in range(1, n_layers)]                        # thresholds
        self.d_w = [np.zeros_like(wi) for wi in self.w]            # cambios de pesos
        self.d_theta = [np.zeros_like(ti) for ti in self.theta]    # cambios de thresholds
        self.d_w_prev = [np.zeros_like(wi) for wi in self.w]       # momentum pesos
        self.d_theta_prev = [np.zeros_like(ti) for ti in self.theta]

        # Para guardar errores por epoch
        self.train_errors = []
        self.val_errors = []

    # ===================== Funciones de activación =====================
    def _activate(self, x):
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.fact == 'tanh':
            return np.tanh(x)
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'linear':
            return x
        else:
            raise ValueError("Activación desconocida")

    def _derivative(self, x):
        if self.fact == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.fact == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.fact == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.fact == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError("Activación desconocida")

    # ===================== Forward propagation =====================
    def _forward(self, x_sample):
        self.xi[0] = x_sample  # capa de entrada
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l-1], self.xi[l-1]) - self.theta[l-1]
            self.xi[l] = self._activate(self.h[l])
        return self.xi[-1]

    # ===================== Backpropagation =====================
    def _backprop(self, y_sample):
        # Calcular delta de la capa de salida
        self.delta[-1] = (y_sample - self.xi[-1]) * self._derivative(self.h[-1])
        # Backprop hacia capas ocultas
        for l in reversed(range(1, self.L-1)):
            self.delta[l] = self._derivative(self.h[l]) * np.dot(self.w[l].T, self.delta[l+1])
        # Actualizar pesos y thresholds
        for l in range(self.L-1):
            d_w_new = self.lr * np.outer(self.delta[l+1], self.xi[l]) + self.momentum * self.d_w_prev[l]
            d_theta_new = -self.lr * self.delta[l+1] + self.momentum * self.d_theta_prev[l]

            self.w[l] += d_w_new
            self.theta[l] += d_theta_new

            self.d_w_prev[l] = d_w_new
            self.d_theta_prev[l] = d_theta_new

    # ===================== Entrenamiento =====================
    def fit(self, X, y):
        # Dividir dataset en train/validation
        if self.val_percent > 0:
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            split = int(len(X) * (1 - self.val_percent))
            train_idx, val_idx = idx[:split], idx[split:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for epoch in range(self.epochs):
            mse_epoch = 0
            for i in range(len(X_train)):
                self._forward(X_train[i])
                self._backprop(y_train[i])
                mse_epoch += (y_train[i] - self.xi[-1])**2
            mse_epoch /= len(X_train)
            self.train_errors.append(mse_epoch)

            # Error validación
            if X_val is not None:
                val_mse = 0
                for i in range(len(X_val)):
                    y_pred = self._forward(X_val[i])
                    val_mse += (y_val[i] - y_pred)**2
                val_mse /= len(X_val)
                self.val_errors.append(val_mse)

    # ===================== Predicción =====================
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            y_pred[i] = self._forward(X[i])
        return y_pred

    # ===================== Errores por época =====================
    def loss_epochs(self):
        return np.array(list(zip(self.train_errors, self.val_errors if self.val_errors else [0]*len(self.train_errors))))
