import numpy as np

class NeuralNet:
    def __init__(self, n_layers, n_units, epochs=100, lr=1e-3, momentum=0.0,
                 activation='sigmoid', val_percent=0.2, batch_size=32, seed=None, shuffle=True):
        assert n_layers == len(n_units), "n_layers must match length of n_units"
        self.L = n_layers
        self.n = list(n_units)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.fact = activation
        self.val_percent = float(val_percent)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.shuffle = bool(shuffle)

        # RNG
        self.rng = np.random.RandomState(seed)

        # Internal arrays per layer (kept as lists for compatibility with assignment)
        # h and xi will store activations/fields for a *single batch* during forward pass,
        # but we still keep the same structure as lists per layer.
        self.h = [None for _ in range(self.L)]
        self.xi = [None for _ in range(self.L)]
        self.delta = [None for _ in range(self.L)]

        # Weights: w[l] corresponds to connections (layer l-1) -> (layer l)
        # We'll store w as list index 1..L-1 
        self.w = [None] + [
            self.rng.randn(self.n[l], self.n[l-1]) * np.sqrt(2.0 / max(1, self.n[l-1]))
            for l in range(1, self.L)
        ]
        # thresholds (biases) for layers 1..L-1
        self.theta = [None] + [np.zeros(self.n[l]) for l in range(1, self.L)]

        # For momentum: d_w and d_theta are last weight/bias increments
        self.d_w = [None] + [np.zeros_like(self.w[l]) for l in range(1, self.L)]
        self.d_theta = [None] + [np.zeros_like(self.theta[l]) for l in range(1, self.L)]
        self.d_w_prev = [None] + [np.zeros_like(self.w[l]) for l in range(1, self.L)]
        self.d_theta_prev = [None] + [np.zeros_like(self.theta[l]) for l in range(1, self.L)]

        # Track errors per epoch
        self.train_errors = []
        self.val_errors = []

    # ---------------- Activation functions (vectorized) ----------------
    def _activate(self, x):
        if self.fact == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.fact == 'tanh':
            return np.tanh(x)
        elif self.fact == 'relu':
            return np.maximum(0.0, x)
        elif self.fact == 'linear':
            return x
        else:
            raise ValueError("Unknown activation: " + str(self.fact))

    def _derivative(self, x, act=None):
        """
        x: pre-activation (h) or activation depending on act. For numerical stability,
        we will compute derivatives using h for relu and using activation for sigmoid/tanh if helpful.
        """
        if act is None:
            act = self.fact
        if act == 'sigmoid':
            # x is pre-activation (h) -> compute sigmoid then derivative
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        elif act == 'tanh':
            return 1.0 - np.tanh(x)**2
        elif act == 'relu':
            return (x > 0.0).astype(float)
        elif act == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError("Unknown activation for derivative: " + str(act))

    # ---------------- Vectorized forward pass ----------------
    def _forward_batch(self, X_batch):
        B = X_batch.shape[0]
        self.xi[0] = X_batch  # shape (B, n0)
        for l in range(1, self.L):
            # z = X @ W.T - theta  where W shape (n_l, n_{l-1})
            # Compute pre-activation h as (B, n_l)
            h_l = np.dot(self.xi[l-1], self.w[l].T) - self.theta[l]  # broadcast theta (n_l,)
            a_l = self._activate(h_l) if (l < self.L - 1 or self.fact != 'linear') else h_l  # last-layer linear allowed
            self.h[l] = h_l
            self.xi[l] = a_l
        return self.xi[-1]  # shape (B, n_L)

    # ---------------- Vectorized backward pass and parameter update ----------------
    def _backprop_batch(self, y_batch):
        # Ensure y_batch shape (B, n_L)
        y_arr = np.array(y_batch)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim == 2 and y_arr.shape[1] != 1:
            # if user passed shape (B,1) it's fine, else keep as is if matching output dims
            pass

        B = y_arr.shape[0]
        # Output layer delta: (a_L - y) * derivative(h_L)
        a_L = self.xi[-1]  # shape (B, n_L)
        h_L = self.h[-1]
        # delta_L shape (B, n_L)
        delta_L = (a_L - y_arr) * self._derivative(h_L, act=self.fact)
        self.delta[-1] = delta_L

        # Backpropagate through hidden layers l = L-1 .. 1
        for l in range(self.L - 1, 0, -1):
            if l > 1:
                # delta_{l-1} = (delta_l @ w_l) * derivative(h_{l-1})
                # w_l: shape (n_l, n_{l-1}) -> w_l.T shape (n_{l-1}, n_l)
                delta_prev = np.dot(delta_L, self.w[l]) * self._derivative(self.h[l-1], act=self.fact)
                self.delta[l-1] = delta_prev
                delta_L = delta_prev

        # Gradients and parameter updates for each layer l=1..L-1
        # For layer l: grad_w = (delta_l.T @ a_{l-1}) / B
        for l in range(1, self.L):
            a_prev = self.xi[l-1]        # shape (B, n_{l-1})
            delta_l = self.delta[l]      # shape (B, n_l)

            # gradient shape (n_l, n_{l-1})
            grad_w = np.dot(delta_l.T, a_prev) / B
            grad_theta = np.mean(delta_l, axis=0) * (-1.0)  # since h = W x - theta, derivative w.r.t theta is -delta; we update subtracting grad

            # Parameter update with momentum:
            d_w_new = -self.lr * grad_w + self.momentum * self.d_w_prev[l]
            d_theta_new = -self.lr * grad_theta + self.momentum * self.d_theta_prev[l]

            # Apply updates
            self.w[l] += d_w_new
            self.theta[l] += d_theta_new

            # save prev
            self.d_w_prev[l] = d_w_new
            self.d_theta_prev[l] = d_theta_new

    # ---------------- Training (fit) ----------------
    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)  # (N,1)

        N = X_arr.shape[0]
        # Split train/val
        if self.val_percent > 0.0:
            idx = np.arange(N)
            self.rng.shuffle(idx)
            split = int(N * (1.0 - self.val_percent))
            train_idx = idx[:split]
            val_idx = idx[split:]
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]
        else:
            X_train, y_train = X_arr, y_arr
            X_val, y_val = None, None

        n_train = X_train.shape[0]
        n_batches = max(1, int(np.ceil(n_train / float(self.batch_size))))

        # Reset error trackers
        self.train_errors = []
        self.val_errors = []

        for epoch in range(self.epochs):
            # Shuffle training set each epoch if requested
            if self.shuffle:
                perm = self.rng.permutation(n_train)
                X_train = X_train[perm]
                y_train = y_train[perm]

            # Mini-batch loop (vectorized inside)
            epoch_mse = 0.0
            for b in range(n_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_train)
                Xb = X_train[start:end]
                yb = y_train[start:end]

                # Forward and backprop for batch
                y_pred_b = self._forward_batch(Xb)           # (B,1) expected if output dim=1
                self._backprop_batch(yb)

                # accumulate mse
                # y_pred_b may be shape (B,1)
                diff = (y_pred_b - yb)
                epoch_mse += np.sum(diff**2)

            epoch_mse /= n_train
            self.train_errors.append(float(epoch_mse))

            # Validation error
            if X_val is not None:
                y_val_pred = self._forward_batch(X_val)
                val_mse = float(np.mean((y_val_pred - y_val)**2))
                self.val_errors.append(val_mse)

        # done training

    # ---------------- Prediction ----------------
    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        # forward in batches to save memory if needed
        N = X_arr.shape[0]
        preds = np.zeros((N, self.n[-1]))
        batch_size = self.batch_size
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            preds[start:end] = self._forward_batch(X_arr[start:end])
        # return flattened if single-output
        if preds.shape[1] == 1:
            return preds.reshape(-1)
        return preds

    # ---------------- Loss history ----------------
    def loss_epochs(self):
        # If no val_errors, return zeros for val
        if len(self.val_errors) == 0:
            val = [0.0] * len(self.train_errors)
        else:
            val = self.val_errors
        return np.vstack([np.array(self.train_errors), np.array(val)]).T
