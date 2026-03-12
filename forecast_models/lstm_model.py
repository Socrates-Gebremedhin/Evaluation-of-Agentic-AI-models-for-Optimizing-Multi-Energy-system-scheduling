"""LSTM model for sequence forecasting."""
import numpy as np
import torch
import torch.nn as nn


class _StackedLSTM(nn.Module):
    """Stack of LSTM layers with possibly different hidden sizes per layer."""

    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, h in enumerate(hidden_sizes):
            in_size = input_size if i == 0 else hidden_sizes[i - 1]
            self.layers.append(nn.LSTM(in_size, h, num_layers=1, batch_first=True))

    def forward(self, x):
        for lstm in self.layers:
            x, _ = lstm(x)
        return x


class LSTM_Model:
    """
    LSTM for sequence-to-one forecasting.
    hidden_size: int (e.g. 16) -> all layers use that size; or sequence (e.g. (16, 32, 12)) -> per-layer sizes.
    num_layers: used only when hidden_size is an int; ignored when hidden_size is a sequence.
    """

    def __init__(
        self,
        input_size,
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        optimizer_cls=torch.optim.Adam,
        lr=0.001,
        epochs=50,
        patience=5,
        target_col=0,
        batch_size=64,
    ):
        self.input_size = input_size
        self.epochs = epochs
        self.patience = patience
        self.target_col = target_col
        self.batch_size = batch_size

        # hidden_size: int -> all layers same size; sequence -> per-layer sizes, e.g. (16, 32, 12)
        if isinstance(hidden_size, (list, tuple)):
            hidden_sizes = tuple(hidden_size)
            num_layers = len(hidden_sizes)
        else:
            hidden_sizes = (hidden_size,) * num_layers
        self.hidden_size = hidden_sizes[-1]  # output size for fc
        self.num_layers = num_layers

        self.model = _StackedLSTM(input_size, hidden_sizes)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_sizes[-1], 1)

        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = self.optimizer_cls(
            list(self.model.parameters()) + list(self.dropout.parameters()) + list(self.fc.parameters()), lr=self.lr
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dropout.to(self.device)
        self.fc.to(self.device)
        self.last_seq = None
        self.history = None

    def fit(self, X, y=None, X_val=None, y_val=None, **kwargs):
        if isinstance(X, tuple):
            X, y = X
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.last_seq = X[-1:].clone()

        X_val_t = None
        y_val_t = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1).to(self.device)

        n = X.shape[0]
        batch_size = min(self.batch_size, n)
        # Early stopping on validation loss (out-of-sample) when provided; else on train loss
        best_score = np.inf
        patience_counter = 0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}

        for epoch in range(self.epochs):
            self.model.train()
            self.dropout.train()
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                X_b = X[idx]
                y_b = y[idx]
                self.optimizer.zero_grad()
                out = self.model(X_b)
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                loss = self.criterion(out, y_b)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss /= max(1, n_batches)
            train_rmse = float(np.sqrt(epoch_loss))
            history["train_loss"].append(epoch_loss)
            history["train_rmse"].append(train_rmse)

            val_loss = np.nan
            val_rmse = np.nan
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                self.dropout.eval()
                with torch.no_grad():
                    out_val = self.model(X_val_t)
                    out_val = self.fc(out_val[:, -1, :])
                    val_loss = self.criterion(out_val, y_val_t).item()
                val_rmse = float(np.sqrt(val_loss))
                history["val_loss"].append(val_loss)
                history["val_rmse"].append(val_rmse)
                # Early stop on validation loss (lower is better)
                stop_score = val_loss
            else:
                history["val_loss"].append(np.nan)
                history["val_rmse"].append(np.nan)
                stop_score = epoch_loss

            if stop_score < best_score:
                best_score = stop_score
                patience_counter = 0
                best_state = {
                    "model": self.model.state_dict(),
                    "fc": self.fc.state_dict(),
                    "dropout": self.dropout.state_dict(),
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if best_state is not None:
                        self.model.load_state_dict(best_state["model"])
                        self.fc.load_state_dict(best_state["fc"])
                        self.dropout.load_state_dict(best_state["dropout"])
                    break

        self.history = history

    def predict_from_sequences(self, X):
        """
        Predict one step ahead for each input sequence in X.
        X shape: (n_sequences, seq_len, n_features).
        Returns: numpy array of shape (n_sequences,).
        """
        if isinstance(X, tuple):
            X, _ = X
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.dropout.eval()
        self.fc.eval()
        with torch.no_grad():
            out = self.model(X_t)
            y_pred = self.fc(out[:, -1, :])[:, 0]  # no dropout in eval
        return y_pred.detach().cpu().numpy()

    def predict(self, horizon):
        preds = []
        seq = self.last_seq.clone()
        self.model.eval()
        self.dropout.eval()
        self.fc.eval()
        with torch.no_grad():
            for _ in range(horizon):
                out = self.model(seq)
                y_pred = self.fc(out[:, -1, :])[:, 0]  # no dropout in eval
                preds.append(y_pred.item())
                next_seq = seq[:, 1:, :].clone()
                next_features = seq[:, -1, :].clone()
                next_features[:, self.target_col] = y_pred
                next_seq = torch.cat([next_seq, next_features.unsqueeze(1)], dim=1)
                seq = next_seq
        return np.array(preds)

    def reset(self):
        self.model.apply(self._weight_reset)
        self.fc.apply(self._weight_reset)

    @staticmethod
    def _weight_reset(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
