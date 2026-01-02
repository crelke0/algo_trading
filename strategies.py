from random import randint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def diversify(capital_allocation, market_data, ticker_to_idx, feature_to_idx):
    capital_allocation = np.full_like(capital_allocation, 1.0 / len(capital_allocation))
    return capital_allocation




class SVDDModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=(16, 16), emb_dim=8):
        super().__init__()
        dims = [in_dim] + list(hidden_dim) + [emb_dim]
        self.fcs = nn.ModuleList(
            nn.Linear(dims[i], dims[i+1], bias=False)
            for i in range(len(dims) - 1)
        )

    def forward(self, x, add_noise=False):
        # noise for smoother gradients in training
        if add_noise: 
            std = 1e-3
            alpha = 0.9
        else:
            std = 0
            alpha = 1
        x = x + torch.randn_like(x)*std
        for layer in self.fcs:
            x = F.relu(layer(x))
        return x
    

class SmoothSVDDLoss(nn.Module):
    def __init__(self, c, lambda_jacobian=0.1):
        super().__init__()
        self.c = c
        self.lambda_jacobian = lambda_jacobian

    def forward(self, z, x=None):
        # Standard SVDD distance loss
        dist = ((z - self.c)**2).sum(dim=1)
        loss_svdd = dist.mean()

        # Jacobian / input gradient regularization
        if x is not None and x.requires_grad:
            # Compute gradients of sum of distances w.r.t inputs
            grad = torch.autograd.grad(
                outputs=dist.sum(),
                inputs=x,
                create_graph=True
            )[0]  # shape (batch_size, input_dim)
            loss_jacobian = (grad**2).mean()
        else:
            loss_jacobian = 0.0

        # Total loss
        loss = loss_svdd + self.lambda_jacobian * loss_jacobian
        return loss

def make_svdd_input(log_returns, log_returns_mean, log_returns_std):
    eps = 1e-3
    log_returns_normalized = (log_returns - log_returns_mean) / log_returns_std
    mean = np.mean(log_returns_normalized[:-1], axis=0)
    std = np.std(log_returns_normalized[:-1], axis=0) + eps
    z_scores = (log_returns_normalized[-1] - mean) / std
    z_scores = (z_scores - np.mean(z_scores)) / (np.std(z_scores) + eps)

    input_data = np.concatenate((log_returns[-1], z_scores))
    return input_data


def train_svdd(market_data, feature_to_idx, window=5, training_fraction = 0.8):
    emb_dim = 8
    
    batch_size = 64
    hidden_dims = (16,8)
    lr = 1e-3
    weight_decay = 1e-4
    lambda_jacobian = 0.1
    epochs = 40
    display_every = 100000

    eps = 1e-3

    # make input data
    close_data = market_data[:, :, feature_to_idx["Close"]]
    log_returns = np.log(close_data[1:] / close_data[:-1])

    train_idx = int(len(log_returns)*training_fraction)
    log_returns_mean = np.mean(log_returns[:train_idx], axis=0)
    log_returns_std = np.std(log_returns[:train_idx], axis=0) + eps
    log_returns_normalized = (log_returns - log_returns_mean) / log_returns_std
    X = []
    for i in range(len(close_data) - window - 1):
        X.append(log_returns_normalized[i:i + window + 1].T.flatten())
        # X.append(make_svdd_input(log_returns[i:i + window + 1], log_returns_mean, log_returns_std))
    X = torch.tensor(X, dtype=torch.float32)

    training_data = X[:train_idx]
    test_data = X[train_idx:]

    model = SVDDModel(in_dim=X.shape[1], hidden_dim=hidden_dims, emb_dim=emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize center c
    with torch.no_grad():
        z = model(training_data)
        c = z.mean(dim=0)
        c[torch.abs(c) < eps] = eps

    criterion = SmoothSVDDLoss(c=c, lambda_jacobian=lambda_jacobian)

    for epoch in range(epochs):
        shuffled = training_data[torch.randperm(training_data.size(0))]
        for i in range(0, len(training_data), batch_size):
            batch = shuffled[i:min(i+batch_size, len(training_data))]
            batch.requires_grad = True
            optimizer.zero_grad()
            z = model(batch, add_noise=True)
            loss = criterion(z, x=batch)
            loss.backward()
            optimizer.step()

        if epoch % display_every == 0:
            optimizer.zero_grad()
            z = model(test_data)
            test_loss = criterion(z, x=test_data)
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

    return model, c, log_returns_mean, log_returns_std

def make_svdd_strategy(market_data, ticker_to_idx, feature_to_idx, training_fraction=0.8):
    window = 5
    eps = 1e-3
    model, c, log_returns_mean, log_returns_std = train_svdd(market_data, feature_to_idx, window=window, training_fraction=training_fraction)

    def strategy(capital_allocation, market_data, ticker_to_idx, feature_to_idx):
        if market_data.shape[0] <= window + 1:
            return diversify(capital_allocation, market_data, ticker_to_idx, feature_to_idx)
        log_returns = np.log(market_data[1:, :, feature_to_idx["Close"]] / market_data[:-1, :, feature_to_idx["Close"]])
        log_returns_normalized = (log_returns - log_returns_mean) / log_returns_std
        x = torch.tensor(log_returns_normalized[-window-1:].T.flatten(), dtype=torch.float32)
        # x = torch.tensor(make_svdd_input(log_returns[-window - 1:], log_returns_mean, log_returns_std), dtype=torch.float32)

        z = model(x.unsqueeze(0))
        score = ((z - c)**2).sum()

        h = 1e-3
        K = 16
        gradient = torch.zeros(len(capital_allocation))

        for _ in range(K):
            u = torch.randn_like(gradient)
            u = u / (u.norm() + 1e-8)
            pu = F.pad(u, (x.shape[0] - gradient.shape[0], 0))

            z1 = model((x + h*pu).unsqueeze(0))
            z2 = model((x - h*pu).unsqueeze(0))

            score1 = ((z1 - c)**2).sum()
            score2 = ((z2 - c)**2).sum()

            gradient += (score1 - score2) / (2*h) * u

        gradient /= K

        # fractions = torch.tensor(fractions, dtype=torch.float32)
        # fractions = torch.clamp(fractions, min=0)
        # model.zero_grad()
        # score.backward()
        # grad = x.grad[:len(x)//2].detach().squeeze(0)
        fractions = torch.clamp(-gradient, min=0)
        if fractions.sum() == 0:
            fractions = torch.ones_like(fractions)
        fractions /= fractions.sum()
        # print(fractions)
        return fractions.tolist()
        
    return strategy

class ReconstructionModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=(32, 32)):
        super().__init__()
        dims = [in_dim] + list(hidden_dim) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i+1], bias=True)
            for i in range(len(dims) - 1)
        )
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
def make_reconstruction_data(log_returns, log_returns_mean, log_returns_std, num_mask=2):
    eps = 1e-3
    n_assets = log_returns.shape[1]

    # Normalize
    log_returns_normalized = (log_returns - log_returns_mean) / log_returns_std
    mean = np.mean(log_returns_normalized[:-1], axis=0)
    std = np.std(log_returns_normalized[:-1], axis=0) + eps

    input_data = np.concatenate((log_returns[-1], mean, std))

    # Randomly select multiple assets to mask
    mask_indices = np.random.choice(n_assets, size=num_mask, replace=False)
    mask = np.ones(n_assets)
    mask[mask_indices] = 0.0

    # Zero out masked assets in input
    # for idx in mask_indices:
    #     input_data[idx] = 0.0           # last day returns
    #     input_data[idx + n_assets] = 0.0  # mean
    #     input_data[idx + 2 * n_assets] = 0.0  # std

    # Append mask to input
    return np.concatenate((input_data, 1 - mask)), mask

def make_reconstruction_strategy(market_data, ticker_to_idx, feature_to_idx, training_fraction=0.8):
    n_tickers = len(ticker_to_idx.keys())
    window = 5
    eps = 1e-3

    batch_size = 256
    hidden_dims = (32,16)
    lr = 1e-4
    weight_decay = 1e-3
    epochs = 10000
    display_every = 100

    # make input data
    close_data = market_data[:, :, feature_to_idx["Close"]]
    log_returns = np.log(close_data[1:] / close_data[:-1])

    train_idx = int(len(log_returns)*training_fraction)
    log_returns_mean = np.mean(log_returns[:train_idx], axis=0)
    log_returns_std = np.std(log_returns[:train_idx], axis=0) + eps
    X = []
    masks = []
    for i in range(len(close_data) - window):
        for _ in range(5):  # create multiple samples per window
            x, mask = make_reconstruction_data(log_returns[i:i + window + 1], log_returns_mean, log_returns_std, num_mask=randint(1, 3))
            X.append(x)
            masks.append(mask)
    X = torch.tensor(X, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.float32)
    train_idx = int(len(X) * training_fraction)

    training_data = X[:train_idx]
    test_data = X[train_idx:]

    model = ReconstructionModel(in_dim=X.shape[1], out_dim=n_tickers, hidden_dim=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # loss_lambda = lambda batch, mask_idx, recon_batch: ((recon_batch - batch[:, :log_returns.shape[1]])[torch.arange(len(batch)), mask_idx] ** 2).mean()
    # loss_lambda = lambda batch, mask_idx, recon_batch: F.smooth_l1_loss(recon_batch * batch[:, :log_returns.shape[1]], batch[:, :log_returns.shape[1]] * batch[:, :log_returns.shape[1]], reduction="sum") / batch[:, :log_returns.shape[1]].sum()
    # def loss_lambda(batch, masks,recon_batch):
    #     n_assets = recon_batch.shape[1]
    #     masks = 1 - masks  # 1 for masked, 0 for unmasked
    #     loss = F.smooth_l1_loss(recon_batch * masks, batch[:, :n_assets] * masks, reduction="sum")
    #     return loss / (masks.sum() + 1e-8)
    
    # def compute_loss(batch, masks, model):
    #     # batch: original unmasked values
    #     # masks: 0=masked, 1=unmasked
    #     masked_batch = batch.clone()
    #     masked_batch[:, :n_tickers] *= masks
    #     # masked_batch[:, n_tickers:2*n_tickers] *= masks
    #     # masked_batch[:, 2*n_tickers:3*n_tickers] *= masks
        
    #     recon_batch = model(masked_batch)
    #     mask_for_loss = 1 - masks  # 1 where masked

    #     loss = F.smooth_l1_loss(recon_batch * mask_for_loss, batch[:, :n_tickers] * mask_for_loss, reduction="sum")
    #     return loss / (mask_for_loss.sum() + eps)
    
    def compute_loss(batch, masks, model,
                 lambda_mean=0.1,
                 lambda_var=0.1,
                 lambda_reversion=1.0):

        masked_batch = batch.clone()
        masked_batch[:, :n_tickers] *= masks

        recon = model(masked_batch)

        mask_for_loss = 1 - masks

        # Reconstruction (as before)
        recon_loss = F.smooth_l1_loss(
            recon * mask_for_loss,
            batch[:, :n_tickers] * mask_for_loss,
            reduction="sum"
        ) / (mask_for_loss.sum() + eps)

        # Residuals
        residual = (batch[:, :n_tickers] - recon) * mask_for_loss

        # Zero-mean residuals
        mean_penalty = residual.mean(dim=0).pow(2).mean()

        # Variance control
        var_penalty = residual.var(dim=0).mean()

        # Mean reversion (AR(1))
        if residual.shape[0] > 1:
            r_t = residual[:-1]
            r_tp1 = residual[1:]
            rho = (r_t * r_tp1).sum() / (r_t.pow(2).sum() + eps)
            reversion_penalty = rho.pow(2)
        else:
            reversion_penalty = 0.0

        return (
            recon_loss
            + lambda_mean * mean_penalty
            + lambda_var * var_penalty
            + lambda_reversion * reversion_penalty
        )


    test_masks = masks[train_idx:]
    for epoch in range(epochs):
        shuffled_indexes = torch.randperm(training_data.size(0))
        shuffled = training_data[shuffled_indexes]
        shuffled_masks = masks[shuffled_indexes]
        for i in range(0, len(training_data), batch_size):
            batch = shuffled[i:min(i+batch_size, len(training_data))]
            batch_masks = shuffled_masks[i:min(i+batch_size, len(training_data))]

            optimizer.zero_grad()
            loss = compute_loss(batch, batch_masks, model)
            loss.backward()
            optimizer.step()

        if epoch % display_every == 0:
            optimizer.zero_grad()

            test_loss = compute_loss(test_data, test_masks, model)
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

    def strategy(capital_allocation, market_data, ticker_to_idx, feature_to_idx):
        if market_data.shape[0] <= window + 1:
            return diversify(capital_allocation, market_data, ticker_to_idx, feature_to_idx)
        log_returns = np.log(market_data[1:, :, feature_to_idx["Close"]] / market_data[:-1, :, feature_to_idx["Close"]])[-window - 1:-1]
        log_returns_normalized = (log_returns - log_returns_mean) / log_returns_std
        mean = np.mean(log_returns_normalized[-window - 1:], axis=0)
        std = np.std(log_returns_normalized[-window - 1:], axis=0) + eps

        fractions = torch.zeros(n_tickers)
        input_data = np.concatenate((log_returns[-1], mean, std))
        for mask_idx in range(n_tickers):
            mask = np.zeros(n_tickers)
            mask[mask_idx] = 1.0
            this_input = input_data.copy()
            this_input = np.concatenate((this_input, mask))
            this_input[mask_idx] = 0.0  # mask one asset at a time

            this_input = torch.tensor(this_input, dtype=torch.float32)
            with torch.no_grad():
                recon = model(this_input.unsqueeze(0)).squeeze(0)
            fractions[mask_idx] = recon[mask_idx] - log_returns[-1, mask_idx]
        fractions = torch.clamp(fractions, min=0)
        if fractions.sum() == 0:
            fractions = torch.ones_like(fractions)
        fractions /= fractions.sum()
        return fractions.tolist()
    return strategy

    
class StatArbModel(nn.Module):
    def __init__(self, n_tickers, window, factor_hls=(32, 3)):
        super().__init__()
        self.n_tickers = n_tickers
        self.window = window
        dims = [window] + list(factor_hls)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())

        self.shared_mlp = nn.Sequential(*layers)
        self.B = nn.Linear(dims[-1], n_tickers, bias=False) # no bias on last layer (factor head to expected returns is linear)
    def forward(self, x):
        # x is (ticker1(t-n), ticker1(t-n+1),...,ticker2(t-n), ticker2(t-n+2),...)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_tickers, self.window) # (batch, n_tickers, window)
        factors = self.shared_mlp(x).mean(dim=1) # (batch, factors)
        out = self.B(factors) # (batch, tickers)
        return out, factors
    

class StatArbConv(nn.Module):
    def __init__(self, n_tickers, window, out_channels=8, kernel_size=5, factor_dims=[10, 4]):
        super().__init__()
        self.n_tickers = n_tickers
        self.window = window
        self.kernel_size = kernel_size
        # per-ticker convolution
        # (batch*n_tickers, 1, window) -> (batch*n_tickers, out_channels, L)
        self.convolve = nn.Sequential(
            nn.Conv1d(1, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
        )

        # cross ticker linear factor head
        dims = [self.window*out_channels*n_tickers] + factor_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())

        self.factor_mlp = nn.Sequential(*layers)

        # factor -> log returns
        self.B = nn.Linear(factor_dims[-1], n_tickers, bias=None)
    def forward(self, x):
        batch = x.shape[0]
        # prepare convolution view
        x = x.view(batch*self.n_tickers, 1, self.window)
        x = F.pad(x, (self.kernel_size-1, 0))

        x = self.convolve(x) # (batch*n_tickers, out_channels, L)
        x = x.view(batch, self.n_tickers*x.shape[1]*x.shape[2])
        f = self.factor_mlp(x)
        y = self.B(f)

        return y, f
    



def make_ml_statarb_strategy(market_data, ticker_to_idx, feature_to_idx, training_fraction=0.8):
    n_tickers = len(ticker_to_idx) # 7
    window = 10
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 10000000
    display_every = 1
    batch_size = 100
    horizon = 3

    model = StatArbConv(n_tickers, window)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # make input data
    close_data = market_data[:, :, feature_to_idx["Close"]]
    log_returns = np.log(close_data[1:] / close_data[:-1])
    train_idx = int(log_returns.shape[0]*training_fraction)
    mean = np.mean(log_returns[:train_idx], axis=0)
    std = np.std(log_returns[:train_idx], axis=0) + 1e-3
    log_returns = (log_returns - mean) / std # normalize

    input_data = np.zeros((len(close_data) - window - horizon - 1, n_tickers*window))
    for t in range(input_data.shape[0]):
        window_slice = log_returns[t:t + window].T
        input_data[t] = window_slice.flatten() # (ticker1(t-n), ticker1(t-n+1),...,ticker2(t-n), ticker2(t-n+2),...)
    input_data = torch.tensor(input_data, dtype=torch.float32)
    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)
    target_data = sum(log_returns_tensor[window + i:i - horizon] for i in range(horizon))


    train_idx = int(input_data.shape[0]*training_fraction)
    training_data = input_data[:train_idx]
    training_targets = target_data[:train_idx]
    test_data = input_data[train_idx:]
    test_targets = target_data[train_idx:]

    lambda_fac_slow = 0.05
    lambda_fac_ortho = 0.1
    lambda_res_var = 0.0
    lambda_res_orhto = 0.0
    def loss(preds, factors, targets):
        loss_mse = F.mse_loss(preds, targets)
        slow_factors_reg = F.mse_loss(factors[1:] - factors[:-1], torch.zeros_like(factors[1:])) # factors should be slow moving
        residuals = preds - targets
        res_var_reg = residuals.var(dim=0, unbiased=False).mean() # residuals should have low variance; be anamolous

        # encourage factors to be orthogonal
        T, K = factors.shape
        C = (factors.T @ factors) / T
        I = torch.eye(K)
        fac_ortho_reg = ((C**2)*(1 - I)).sum() / (K*(K-1))

        # encourage residuals to be orthogonal
        T, K = residuals.shape
        C = (residuals.T @ residuals) / T
        I = torch.eye(K)
        res_ortho_reg = ((C**2)*(1 - I)).sum() / (K*(K-1))

        return loss_mse + lambda_fac_slow*slow_factors_reg + lambda_res_var*res_var_reg + lambda_fac_ortho*fac_ortho_reg + lambda_res_orhto*res_ortho_reg
    for epoch in range(epochs):
        shuffled_indexes = torch.randperm(training_data.size(0))
        shuffled = training_data[shuffled_indexes]
        shuffled_targets = training_targets[shuffled_indexes]

        for i in range(0, len(training_data), batch_size):
            batch = shuffled[i:min(i+batch_size, len(training_data))]
            targets = shuffled_targets[i:min(i+batch_size, len(training_data))]
            optimizer.zero_grad()
            preds, factors = model(batch)
            train_loss = F.mse_loss(preds, targets)
            train_loss.backward()
            optimizer.step()

        if epoch % display_every == 0:
            # with torch.no_grad():
            #     preds, factors = model(training_data)
            #     residuals = preds - training_targets
            #     residuals_np = residuals.numpy()  # shape (batch, n_tickers)
            #     col1 = residuals_np[:, 0]
            #     col2 = residuals_np[:, 1]
            #     corr = np.corrcoef(col1, col2)[0, 1]
            #     print("Factor std:", factors.std(dim=0))
            #     print("Residual std:", residuals.std(dim=0))
            #     print("Residual correlation:", corr)

            with torch.no_grad():
                test_preds, test_factors = model(test_data)
                factor_var = test_factors.var(dim=0)
                test_loss = loss(test_preds, test_factors, test_targets)
                print(f"Epoch {epoch}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}")#, Factor Var: {factor_var}")
