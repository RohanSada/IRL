import pickle
import joblib
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr, kendalltau
matplotlib.use('MacOSX')

# BNN model definition
class BNN(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = PyroModule[torch.nn.Linear](input_dim, 64)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([64, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))

        self.fc2 = PyroModule[torch.nn.Linear](64, 64)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([64, 64]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))

        self.fc3 = PyroModule[torch.nn.Linear](64, 1)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, 64]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x).squeeze(-1)
        sigma = 0.001
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(out, sigma), obs=y)
        return out

if __name__ == '__main__':

    # Load expert data
    trajectories = joblib.load("expert_trajectories.pkl")

    # Flatten into (state, next_state) pairs
    all_states, all_next_states = [], []
    for episode in trajectories:
        for (s_t, _), (s_tp1, _) in zip(episode[:-1], episode[1:]):
            all_states.append(s_t)
            all_next_states.append(s_tp1)
    all_states = np.array(all_states, dtype=np.float32)
    all_next_states = np.array(all_next_states, dtype=np.float32)

    # Reward shaping
    def composite_reward(state, lengthscale=5.0, goal_weight=1.0, obs_weight=0.2):
        dist_to_goal = state[4]
        min_obstacle = np.min(state[6:])
        goal_term = np.exp(- (dist_to_goal / lengthscale)**2)
        obstacle_term = 1.0 / (1.0 + (min_obstacle / 50.0)**2)
        return goal_weight * goal_term + obs_weight * obstacle_term

    # Updated shaping reward
    gamma = 0.99
    time_penalty = 0.02
    goal_tol = 0.5
    goal_bonus = 30.0          
    reward_exponent = 2.0      

    def shaped_reward(state, next_state):
        base_r = composite_reward(state) - time_penalty
        phi = -state[4]
        phi_next = -next_state[4]
        shaped = base_r + (gamma * phi_next - phi)
        if next_state[4] < goal_tol:
            shaped += goal_bonus
        return shaped ** reward_exponent

    # Compute reward targets
    y = np.array([
        shaped_reward(s, s_next)
        for s, s_next in zip(all_states, all_next_states)
    ], dtype=np.float32)

    # Feature engineering: + is_goal_reached
    all_actions = []
    for episode in trajectories:
        for (s_t, a_t), _ in zip(episode[:-1], episode[1:]):
            all_actions.append(a_t)
    all_actions = np.array(all_actions, dtype=np.float32)

    dist_to_goal = all_states[:, [4]]
    lidar_min = np.min(all_states[:, 6:], axis=1, keepdims=True)
    lidar_mean = np.mean(all_states[:, 6:], axis=1, keepdims=True)
    lidar_std = np.std(all_states[:, 6:], axis=1, keepdims=True)
    action_delta = all_actions[:, [1]]
    goal_binary = (all_next_states[:, 4:5] < goal_tol).astype(np.float32)  

    X_reduced = np.hstack([
        dist_to_goal,
        lidar_min,
        lidar_mean,
        lidar_std,
        action_delta,
        goal_binary
    ])  # shape = (N, 6)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # Prepare DataLoader
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    n_train = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Instantiate model
    input_dim = X_tensor.shape[1]
    model = BNN(input_dim)
    guide = AutoDiagonalNormal(model)
    optim = pyro.optim.Adam({"lr": 1e-4})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    # Train
    losses = []
    for epoch in range(1000):
        total_loss = 0.0
        for xb, yb in train_loader:
            total_loss += svi.step(xb, yb)
        losses.append(total_loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:<3}  ELBO loss = {total_loss:.2f}")

    # Plot training loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("SVI Loss")
    plt.title("Training ELBO")
    plt.grid(True)
    plt.show()

    # Evaluate
    predictive = Predictive(model, guide=guide, num_samples=100)
    xt, yt = zip(*test_ds)
    x_test = torch.stack(xt)
    y_test = torch.stack(yt).numpy()
    samples = predictive(x_test)
    y_pred = samples["obs"].mean(0).detach().numpy()
    mse = np.mean((y_test - y_pred)**2)
    r2 = linregress(y_test, y_pred).rvalue**2
    print(f"Test MSE = {mse:.4f},  R² = {r2:.4f}")
    rho, _ = spearmanr(y_test, y_pred)
    tau, _ = kendalltau(y_test, y_pred)
    print(f"Spearman ρ = {rho:.4f}, Kendall τ = {tau:.4f}")

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("True Shaped+Bonus Reward")
    plt.ylabel("Predicted Reward")
    plt.title("BNN Reward Prediction")
    plt.grid(True)
    plt.show()

    # Save artifacts
    torch.save(guide, "bnn_reward_guide.pt")
    joblib.dump(scaler, "reward_scaler.pkl")