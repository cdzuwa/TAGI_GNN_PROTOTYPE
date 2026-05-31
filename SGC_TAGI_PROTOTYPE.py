# =========================================
# SGC-TAGI Multi-Session Experiment Script
# =========================================
import os
import time
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, WebKB
from torch_geometric.transforms import NormalizeFeatures, RandomNodeSplit
import torch_geometric.transforms as T
from tqdm.auto import tqdm

# ----------------------------
# Utilities
# ----------------------------
def masked_accuracy(mu_y, y, mask):
    idx = np.where(mask)[0]
    correct = 0
    for i in idx:
        correct += (np.argmax(mu_y[i]) == np.argmax(y[i]))
    return correct / len(idx)

def normalize_adjacency_sparse(edge_index, num_nodes):
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row))
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + sp.eye(num_nodes)
    A = A.tocoo()
    deg = np.array(A.sum(1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    A.data = deg_inv_sqrt[A.row] * A.data * deg_inv_sqrt[A.col]
    return A.tocsr()

def tsne_visualization(model, graph, var_v, mask_name="test", perplexity=30, save_path=None):
    cache = model.complete_forward_pass(graph["X"], var_v)
    mu_zL = cache[model.n_layers - 1]["mu_z"].squeeze(-1)
    labels = np.argmax(graph["y"], axis=1)
    mask = {"train": graph["train_mask"], "val": graph["val_mask"], "test": graph["test_mask"]}[mask_name]

    X_emb = mu_zL[mask]
    y_emb = labels[mask]

    X_emb = StandardScaler().fit_transform(X_emb)
    if X_emb.shape[1] > 50:
        X_emb = PCA(n_components=50).fit_transform(X_emb)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init="pca", random_state=42)
    Z = tsne.fit_transform(X_emb)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=y_emb, cmap="tab10", s=20, alpha=0.8)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"t-SNE saved to {save_path}")
    else:
        plt.show()

# ----------------------------
# Load PyG Graph + SGC features
# ----------------------------
def load_graph_pyg(dataset_class, name=None, data_root="./data", seed=42, k=2, normalize_features=True,
                   use_random_split=False, split_params=None, aug_features=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transforms = []
    if normalize_features:
        transforms.append(NormalizeFeatures())
    if use_random_split:
        if split_params is None:
            split_params = dict(split="random", num_train_per_class=20, num_val=500, num_test=1000)
        transforms.append(RandomNodeSplit(**split_params))

    transform = T.Compose(transforms) if len(transforms) > 0 else None
    dataset = dataset_class(root=data_root, name=name, transform=transform)
    data = dataset[0]
    N = data.num_nodes

    X0 = data.x.cpu().numpy() if data.x is not None else np.eye(N)
    y = data.y.cpu().numpy()
    num_classes = int(y.max()) + 1
    Y_onehot = np.eye(num_classes)[y]

    A_norm = normalize_adjacency_sparse(data.edge_index, N)

    # SIGN-style propagation
    X_prev = X0
    if aug_features:
        X_list = [X0]
        for _ in range(k):
            X_prev = A_norm @ X_prev
            X_list.append(X_prev)
        X_cat = np.concatenate(X_list, axis=1)
    else:
        for _ in range(k):
            X_prev = A_norm @ X_prev
        X_cat = X_prev

    graph_dict = {"X": X_cat, "y": Y_onehot,
                  "train_mask": data.train_mask.cpu().numpy(),
                  "val_mask": data.val_mask.cpu().numpy(),
                  "test_mask": data.test_mask.cpu().numpy()}
    return [graph_dict], X_cat.shape[1], num_classes

# ----------------------------
# G-TAGI Network
# ----------------------------
class GTagiNetwork:
    def __init__(self, k, layer_sizes):
        self.k = k
        self.layers = layer_sizes
        self.n_layers = len(layer_sizes)
        self.mean_w, self.sigma_w, self.mean_b, self.sigma_b = {}, {}, {}, {}
        for l in range(1, self.n_layers):
            in_dim = layer_sizes[l-1]
            out_dim = layer_sizes[l]
            self.mean_w[l] = np.random.normal(0, np.sqrt(1/max(1,in_dim)), (out_dim, in_dim))
            self.sigma_w[l] = (1/max(1,in_dim))*np.ones((out_dim, in_dim))
            self.mean_b[l] = np.random.normal(0, np.sqrt(1/max(1,in_dim)), (out_dim,1))
            self.sigma_b[l] = (1/max(1,in_dim))*np.ones((out_dim,1))

    # ----------------------------
    # Feedforward + observation
    # ----------------------------
    def feedforward_graph(self, X_nodes):
        n_nodes = X_nodes.shape[0]
        mu_a_prev = X_nodes.reshape(n_nodes, X_nodes.shape[1],1)
        var_a_prev = np.zeros_like(mu_a_prev)
        cache = {}
        for l in range(1,self.n_layers):
            mu_z, var_z, mu_a_agg, var_a_agg = self.mean_var_layer_graph(
                self.mean_w[l], self.sigma_w[l], mu_a_prev, var_a_prev, self.mean_b[l], self.sigma_b[l]
            )
            mu_a, var_a = mu_z, var_z
            var_b_z = self.sigma_b[l]
            var_w_z_nodes = self.cov_w_z_next_graph(self.sigma_w[l], mu_a_agg)
            cache[l] = {"mu_z": mu_z, "var_z": var_z,
                        "mu_a_agg": mu_a_agg, "var_a_agg": var_a_agg,
                        "var_w_z_nodes": var_w_z_nodes, "var_b_z": var_b_z}
            mu_a_prev, var_a_prev = mu_a, var_a
        return cache

    def complete_forward_pass(self, X_nodes, var_v):
        cache = self.feedforward_graph(X_nodes)
        mu_zL = cache[self.n_layers-1]["mu_z"]
        var_zL = cache[self.n_layers-1]["var_z"]
        mu_y, var_y = self.observation(mu_zL, var_zL, var_v)
        cache["output"] = {"mean_y": mu_y, "sigma_y": var_y}
        return cache

    def mean_var_layer_graph(self, mu_w, var_w, mu_node_feats, var_node_feats, mu_b, var_b):
        n_nodes = mu_node_feats.shape[0]
        in_dim = mu_w.shape[1]
        out_dim = mu_w.shape[0]
        mu_a_flat = mu_node_feats.reshape(n_nodes, in_dim)
        var_a_flat = var_node_feats.reshape(n_nodes, in_dim)
        term1 = (mu_w*mu_w)@var_a_flat.T
        term1 = term1.T
        term2 = var_w@var_a_flat.T
        term2 = term2.T
        term3 = var_w@(mu_a_flat*mu_a_flat).T
        term3 = term3.T
        mu_z_flat = mu_a_flat@mu_w.T + mu_b.reshape(1,out_dim)
        var_z_flat = term1 + term2 + term3 + var_b.reshape(1,out_dim)
        mu_z = mu_z_flat.reshape(n_nodes, out_dim,1)
        var_z = var_z_flat.reshape(n_nodes, out_dim,1)
        return mu_z, var_z, mu_a_flat, var_a_flat

    def cov_w_z_next_graph(self, var_w, mu_a_agg_nodes):
        N, in_dim = mu_a_agg_nodes.shape
        out_dim, in_dim2 = var_w.shape
        mu_a = mu_a_agg_nodes.reshape(N, in_dim,1)
        Wt = var_w.T.reshape(1, in_dim, out_dim)
        return mu_a*Wt

    def observation(self, mu_z, var_z, sigma_v):
        return mu_z, var_z + sigma_v

    # ----------------------------
    # Masked updates + parameters
    # ----------------------------
    def update_last_GCN_masked(self, cache, y, train_mask, sigma_z, mean_z):
        mu_y = cache["output"]["mean_y"]
        sigma_y = cache["output"]["sigma_y"]
        mask = train_mask[:,None,None].astype(float)
        y_exp = y[:,:,None]
        delta_y = y_exp - mu_y
        K = sigma_z / sigma_y
        mean_z_y = mean_z + mask*K*delta_y
        sigma_z_y = sigma_z - mask*K*sigma_z
        return mean_z_y, sigma_z_y

    def update_parameters(self, cache, updated_z, lr=0.1):
        for l in reversed(range(1,self.n_layers)):
            mu_z = cache[l]["mu_z"]
            sigma_z = cache[l]["var_z"]
            mu_z_y, sigma_z_y = updated_z[l]
            n_nodes = mu_z.shape[0]
            out_dim = mu_z.shape[1]
            sigma_b = self.sigma_b[l]
            sigma_b_z = cache[l]["var_b_z"]
            sigma_z_nodes = sigma_z.reshape(n_nodes,out_dim)
            sigma_z_inv_nodes = 1.0/sigma_z_nodes
            jb = sigma_b_z*sigma_z_inv_nodes.T
            mean_b_nodes = jb.T*(mu_z_y.reshape(n_nodes,out_dim)-mu_z.reshape(n_nodes,out_dim))
            sigma_b_nodes = jb.T*(sigma_z_y.reshape(n_nodes,out_dim)-sigma_z.reshape(n_nodes,out_dim))*jb.T
            avg_mean_b = lr*np.mean(mean_b_nodes,axis=0).reshape(-1,1)
            avg_sigma_b = lr*np.mean(sigma_b_nodes,axis=0).reshape(-1,1)
            self.mean_b[l] += avg_mean_b
            self.sigma_b[l] = np.abs(self.sigma_b[l]+avg_sigma_b)
            cov_w_z_nodes = cache[l]["var_w_z_nodes"]
            sum_w_mean, sum_w_sigma = np.zeros(self.mean_w[l].shape), np.zeros(self.mean_w[l].shape)
            for node in range(n_nodes):
                jw = cov_w_z_nodes[node].T*sigma_z_inv_nodes[node].reshape(-1,1)
                delta_mu_z = mu_z_y[node]-mu_z[node]
                sum_w_mean += jw*delta_mu_z
                sigma_z_diff = sigma_z_y[node]-sigma_z[node]
                sum_w_sigma += jw*sigma_z_diff*jw
            self.mean_w[l] += lr*sum_w_mean
            self.sigma_w[l] = np.abs(self.sigma_w[l]+lr*sum_w_sigma)

    # ----------------------------
    # Training
    # ----------------------------
    def train_on_graph(self, X_nodes, y, train_mask, var_v, lr):
        cache = self.complete_forward_pass(X_nodes,var_v)
        mu_y = cache["output"]["mean_y"]
        sigma_y = cache["output"]["sigma_y"]
        mean_z = cache[self.n_layers-1]["mu_z"]
        sigma_z = cache[self.n_layers-1]["var_z"]
        ll_elementwise = np.log(2*np.pi*sigma_y)+(y[:,:,None]-mu_y)**2/sigma_y
        mask = train_mask[:,None,None].astype(float)
        ll = -0.5*np.sum(mask*ll_elementwise)
        mean_z_y, sigma_z_y = self.update_last_GCN_masked(cache,y,train_mask,sigma_z,mean_z)
        updated_z = {self.n_layers-1:(mean_z_y,sigma_z_y)}
        self.update_parameters(cache, updated_z, lr=lr)
        return ll, mu_y

    def fit(self, graphs, n_epochs=10, var_v=1.0, verbose=True, lr=0.1):
        history = {"ll":[],"train_acc":[],"val_acc":[],"test_acc":[]}
        g = graphs[0]
        for ep in tqdm(range(n_epochs), desc="ep"):
            ll, mu_y = self.train_on_graph(g["X"], g["y"], g["train_mask"], var_v, lr)
            history["ll"].append(ll)
            history["train_acc"].append(masked_accuracy(mu_y,g["y"],g["train_mask"]))
            history["val_acc"].append(masked_accuracy(mu_y,g["y"],g["val_mask"]))
            history["test_acc"].append(masked_accuracy(mu_y,g["y"],g["test_mask"]))
            if verbose:
                print(f"Epoch {ep+1:03d} | LL={ll:.3f} | Train={history['train_acc'][-1]:.4f} | Val={history['val_acc'][-1]:.4f} | Test={history['test_acc'][-1]:.4f}")
        return history

# ----------------------------
# Multi-session experiment
# ----------------------------
def run_multi_session_experiment(dataset_class, name=None, n_sessions=10, n_epochs=200, k=2, lr=1, var_v=1e-2,
                                 split_params=None, aug_features=True, results_dir="./results"):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    all_results = []
    total_start_time = time.time()

    for session in range(n_sessions):
        seed = 42 + session
        print(f"\n========== Session {session+1}/{n_sessions} | Seed={seed} ==========")
        session_start_time = time.time()

        train_graphs, in_dim, out_dim = load_graph_pyg(dataset_class=dataset_class, name=name, seed=seed, k=k,
                                                       use_random_split=True, split_params=split_params, aug_features=aug_features)
        model = GTagiNetwork(k,[in_dim,out_dim])
        history = model.fit(train_graphs, n_epochs=n_epochs, var_v=var_v, verbose=False, lr=lr)

        session_end_time = time.time()
        session_duration = session_end_time - session_start_time

        # Save per-epoch losses
        loss_df = pd.DataFrame({"epoch": np.arange(1,n_epochs+1),
                                "log_likelihood": history["ll"],
                                "train_acc": history["train_acc"],
                                "val_acc": history["val_acc"],
                                "test_acc": history["test_acc"]})
        loss_csv_path = os.path.join(results_dir, f"{name}_session{session+1}_losses.csv")
        loss_df.to_csv(loss_csv_path, index=False)
        print(f"Saved training logs to {loss_csv_path}")

        # t-SNE
        tsne_path = os.path.join(results_dir, f"{name}_session{session+1}_tsne.png")
        tsne_visualization(model, train_graphs[0], var_v=var_v, mask_name="test", save_path=tsne_path)

        best_epoch = int(np.argmax(history["val_acc"]))
        best_val = history["val_acc"][best_epoch]
        best_test = history["test_acc"][best_epoch]

        all_results.append({"session": session+1, "seed":seed, "best_epoch":best_epoch+1,
                            "best_val_acc":float(best_val), "test_acc_at_best_val":float(best_test),
                            "train_time_sec": session_duration})

        print(f"Session {session+1} | Best Epoch={best_epoch+1} | Val={best_val:.4f} | Test={best_test:.4f} | Time={session_duration:.2f}s")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTotal time for {n_sessions} sessions: {total_duration:.2f}s")

    # Aggregate summary
    df = pd.DataFrame(all_results)
    mean_test = df["test_acc_at_best_val"].mean()
    std_test = df["test_acc_at_best_val"].std()
    mean_time = df["train_time_sec"].mean()
    std_time = df["train_time_sec"].std()
    summary = pd.DataFrame([{"session":"mean","best_epoch":"-","best_val_acc":"-","test_acc_at_best_val":mean_test,"train_time_sec":mean_time},
                            {"session":"std","best_epoch":"-","best_val_acc":"-","test_acc_at_best_val":std_test,"train_time_sec":std_time}])
    df = pd.concat([df,summary],ignore_index=True)
    save_path = os.path.join(results_dir,f"{name}_all_sessions_summary.csv")
    df.to_csv(save_path,index=False)
    print(f"Saved summary to: {save_path}")
    return df

# ----------------------------
# Main
# ----------------------------
if __name__=="__main__":
    split_params = dict(split="train_rest", num_val=0.2, num_test=0.2)
    df_summary = run_multi_session_experiment(dataset_class=Amazon, name="photo", n_sessions=1, n_epochs=200,
                                              k=2, lr=1, var_v=1e-2, split_params=split_params,
                                              aug_features=True, results_dir="./results")