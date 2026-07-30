########################################################################################
#
# The code in this file is copied from
#
# https://github.com/samholt/generative-simulations/blob/72b5d51a7790b8b2ebc87973ee5e9d21aa818ece/libs/SIR/env.py
#
# This code was written by Samuel Holt and released under the MIT license:
#
# MIT License
#
# Copyright (c) 2025 Samuel Holt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################################

# --- Required imports for robust implementation ---
import signal
import traceback
from typing import Dict, Tuple

import numpy as np
import ot  # POT library
import ray
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver

# sbi imports
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from tqdm import tqdm


class DotDict(dict):
    def __getattr__(self, name):
        return self.get(name, None)
    def __setattr__(self, name, value):
        self[name] = value

def to_dot_dict(d):
    dot_dict = DotDict()
    for key, value in d.items():
        if isinstance(value, dict):
            dot_dict[key] = to_dot_dict(value)
        else:
            dot_dict[key] = value
    return dot_dict

def wasserstein_distance_nd(X, Y):
    N, M = X.shape[0], Y.shape[0]
    X, Y = X.reshape(N, -1), Y.reshape(M, -1)
    cost_matrix = ot.dist(X, Y, metric="euclidean")
    a, b = np.ones(N) / N, np.ones(M) / M
    transport_plan = ot.emd(a, b, cost_matrix)
    return np.sum(cost_matrix * transport_plan)

def rbf_kernel(X, Y, sigma=1.0):
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    distances = XX + YY - 2 * np.dot(X, Y.T)
    return np.exp(-distances / (2 * sigma**2))

def compute_mmd(X, Y, sigma=1.0):
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    N, M = X.shape[0], Y.shape[0]
    if N < 2 or M < 2: return np.nan
    
    Kxx, Kyy, Kxy = rbf_kernel(X, X, sigma), rbf_kernel(Y, Y, sigma), rbf_kernel(X, Y, sigma)
    sumKxx = np.sum(Kxx) - np.sum(np.diag(Kxx))
    sumKyy = np.sum(Kyy) - np.sum(np.diag(Kyy))
    sumKxy = np.sum(Kxy)
    return sumKxx / (N * (N - 1)) + sumKyy / (M * (M - 1)) - 2 * sumKxy / (N * M)

class SimulatorStep:
    def __init__(self):
        self.parameters = np.array([0.5, 0.1], dtype=float)

    def step(self, state: dict, action: int, rng=None) -> dict:
        S, I, R = state["S"], state["I"], state["R"]
        beta, gamma = self.parameters
        N = S + I + R
        if N <= 0: return {"S": 0, "I": 0, "R": 0}
        
        # Define the infection probability
        prob_infection = np.clip(1.0 - np.exp(-beta * I / N), 0.0, 1.0)
        
        if rng is None: rng = np.random.default_rng()
        
        # --- FIXED HERE: Used correct variable name 'prob_infection' ---
        new_infections = rng.binomial(S, prob_infection) if S > 0 else 0
        new_recoveries = rng.binomial(I, gamma) if I > 0 else 0
        
        return {"S": S - new_infections, "I": I + new_infections - new_recoveries, "R": R + new_recoveries}

    def get_parameters(self) -> np.ndarray: return self.parameters
    def set_parameters(self, parameters: np.ndarray): self.parameters = parameters.astype(float)
    def get_parameters_uniform_prior_min_max(self) -> np.ndarray: return np.array([[0.0, 0.0], [2.0, 1.0]], dtype=float)

    def _create_predicted_and_true_trajectories(self, Model, params_np: np.ndarray, loss_states: list, T: int) -> Tuple[np.ndarray, np.ndarray]:
        simulator = Model()
        simulator.set_parameters(params_np)
        trajectories = []
        for data_traj in loss_states:
            states = [data_traj[0]]
            for t in range(T):
                states.append(simulator.step(state=states[-1], action=0, rng=np.random.default_rng()))
            trajectories.append(states)
        t_data = trajectories_to_numpy(loss_states)
        t_re_created_data = trajectories_to_numpy(trajectories)
        return t_data, t_re_created_data

    def _wass_and_mmd_distance(self, Model, params_np: np.ndarray, loss_states: list, mmd_sigma: float) -> Tuple[float, float]:
        simulator = Model()
        gt_simulator = SimulatorStep()
        simulator.set_parameters(params_np)
        N_SAMPLES = 200
        wdist_total, mmdist_total = [], []
        for data_traj in tqdm(loss_states, desc="WASS/MMD", leave=False, ncols=80):
            for traj_state in data_traj:
                samples = [simulator.step(state=traj_state, action=0, rng=np.random.default_rng()) for _ in range(N_SAMPLES)]
                gt_samples = [gt_simulator.step(state=traj_state, action=0, rng=np.random.default_rng()) for _ in range(N_SAMPLES)]
                samples_np, gt_samples_np = trajectories_to_numpy(samples), trajectories_to_numpy(gt_samples)
                wdist_total.append(wasserstein_distance_nd(samples_np, gt_samples_np))
                mmdist_total.append(compute_mmd(samples_np, gt_samples_np, mmd_sigma))
        return np.mean(wdist_total), np.mean(mmdist_total)

    def _compute_all_losses_with_states(self, Model, params_np: np.ndarray, loss_states: list, T: int, mmd_sigma: float, full=False) -> Dict:
        true_x, predicted_x = self._create_predicted_and_true_trajectories(Model, params_np, loss_states, T)
        mse = np.mean(np.square(true_x - predicted_x))
        if full:
            wass, mmd = self._wass_and_mmd_distance(Model, params_np, loss_states, mmd_sigma)
            return dict(mse=mse, mmd=mmd, wass=wass)
        return dict(mse=mse)
    
    def _compute_mse_per_dimension_with_states(self, Model, params_np: np.ndarray, loss_states: list, T: int) -> np.ndarray:
        true_x, predicted_x = self._create_predicted_and_true_trajectories(Model, params_np, loss_states, T)
        return np.mean(np.square(true_x - predicted_x), axis=(0, 1))

    def evaluate_simulator_code_wrapper(self, model, train_data, val_data, test_data, config={}, logger=None, env_name=""):
        if config.run.optimizer == "ES":
            results = self.evaluate_simulator_code_using_es(model, train_data, val_data, test_data, config, logger, env_name)
        elif config.run.optimizer == "SBI":
            results = self.evaluate_simulator_code_using_sbi(model, train_data, val_data, test_data, config, logger, env_name)
        else:
            raise ValueError(f"Unknown optimizer: {config.run.optimizer}")
        
        train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = results
        
        try:
            num_dims_in_loss = len(loss_per_dim)
            if env_name == "SIR" and num_dims_in_loss == 3:
                loss_per_dim_dict = {"S": loss_per_dim[0], "I": loss_per_dim[1], "R": loss_per_dim[2]}
            else:
                loss_per_dim_dict = {f"dim_{i}": v for i, v in enumerate(loss_per_dim)}
        except (TypeError, IndexError):
            loss_per_dim_dict = {"error": "could not unpack loss_per_dim", "value": loss_per_dim}

        return train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss

    def evaluate_simulator_code_using_es(self, Model, train_data, val_data, test_data, config, logger, env_name):
        if not ray.is_initialized(): ray.init()
        train_states, _ = train_data
        meta_simulator = Model()
        params = meta_simulator.get_parameters()
        param_bounds = meta_simulator.get_parameters_uniform_prior_min_max()
        T = len(train_states[0]) - 1

        if config.run.optimize_params:
            def function_to_optimize(normalized_params: torch.Tensor) -> torch.Tensor:
                norm_params_np = np.clip(normalized_params.cpu().numpy(), -1, 1)
                traj_params = param_bounds[0] + (param_bounds[1] - param_bounds[0]) * (norm_params_np + 1) / 2
                losses = self._compute_all_losses_with_states(Model, traj_params, train_states, T, config.run.mmd_sigma)
                return torch.tensor(losses['mse'])

            problem = Problem("min", function_to_optimize, initial_bounds=(-1.0, 1.0), solution_length=len(params), vectorized=False, num_actors=ray.available_resources().get("CPU", 1))
            searcher = GeneticAlgorithm(problem, popsize=200, operators=[SimulatedBinaryCrossOver(problem, tournament_size=4, cross_over_rate=1.0, eta=8), GaussianMutation(problem, stdev=0.03)])
            StdOutLogger(searcher, interval=10)
            searcher.run(config.run.es_generations)
            optimized_norm_params = searcher.status.get("best").values
            optimized_params_np = param_bounds[0] + (param_bounds[1] - param_bounds[0]) * (np.clip(optimized_norm_params.cpu().numpy(), -1, 1) + 1) / 2
        else:
            optimized_params_np = meta_simulator.get_parameters()

        val_states, _ = val_data
        test_states, _ = test_data
        train_losses = self._compute_all_losses_with_states(Model, optimized_params_np, train_states, T, config.run.mmd_sigma, full=True)
        val_losses = self._compute_all_losses_with_states(Model, optimized_params_np, val_states, T, config.run.mmd_sigma, full=True)
        test_losses = self._compute_all_losses_with_states(Model, optimized_params_np, test_states, T, config.run.mmd_sigma, full=True)
        val_loss_per_dim = self._compute_mse_per_dimension_with_states(Model, optimized_params_np, val_states, T)
        
        return train_losses, val_losses, optimized_params_np.tolist(), val_loss_per_dim, test_losses

    def evaluate_simulator_code_using_sbi(self, Model, train_data, val_data, test_data, config, logger, env_name):
        sbi_config = {
            'num_simulations': config.get("run", {}).get("sbi_num_simulations", 1000),
            'num_samples_posterior': config.get("run", {}).get("sbi_num_samples_posterior", 10000),
            'sampling_timeout': config.get("run", {}).get("sbi_sampling_timeout", 60),
        }
        train_states, _ = train_data
        val_states, _ = val_data
        test_states, _ = test_data
        T = len(train_states[0]) - 1

        observed_trajectory_np = trajectories_to_numpy(train_states[0])
        x_o = torch.tensor(observed_trajectory_np, dtype=torch.float32).reshape(1, -1)

        meta_simulator = Model()
        param_bounds = meta_simulator.get_parameters_uniform_prior_min_max()
        low = torch.tensor(param_bounds[0], dtype=torch.float32)
        high = torch.tensor(param_bounds[1], dtype=torch.float32)

        def _simulation_wrapper(params: torch.Tensor) -> torch.Tensor:
            simulated_trajectories = []
            for p in params:
                _, predicted_trajectory_np = self._create_predicted_and_true_trajectories(Model, p.cpu().numpy(), [train_states[0]], T)
                simulated_trajectories.append(predicted_trajectory_np)
            batch_output = torch.from_numpy(np.array(simulated_trajectories)).float()
            return batch_output.reshape(batch_output.shape[0], -1)
        
        nan_results = ({'mse': np.nan, 'mmd': np.nan, 'wass': np.nan}, {'mse': np.nan, 'mmd': np.nan, 'wass': np.nan}, [np.nan] * len(low), np.array([np.nan] * 3), {'mse': np.nan, 'mmd': np.nan, 'wass': np.nan})
        
        try:
            prior = utils.torchutils.BoxUniform(low=low, high=high)
            print(f"Generating {sbi_config['num_simulations']} simulations for SBI...")
            theta, x = simulate_for_sbi(_simulation_wrapper, proposal=prior, num_simulations=sbi_config['num_simulations'], show_progress_bar=True)

            nan_mask = torch.isnan(x).any(dim=1)
            num_nans = nan_mask.sum().item()
            if num_nans > 0:
                print(f"WARNING: Removed {num_nans} out of {len(theta)} simulations containing NaNs.")
                if num_nans == len(theta):
                    print("ERROR: All simulations resulted in NaNs. Aborting SBI.")
                    return nan_results
                theta, x = theta[~nan_mask], x[~nan_mask]
            
            print("Training SBI posterior estimator...")
            inference = NPE(prior=prior, show_progress_bars=True)
            density_estimator = inference.append_simulations(theta, x).train()
            posterior = inference.build_posterior(density_estimator)
            print("SBI training successful.")
        except Exception as e:
            print(f"ERROR during SBI training: {type(e).__name__}: {traceback.format_exc()}")
            return nan_results

        posterior_samples = None
        def _timeout_handler(signum, frame): raise TimeoutError(f"timed out after {sbi_config['sampling_timeout']}s.")
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(sbi_config['sampling_timeout'])
            print(f"Sampling {sbi_config['num_samples_posterior']} samples from the posterior...")
            posterior_samples = posterior.sample((sbi_config['num_samples_posterior'],), x=x_o)
            print("Posterior sampling successful.")
        except (TimeoutError, Exception) as e:
            print(f"WARNING: Could not sample from posterior. Reason: {e}")
        finally:
            signal.alarm(0)

        if posterior_samples is None:
            print("Could not obtain posterior samples. Aborting evaluation.")
            return nan_results
            
        print("Evaluating using the posterior mean of the parameters...")
        optimized_parameters_np = posterior_samples.mean(dim=0).cpu().numpy()
        
        train_losses = self._compute_all_losses_with_states(Model, optimized_parameters_np, train_states, T, config.run.mmd_sigma, full=True)
        val_losses = self._compute_all_losses_with_states(Model, optimized_parameters_np, val_states, T, config.run.mmd_sigma, full=True)
        test_losses = self._compute_all_losses_with_states(Model, optimized_parameters_np, test_states, T, config.run.mmd_sigma, full=True)
        val_loss_per_dim = self._compute_mse_per_dimension_with_states(Model, optimized_parameters_np, val_states, T)

        return train_losses, val_losses, optimized_parameters_np.tolist(), val_loss_per_dim, test_losses


def trajectories_to_numpy(data) -> np.ndarray:
    if data is None or len(data) == 0: return np.array([])
    if isinstance(data, np.ndarray): return np.squeeze(data)
    if isinstance(data[0], dict): data = [data]
    
    n_traj, traj_len = len(data), len(data[0])
    arr = np.zeros((n_traj, traj_len, 3), dtype=float)
    for i in range(n_traj):
        for t in range(traj_len):
            arr[i, t, 0] = data[i][t].get("S", np.nan)
            arr[i, t, 1] = data[i][t].get("I", np.nan)
            arr[i, t, 2] = data[i][t].get("R", np.nan)
    return np.squeeze(arr)

def simulate(n=100, T=60, env_name="", variation="train"):
    sim = SimulatorStep()
    trajectories = []
    for _ in range(n):
        state = {"S": np.random.randint(50, 101), "I": np.random.randint(1, 6), "R": 0}
        states = [state]
        for _ in range(T):
            states.append(sim.step(state=states[-1], action=0))
        trajectories.append(states)
    return (trajectories, None)

def load_data(n=100, config={}, seed=0, env_name=""):
    description = """SIR"""
    test_set = simulate(n, env_name=env_name, variation="test")
    train_set = simulate(n, env_name=env_name, variation="train")
    val_set = simulate(n, env_name=env_name, variation="val")

    test_set_np = (trajectories_to_numpy(test_set[0]), None)
    train_set_np = (trajectories_to_numpy(train_set[0]), None)
    val_set_np = (trajectories_to_numpy(val_set[0]), None)

    return (
        train_set,
        val_set,
        test_set,
        description,
        train_set_np,
        val_set_np,
        test_set_np,
    )


if __name__ == "__main__":
    config = {
        "run": {
            "optimizer": "SBI",
            "es_generations": 10,
            "optimize_params": True,
            "mmd_sigma": 0.2,
            "sbi_num_simulations": 1000,
            "sbi_num_samples_posterior": 1000,
            "sbi_sampling_timeout": 30,
        }
    }
    config = to_dot_dict(config)
    train_set, val_set, test_set, _ = load_data(n=10, config=config, env_name="SIR")

    class SimulatorStepGenerated():
        def __init__(self):
            self.beta = 0.5
            self.gamma = 0.3
        def step(self, state: dict, action: int, rng: np.random.Generator) -> dict:
            S, I, R, N = state['S'], state['I'], state['R'], state['S']+state['I']+state['R']
            if N <= 0: return {'S': 0, 'I': 0, 'R': 0}
            p_inf = np.clip(1.0 - np.exp(-self.beta * I / N), 0.0, 1.0)
            new_inf = rng.binomial(S, p_inf)
            p_rec = np.clip(1.0 - np.exp(-self.gamma), 0.0, 1.0)
            new_rec = rng.binomial(I, p_rec)
            return {'S': max(S - new_inf, 0), 'I': max(I + new_inf - new_rec, 0), 'R': max(R + new_rec, 0)}
        def get_parameters(self) -> np.ndarray: return np.array([self.beta, self.gamma], dtype=float)
        def set_parameters(self, p: np.ndarray): self.beta, self.gamma = p[0], p[1]
        def get_parameters_uniform_prior_min_max(self) -> np.ndarray: return np.array([[0.0, 0.0], [2.0, 2.0]], dtype=float)

    simulator_evaluator = SimulatorStep()
    results = simulator_evaluator.evaluate_simulator_code_wrapper(
        model=SimulatorStepGenerated, train_data=train_set, val_data=val_set, test_data=test_set, config=config, env_name="SIR"
    )
    train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss = results
    
    print("\n--- FINAL RESULTS ---")
    print(f"Train Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    print(f"Test Loss: {test_loss}")
    print(f"Optimized Parameters: {optimized_parameters}")
    print(f"Validation Loss per Dimension: {loss_per_dim_dict}")