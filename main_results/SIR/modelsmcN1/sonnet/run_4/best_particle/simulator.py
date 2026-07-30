import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		return

	def get_parameters(self) -> np.ndarray:
		"""
		Returns the model parameters as an array.
		"""
		return self.parameters

	def set_parameters(self, parameters: np.ndarray):
		"""
		Updates the model parameters.

		Args:
			parameters (np.ndarray): Array of parameters to update.
		"""
		assert len(parameters) == 2, "Parameter array must have length 2."
		self.parameters = parameters

	def step(self, state: dict, action: int | None, rng: np.random.Generator) -> dict:
		"""
		Wrapper to call the forward method.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""
		return self.forward(state=state, parameters=self.get_parameters(), action=action, rng=rng)

	def forward(
		self,
		state: dict,
		parameters: np.ndarray,
		action: int | None,
		rng: np.random.Generator,
	) -> dict:
		"""
		Implements one stochastic simulation step of the discrete-time SIR model.

		ITER-103: Exponential p_inf formula (Iter-84, NLE -56.7) + COVID-realistic parameter ranges.

		  ROOT CAUSE OF ITER-102 FAILURE (NLE -565):
		    The beta range (0.020, 0.770) with gamma range (0.020, 0.200) allows R0 = beta/gamma
		    up to 0.770/0.020 = 38.5. This is orders of magnitude above realistic COVID R0 (2-5),
		    causing the entire susceptible population to be infected within days, placing near-zero
		    likelihood on 60-day observed trajectories with realistic epidemic curves.

		  FIX: COVID-REALISTIC PARAMETER RANGES (per Iter-102 feedback, issue 1, CRITICAL):
		    beta  = 0.05 + 0.45 * sig0  -> range (0.05, 0.50)
		      - Corresponds to daily transmission rates observed for COVID-19
		      - R0 = beta / gamma: max R0 = 0.50 / 0.05 = 10 (COVID range: 2-5, so this covers it)
		    gamma = 0.05 + 0.10 * sig1  -> range (0.05, 0.15)
		      - Recovery period = 1/gamma: range 6.7 to 20 days (COVID: ~7-14 days, correct)
		    R0 = beta/gamma range: (0.05/0.15) to (0.50/0.05) = 0.33 to 10 (realistic)

		  EXPONENTIAL FORMULA (validated at Iter-84, NLE -56.7):
		    p_inf = 1.0 - np.exp(-beta * float(I) / float(N))
		    - Reed-Frost / exact stochastic SIR discrete-time formula
		    - Derived from Poisson process: P(escape) = exp(-force_of_infection * dt), dt=1 day
		    - p_inf in [0, 1) for all finite beta > 0, I/N in [0, 1]: no clipping needed
		    - When I=0: p_inf = 0 exactly (no spurious infections)
		    - Saturates correctly without Euler upward bias

		  ITER-103 STRUCTURE (pure 2-draw, exponential p_inf, COVID ranges):
		    Step 1: Unpack S, I, R; compute N = S + I + R
		    Step 2: if N == 0: return early (no draws)
		    Step 3: compute beta, gamma from COVID-realistic sigmoid transforms
		    Step 4: p_inf = 1.0 - np.exp(-beta * float(I) / float(N))
		    Step 5: new_infections  = rng.binomial(int(S), p_inf)      [DRAW 1]
		    Step 6: p_rec = gamma
		    Step 7: new_recoveries = rng.binomial(int(I), p_rec)       [DRAW 2]
		    Step 8: update S, I, R compartments
		    Total: 2 RNG draws (pure Binomial, NO Normal anywhere)

		CONDENSED EMPIRICAL RECORD:
		  Iter 40:  output guards                               -> NLE -359  CATASTROPHE
		  Iter 51:  eps_rec sigma=0.01                          -> NLE -164  REGRESSION
		  Iter 54:  narrow beta range (Euler formula)           -> NLE -536  CATASTROPHE
		  Iter 70:  additive eps_inf before guard, sigma=0.020  -> NLE -40.6
		  Iter 73:  int() on output compartments                -> NLE -144  REGRESSION
		  Iter 77:  narrow beta/gamma (Euler formula)           -> NLE -553  CATASTROPHE
		  Iter 84:  1-exp(-beta*I/N), 2-draw, wide ranges       -> NLE -56.7 BEST RELIABLE
		  Iter 87:  over-wide beta(0.01,1.0) (Euler)            -> NLE 3.82e+26 CATASTROPHE
		  Iter 88:  narrow beta(0.1,0.4) (Euler)                -> NLE -558  CATASTROPHE
		  Iter 91:  beta*I/N, 2-draw (claimed -79.2)            -> unreproducible (3 attempts)
		  Iter 95:  eps sigma=0.019                             -> NLE -212  CATASTROPHE
		  Iter 97:  narrow COVID ranges (Euler)                 -> NLE -567  CATASTROPHE
		  Iter 98:  low gamma floor 0.010 (Euler)               -> NLE -557  CATASTROPHE
		  Iter 99:  beta*I/N, 2-draw, with clip                 -> NLE -549  CATASTROPHE
		  Iter 101: beta*I/N, 2-draw, no clip                   -> NLE -550  CATASTROPHE
		  Iter 102: 1-exp(-beta*I/N), wide ranges (0.02-0.77)   -> NLE -565  CATASTROPHE
		  Iter 103: 1-exp(-beta*I/N), COVID ranges (0.05-0.50)  -> target NLE ~-56.7

		  NOTE on prior narrowing catastrophes (Iter-54, -77, -88, -97, -98):
		    All used the Euler formula beta*I/N. With exponential formula, narrowing to
		    COVID-realistic ranges is physically motivated and distinct from those failures.
		    The key distinction: COVID ranges (0.05-0.50 beta, 0.05-0.15 gamma) are
		    physiologically correct, not arbitrary truncations.

		Args:
			state (dict): The environment state: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing [theta_0, theta_1].
			action (int | None): None (unused).
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state: "S": int, "I": int, "R": int.
		"""
		# --- Unpack state ---
		S = state["S"]  # scalar int/numpy.int64: susceptible count, >= 0
		I = state["I"]  # scalar int/numpy.int64: infected count, >= 0
		R = state["R"]  # scalar int/numpy.int64: recovered count, >= 0

		# --- Total population (conserved exactly by binomial sampling) ---
		N = S + I + R  # scalar: total population, >= 0

		# --- N==0 guard: early return BEFORE all draws ---
		# Pure 2-draw structure: no unconditional pre-guard draws needed.
		if N == 0:
			return {"S": S, "I": I, "R": R}

		# --- Unpack parameters ---
		theta0 = parameters[0]  # scalar float64: unconstrained real mapped to beta
		theta1 = parameters[1]  # scalar float64: unconstrained real mapped to gamma

		# --- Sigmoid transforms ---
		sig0 = 1.0 / (1.0 + np.exp(-theta0))  # scalar float64: sigmoid(theta0), in (0, 1)
		sig1 = 1.0 / (1.0 + np.exp(-theta1))  # scalar float64: sigmoid(theta1), in (0, 1)

		# --- Beta: COVID-realistic transmission rate ---
		# Range (0.05, 0.50). With gamma in (0.05, 0.15):
		#   R0 = beta/gamma ranges from 0.05/0.15 ≈ 0.33 to 0.50/0.05 = 10.
		# COVID-19 published R0: 2–5 (original strain), up to 8–10 (Omicron).
		# Prior wide range (0.020-0.770) gave R0 up to 38.5 -> catastrophic mismatch
		# with 60-day observed trajectories (entire population infected in days).
		beta = 0.05 + 0.45 * sig0  # scalar float64: daily transmission rate, in (0.05, 0.50)

		# --- Gamma: COVID-realistic recovery rate ---
		# Range (0.05, 0.15). Recovery period = 1/gamma: range ~6.7 to 20 days.
		# COVID-19 infectious period: ~7-14 days -> gamma ~ 0.07-0.14, captured here.
		# Prior range (0.020-0.200) had min gamma=0.020 enabling R0=38.5 catastrophe.
		gamma = 0.05 + 0.10 * sig1  # scalar float64: daily recovery rate, in (0.05, 0.15)

		# --- Force of infection: EXPONENTIAL DISCRETE-TIME FORMULA (Reed-Frost / Poisson) ---
		# p_inf = 1.0 - exp(-beta * I / N)
		# Derived from: in continuous time, each susceptible escapes infection at rate beta*I/N.
		# Over one day (dt=1): P(escape) = exp(-beta*I/N), P(infected) = 1 - exp(-beta*I/N).
		# Properties:
		#   - p_inf in [0, 1) for all finite beta > 0: valid Binomial probability, no clip needed
		#   - When I=0: p_inf = 0 exactly (no spurious infections at disease-free equilibrium)
		#   - When I=N: p_inf = 1-exp(-beta) in (1-exp(-0.50), 1-exp(-0.05)) ≈ (0.394, 0.049)
		#   - Avoids Euler upward bias: Taylor expansion gives beta*I/N + O((beta*I/N)^2)
		#     At beta=0.5, I/N=0.5: Euler gives 0.25, exact gives 1-exp(-0.25)=0.221 (12% less)
		# Empirically validated: Iter-84 NLE -56.7 with this formula.
		# Euler formula beta*I/N: 3 consecutive attempts all NLE ~-549/-550 CATASTROPHIC.
		p_inf = 1.0 - np.exp(-beta * float(I) / float(N))  # scalar float64: infection probability, in [0.0, ~0.394)

		# --- DRAW 1: new infections, Binomial(S, p_inf) ---
		# int(S): Python built-in int (ensures compatibility with rng.binomial).
		# p_inf in [0, 1) by mathematical construction, always valid.
		new_infections = rng.binomial(n=int(S), p=p_inf)  # scalar numpy.int64: S->I flow, in [0, S]

		# --- Recovery probability: deterministic gamma, NO eps_rec ---
		# eps_rec (sigma=0.01) caused NLE -164 at Iter-51. BANNED permanently.
		# gamma in (0.05, 0.15) always a valid Binomial probability.
		p_rec = gamma  # scalar float64: daily recovery probability, in (0.05, 0.15)

		# --- DRAW 2: new recoveries, Binomial(I, p_rec) ---
		# int(I): Python built-in int.
		new_recoveries = rng.binomial(n=int(I), p=p_rec)  # scalar numpy.int64: I->R flow, in [0, I]

		# --- Compartment update ---
		# Non-negativity guaranteed by binomial construction:
		#   new_infections <= S  =>  S_new = S - new_infections >= 0
		#   new_recoveries <= I  =>  I_new = I + new_infections - new_recoveries >= 0
		# Population conservation: S_new + I_new + R_new = N exactly.
		# NO int() on output (int() on outputs caused NLE -144 at Iter-73).
		# NO output guards (output guards caused NLE -359 at Iter-40).
		S_new = S - new_infections                   # native numpy scalar: susceptible, >= 0
		I_new = I + new_infections - new_recoveries  # native numpy scalar: infected, >= 0
		R_new = R + new_recoveries                   # native numpy scalar: recovered, >= 0

		return {"S": S_new, "I": I_new, "R": R_new}