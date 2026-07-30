import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment — canonical stochastic SIR with Binomial draws.

		Iteration history and feedback corrections:
		  Prior run (K=0.10): extreme NegBin overdispersion Var=μ+10μ²; NLE≈-102; false 'global optimum'.
		  Prior run (K=2.0):  NegBin still too broad at epidemic scales; NLE≈-251.
		  Feedback: use Binomial draws (K→∞ limit = canonical stochastic SIR); Var=np(1-p).
		  Feedback: tighten parameter ranges to COVID-specific values.

		Current configuration:
		  Binomial(S, p_inf): canonical stochastic SIR infection; Var=S×p×(1-p); tight predictive distribution.
		  Binomial(I, p_rec): canonical stochastic SIR recovery; pool=I (standard; not I+new_inf).
		  R0  ∈ (1.5, 6.0):  exp(log(1.5) + log(4.0)×sigmoid(p0)); COVID-plausible range.
		  γ   ∈ (0.05, 0.20): exp(log(0.05) + log(4.0)×sigmoid(p1)); 5–20 day recovery.
		  ALPHA=1.0: standard mass-action; density-dependent I/N.
		"""
		super(SimulatorStep, self).__init__()
		return

	def get_parameters(self) -> np.ndarray:
		"""
		Returns the model parameters as an array.
		"""
		return self.parameters  # shape: (2,) numpy array set by set_parameters()

	def set_parameters(self, parameters: np.ndarray):
		"""
		Updates the model parameters.

		Args:
			parameters (np.ndarray): Array of parameters to update.
		"""
		assert len(parameters) == 2, "Parameter array must have length 2."
		self.parameters = parameters  # shape: (2,)

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
		One stochastic discrete-time SIR step — canonical Binomial formulation.

		Feedback corrections applied:
		  (1) Binomial draws: new_inf~Binomial(S,p_inf), new_rec~Binomial(I,p_rec).
		      NegBin(K=2.0) produced NLE≈-251 due to excess variance at epidemic scales.
		      Binomial is the standard stochastic SIR (Gillespie tau-leaping); Var=np(1-p).
		  (2) Tightened COVID ranges: R0∈(1.5,6.0), gamma∈(0.05,0.20).
		      Prior broad ranges wasted SBI prior mass on non-COVID regimes.
		  (3) pool=I standard SIR (not I+new_inf; that violated one-step latency).

		Model equations (dt=1 day):
		  R0    = exp(log(1.5) + log(4.0) × sigmoid(param[0]))    ∈ (1.500, 6.000)
		  gamma = exp(log(0.05) + log(4.0) × sigmoid(param[1]))   ∈ (0.050, 0.200)
		  beta  = R0 × gamma
		  frac  = I / N
		  p_inf = 1 - exp(-beta × frac)                            [standard mass-action ALPHA=1.0]
		  p_rec = 1 - exp(-gamma)
		  new_inf ~ Binomial(n=S, p=p_inf),  clip [0, S]           [canonical stochastic SIR]
		  new_rec ~ Binomial(n=I, p=p_rec),  clip [0, I]           [pool=I; standard SIR]
		  S_new = S - new_inf
		  I_new = I + new_inf - new_rec
		  R_new = N - S_new - I_new                                 [exact N conservation]

		Args:
			state (dict): "S": int, "I": int, "R": int
			parameters (np.ndarray): shape (2,) unconstrained SBI parameters ∈ ℝ.
			action (int | None): unused.
			rng (np.random.Generator): seeded numpy RNG.

		Returns:
			next_state (dict): "S": int, "I": int, "R": int
		"""
		# ── Hyperparameters ──────────────────────────────────────────────────────────────────────
		# COVID-specific ranges per feedback; Binomial draws per feedback (NegBin→extreme overdispersion)
		LOG_R0_MIN   =  float(np.log(1.5))    # scalar: log(1.5)≈0.405; R0_min=1.5; COVID lower bound; feedback recommended
		LOG_R0_RANGE =  float(np.log(4.0))    # scalar: log(4.0)≈1.386; R0_max=1.5×4=6.0; COVID upper bound; feedback recommended
		LOG_G_MIN    =  float(np.log(0.05))   # scalar: log(0.05)≈-2.996; gamma_min=0.05; 20-day recovery; feedback recommended
		LOG_G_RANGE  =  float(np.log(4.0))    # scalar: log(4.0)≈1.386; gamma_max=0.05×4=0.20; 5-day recovery; feedback recommended

		# --- State extraction (pure SIR; SEIR closed) ---
		S = int(state["S"])   # scalar int ≥ 0: susceptible count at step start
		I = int(state["I"])   # scalar int ≥ 0: infectious count at step start
		R = int(state["R"])   # scalar int ≥ 0: recovered count at step start
		N = S + I + R         # scalar int ≥ 0: conserved total population; N=S+I+R throughout

		# --- Absorbing states ---
		if N <= 0:
			return {"S": 0, "I": 0, "R": 0}          # empty population; exact N=0 conservation
		if I <= 0:
			return {"S": S, "I": 0, "R": N - S}      # epidemic extinct; I=0 absorbing state; exact N conservation

		# --- Parameter transforms: ℝ → bounded positive via sigmoid + exp ---
		p0 = float(parameters[0])                                                    # scalar ∈ ℝ: raw SBI unconstrained param 0 (R0 axis)
		p1 = float(parameters[1])                                                    # scalar ∈ ℝ: raw SBI unconstrained param 1 (gamma axis)
		s0 = 1.0 / (1.0 + np.exp(-np.clip(p0, -30.0, 30.0)))                        # scalar ∈ (0,1): sigmoid(p0); clip ±30 prevents float64 overflow in exp
		s1 = 1.0 / (1.0 + np.exp(-np.clip(p1, -30.0, 30.0)))                        # scalar ∈ (0,1): sigmoid(p1); clip ±30 prevents float64 overflow in exp
		R0    = float(np.exp(LOG_R0_MIN   + LOG_R0_RANGE * s0))                      # scalar ∈ (1.5, 6.0): basic reproduction number; COVID-specific
		gamma = float(np.exp(LOG_G_MIN    + LOG_G_RANGE  * s1))                      # scalar ∈ (0.05, 0.20): per-day recovery rate; 5–20 day infectious period
		beta  = R0 * gamma                                                            # scalar > 0: per-day transmission rate; standard β=R0×γ reparameterisation

		# --- Transition probabilities ---
		# Standard mass-action: force_inf = beta × (I/N); ALPHA=1.0; density-dependent.
		# Complementary exponential CDF → exact discrete-time conversion from continuous rates.
		frac      = float(I) / float(N)                                               # scalar ∈ (0,1]: infectious fraction I/N; I>0 and N>0 guaranteed by absorbing state guards
		force_inf = beta * frac                                                        # scalar ≥ 0: β×(I/N); standard SIR force of infection (ALPHA=1.0)
		p_inf     = float(np.clip(1.0 - np.exp(-force_inf), 0.0, 1.0))               # scalar ∈ [0,1]: per-susceptible infection probability; complementary exponential CDF
		p_rec     = float(np.clip(1.0 - np.exp(-gamma),     0.0, 1.0))               # scalar ∈ [0,1]: per-infectious recovery probability; γ∈(0.05,0.20)→p_rec∈(0.049,0.181)

		# --- S → I: Binomial infection draw (canonical stochastic SIR) ---
		# Binomial(n=S, p=p_inf): Var=S×p×(1-p); tight predictive distribution at epidemic scales.
		# Feedback: NegBin(K=0.10)→Var=μ+10μ²→NLE≈-102; NegBin(K=2.0)→NLE≈-251; Binomial is correct.
		# Standard Gillespie/tau-leaping discrete SIR formulation.
		if S <= 0:
			new_inf = 0                                                                # scalar int: no susceptibles → 0 new infections
		else:
			raw_inf = int(rng.binomial(n=S, p=p_inf))                                 # scalar int ∈ [0,S]: Binomial draw; n=S susceptibles; p=daily infection prob; canonical stochastic SIR
			new_inf = int(min(max(raw_inf, 0), S))                                    # scalar int ∈ [0,S]: defensive clip; Binomial already bounded but guards against edge cases

		# --- I → R: Binomial recovery draw (canonical stochastic SIR) ---
		# Binomial(n=I, p=p_rec): Var=I×p×(1-p); standard tau-leaping SIR recovery.
		# pool=I per feedback: standard SIR (was I+new_inf → violated one-step latency invariant).
		# Feedback: I-only pool confirmed correct; pool=I+new_inf was a prior error.
		if I <= 0:
			new_rec = 0                                                                # scalar int: no infectious → 0 recoveries
		else:
			raw_rec = int(rng.binomial(n=I, p=p_rec))                                 # scalar int ∈ [0,I]: Binomial draw; n=I infectious; p=daily recovery prob; canonical stochastic SIR
			new_rec = int(min(max(raw_rec, 0), I))                                    # scalar int ∈ [0,I]: defensive clip; Binomial already bounded but guards against edge cases

		# --- Compartment updates (exact N conservation via residual R assignment) ---
		S_new = int(max(S - new_inf,           0))    # scalar int ≥ 0: S depleted by new infections; new_inf≤S by Binomial construction
		I_new = int(max(I + new_inf - new_rec, 0))    # scalar int ≥ 0: net I flow; max() defensive guard against floating-point edge cases
		R_new = int(max(N - S_new - I_new,     0))    # scalar int ≥ 0: residual assignment enforces exact S_new+I_new+R_new=N conservation

		next_state = {
			"S": S_new,   # scalar int: updated susceptible count; SIR only
			"I": I_new,   # scalar int: updated infectious count
			"R": R_new,   # scalar int: updated recovered count; residual for exact N conservation
		}

		return next_state