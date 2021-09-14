"""
Notes to self:

are we missing q anywhere (where we have eta)?

write a description at the top of this file

at the moment only implements the fast method of optimizing
should also check the slow method...
"""


def cond_ent(joint, marg):
	"""
	Returns H(A|B) = H(AB) - H(B)

	Inputs:
		joint 	-- 	joint distribution on AB
		marg 	-- 	marginal distribution on B
	"""

	hab, hb = 0.0, 0.0

	for prob in joint:
		if 0.0 < prob < 1.0:
			hab += -prob*log2(prob)

	for prob in marg:
		if 0.0 < prob < 1.0:
			hb += -prob*log2(prob)

	return hab - hb

def objective(ti, q):
	"""
	Returns the objective function for the faster computations.
		Key generation on X=0
		Only two outcomes for Alice

		ti 	-- 	i-th node
		q  	--	bit flip probability
	"""
	obj = 0.0
	F = [A[0][0], 1 - A[0][0]]				# POVM for Alices key gen measurement
	for a in range(A_config[0]):
		b = (a + 1) % 2 					# (a + 1 mod 2)
		M = (1-q) * F[a] + q * F[b] 		# Noisy preprocessing povm element
		obj += M * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

	return obj


def compute_entropy(SDP, q):
	"""
	Computes lower bound on H(A|X=0,E) using the fast (but less accurate) method

		SDP -- sdp relaxation object
		q 	-- probability of bitflip
	"""
	ck = 0.0		# kth coefficient
	ent = 0.0		# lower bound on H(A|X=0,E)

	# We can also decide whether to perform the final optimization in the sequence
	# or bound it trivially. Best to keep it unless running into numerical problems
	# with it. There's potentially also a nontrivial way to bound it (need to investigate).
	if KEEP_M:
		num_opt = len(T)
	else:
		num_opt = len(T) - 1
		ent = 2 * q * (1-q) * W[-1] / log(2)

	for k in range(num_opt):
		ck = W[k]/(T[k] * log(2))

		# Get the k-th objective function
		new_objective = objective(T[k], q)

		SDP.set_objective(new_objective)
		SDP.solve('mosek', solverparameters = {'num_threads': int(NUM_SUBWORKERS)})

		if SDP.status == 'optimal':
			# 1 contributes to the constant term
			ent += ck * (1 + SDP.dual)
		else:
			# If we didn't solve the SDP well enough then just bound the entropy
			# trivially
			ent = 0
			if VERBOSE:
				print('Bad solve: ', k, SDP.status)
			break

	return ent

def HAgB(sys, eta, q):
	"""
	Computes the error correction term in the key rate for a given system,
	a fixed detection efficiency and noisy preprocessing. Computes the relevant
	components of the distribution and then evaluates the conditional entropy.

		sys	--	parameters of system
		eta --	detection efficiency
		q 	--	bitflip probability
	"""

	# Computes H(A|B) required for rate
	[id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
	[theta, a0, a1, b0, b1, b2] = sys[:]
	rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

	# Noiseless measurements
	a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
	b20 = 0.5*(id + cos(b2)*sz + sin(b2)*sx)

	# Alice bins to 0 transforms povm
	A00 = eta * a00 + (1-eta) * id
	# Final povm transformation from the bitflip
	A00 = (1-q) * A00 + q * (id - A00)
	A01 = id - A00

	# Bob has inefficient measurement but doesn't bin
	B20 = eta * b20
	B21 = eta * (id - b20)
	B22 = (1-eta) * id

	# joint distribution
	q00 = (rho*qtp.tensor(A00, B20)).tr().real
	q01 = (rho*qtp.tensor(A00, B21)).tr().real
	q02 = (rho*qtp.tensor(A00, B22)).tr().real
	q10 = (rho*qtp.tensor(A01, B20)).tr().real
	q11 = (rho*qtp.tensor(A01, B21)).tr().real
	q12 = (rho*qtp.tensor(A01, B22)).tr().real

	qb0 = (rho*qtp.tensor(id, B20)).tr().real
	qb1 = (rho*qtp.tensor(id, B21)).tr().real
	qb2 = (rho*qtp.tensor(id, B22)).tr().real

	qjoint = [q00,q01,q02,q10,q11,q12]
	qmarg = [qb0,qb1,qb2]

	return cond_ent(qjoint, qmarg)

def compute_rate(SDP, sys, eta, q):
	"""
	Computes a lower bound on the rate H(A|X=0,E) - H(A|X=0,Y=2,B) using the fast
	method

		SDP		--	sdp relaxation object
		sys 	-- 	parameters of the system
		eta 	-- 	detection efficiency
		q 		-- 	bitflip probability
	"""
	score_cons = score_constraints(sys[:], eta)
	SDP.process_constraints(equalities = op_eqs,
						inequalities = op_ineqs,
						momentequalities = moment_eqs[:] + score_cons[:],
						momentinequalities = moment_ineqs)
	ent = compute_entropy(SDP, q)
	err = HAgB(sys, eta, q)
	return ent - err

def compute_dual_vector(SDP, q):
	"""
	Extracts the vector from the dual problem(s) that builds into the affine function
	of the constraints that lower bounds H(A|X=0,E)

		SDP	--	sdp relaxation object
		q 	-- 	probability of bitflip
	"""

	dual_vec = np.zeros(8)	# dual vector
	ck = 0.0				# kth coefficient
	ent = 0.0				# lower bound on H(A|X=0,E)

	if KEEP_M:
		num_opt = len(T)
	else:
		num_opt = len(T) - 1
		ent = 2 * q * (1-q) * W[-1] / log(2)

	# Compute entropy and build dual vector from each sdp solved
	for k in range(num_opt):
		ck = W[k]/(T[k] * log(2))

		# Get the k-th objective function
		new_objective = objective(T[k], q)

		# Set the objective and solve
		SDP.set_objective(new_objective)
		SDP.solve('mosek',solverparameters = {'num_threads': int(NUM_SUBWORKERS)})

		# Check solution status
		if SDP.status == 'optimal':
			ent += ck * (1 + SDP.dual)
			# Extract the dual vector from the solved sdp
			d = sdp_dual_vec(SDP)
			# Add the dual vector to the total dual vector
			dual_vec = dual_vec + ck * d
		else:
			ent = 0
			dual_vec = np.zeros(8)
			break

	return dual_vec, ent


def score_constraints(sys, eta=1.0):
	"""
	Returns the moment equality constraints for the distribution specified by the
	system sys and the detection efficiency eta. We only look at constraints coming
	from the inputs 0/1. Potential to improve by adding input 2 also?

		sys	--	system parameters
		eta	-- 	detection efficiency
	"""

	# Extract the system
	[id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
	[theta, a0, a1, b0, b1, b2] = sys[:]
	rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

	# Define the first projectors for each of the measurements of Alice and Bob
	a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
	a01 = id - a00
	a10 = 0.5*(id + cos(a1)*sz + sin(a1)*sx)
	a11 = id - a10
	b00 = 0.5*(id + cos(b0)*sz + sin(b0)*sx)
	b01 = id - b00
	b10 = 0.5*(id + cos(b1)*sz + sin(b1)*sx)
	b11 = id - b10

	A_meas = [[a00, a01], [a10, a11]]
	B_meas = [[b00, b01], [b10, b11]]

	constraints = []

	# Add constraints for p(00|xy)
	for x in range(2):
		for y in range(2):
			constraints += [A[x][0]*B[y][0] - (eta**2 * (rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + \
						+ eta*(1-eta)*((rho*qtp.tensor(A_meas[x][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[y][0])).tr().real) + \
						+ (1-eta)*(1-eta))]

	# Now add marginal constraints p(0|x) and p(0|y)
	constraints += [A[0][0] - eta * (rho*qtp.tensor(A_meas[0][0], id)).tr().real - (1-eta)]
	constraints += [B[0][0] - eta * (rho*qtp.tensor(id, B_meas[0][0])).tr().real - (1-eta)]
	constraints += [A[1][0] - eta * (rho*qtp.tensor(A_meas[1][0], id)).tr().real - (1-eta)]
	constraints += [B[1][0] - eta * (rho*qtp.tensor(id, B_meas[1][0])).tr().real - (1-eta)]

	return constraints[:]

def sys2vec(sys, eta = 1.0):
	"""
	Returns a vector of probabilities determined from the system in the same order as specified
	in the function score_constraints()

		sys	--	system parameters
		eta	-- 	detection efficiency
	"""
	# Get the system from the parameters
	[id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
	[theta, a0, a1, b0, b1, b2] = sys[:]
	rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

	# Define the first projectors for each of the measurements of Alice and Bob
	a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
	a01 = id - a00
	a10 = 0.5*(id + cos(a1)*sz + sin(a1)*sx)
	a11 = id - a10
	b00 = 0.5*(id + cos(b0)*sz + sin(b0)*sx)
	b01 = id - b00
	b10 = 0.5*(id + cos(b1)*sz + sin(b1)*sx)
	b11 = id - b10

	A_meas = [[a00, a01], [a10, a11]]
	B_meas = [[b00, b01], [b10, b11]]

	vec = []

	# Get p(00|xy)
	for x in range(2):
		for y in range(2):
			vec += [(eta**2 * (rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + \
						+ eta*(1-eta)*((rho*qtp.tensor(A_meas[x][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[y][0])).tr().real) + \
						+ (1-eta)*(1-eta))]

	# And now the marginals
	vec += [eta * (rho*qtp.tensor(A_meas[0][0], id)).tr().real + (1-eta)]
	vec += [eta * (rho*qtp.tensor(id, B_meas[0][0])).tr().real + (1-eta)]
	vec += [eta * (rho*qtp.tensor(A_meas[1][0], id)).tr().real + (1-eta)]
	vec += [eta * (rho*qtp.tensor(id, B_meas[1][0])).tr().real + (1-eta)]

	return vec

def sdp_dual_vec(SDP):
	"""
	Extracts the dual vector from the solved sdp by ncpol2sdpa

		SDP -- sdp relaxation object

	Would need to be modified if the number of moment constraints or their
	nature (equalities vs inequalities) changes.
	"""
	raw_vec = SDP.y_mat[-16:]
	vec = [0 for _ in range(8)]
	for k in range(8):
		vec[k] = raw_vec[2*k][0][0] - raw_vec[2*k + 1][0][0]
	return np.array(vec[:])

def optimise_sys(SDP, sys, eta, q):
	"""
	Optimizes the rate using the iterative method via the dual vectors.

		SDP	--	sdp relaxation object
		sys	--	parameters of system that are optimized
		eta -- 	detection efficiency
		q 	-- 	bitflip probability
	"""

	NEEDS_IMPROVING = True			# Flag to check if needs optimizing still
	FIRST_PASS = True				# Checks if first time through loop
	improved_sys = sys[:]			# Improved choice of system
	best_sys = sys[:]				# Best system found
	dual_vec = np.zeros(8)			# Dual vector same length as num constraints

	# Loop until we converge on something
	while(NEEDS_IMPROVING):
		# On the first loop we just solve and extract the dual vector
		if not FIRST_PASS:
			# Here we optimize the dual vector
			# The distribution associated with the improved system
			pstar = sys2vec(improved_sys[:], eta)

			# function to optimize parameters over
			def f0(x):
				#x is sys that we are optimizing
				p = sys2vec(x, eta)
				return -np.dot(p, dual_vec) + HAgB(x, eta, q)

			# Bounds on the parameters of sys
			bounds = [[0, pi/2]] + [[-pi,pi] for _ in range(len(sys) - 1)]
			# Optmize qubit system (maximizing due to negation in f0)
			res = minimize(f0, improved_sys[:], bounds = bounds)
			improved_sys = res.x.tolist()[:]	# Extract optimizer

		# Apply the new system to the sdp
		score_cons = score_constraints(improved_sys[:], eta)
		SDP.process_constraints(equalities = op_eqs,
							inequalities = op_ineqs,
							momentequalities = moment_eqs[:] + score_cons[:],
							momentinequalities = moment_ineqs)

		# Compute new dual vector and the rate
		dual_vec, new_ent = compute_dual_vector(SDP, q)
		new_rate = new_ent - HAgB(improved_sys[:], eta, q)

		if not FIRST_PASS:
			if new_rate < best_rate + best_rate*EPS_M or new_rate < best_rate + EPS_A :
				NEEDS_IMPROVING = False
		else:
			# If first run through then this is the initial entropy
			starting_rate = new_rate
			best_rate = new_rate
			FIRST_PASS = False

		if new_rate > best_rate:
			if VERBOSE > 0:
				print('Optimizing sys (eta, q) =', (eta,q), ' ... ', starting_rate, '->', new_rate)
			best_rate = new_rate
			best_sys = improved_sys[:]


	return best_rate, best_sys[:]

def optimise_q(SDP, sys, eta, q):
	"""
	Optimizes the choice of q.

		SDP	--	sdp relaxation object
		sys	--	parameters of system that are optimized
		eta -- 	detection efficiency
		q 	-- 	bitflip probability

	This function can probably be improved to make the search a bit more efficient and fine grained.
	"""
	q_eps = 0.005	# Can be tuned
	q_eps_min = 0.001

	opt_q = q
	rate = compute_rate(SDP, sys, eta, q) # Computes rate for given q
	starting_rate = rate

	# We check if we improve going left
	if q - q_eps < 0:
		LEFT = 0
	else:
		new_rate = compute_rate(SDP, sys, eta, opt_q - q_eps)
		if new_rate > rate:
			opt_q = opt_q - q_eps
			rate = new_rate
			if VERBOSE > 0:
				print('Optimizing q (eta,q) =', (eta, opt_q), ' ... ', starting_rate, '->', rate)
			LEFT = 1
		else:
			LEFT = 0


	def next_q(q0, step_size):
		q1 = q0 + ((-1)**LEFT) * step_size
		if q1 >= 0 and q1 <= 0.5:
			return q1
		elif step_size/2 >= q_eps_min:
			return next_q(q0, step_size/2)
		else:
			return -1


	STILL_OPTIMIZING = 1

	while STILL_OPTIMIZING:
		# define the next q
		new_q = next_q(opt_q, q_eps)
		if new_q < 0:
			break

		#compute the rate
		new_rate = compute_rate(SDP, sys, eta, new_q)

		if new_rate > rate:
			opt_q = new_q
			rate = new_rate
			if VERBOSE > 0:
				print('Optimizing q (eta,q) =', (eta, opt_q), ' ... ', starting_rate, '->', rate)
		else:
			# If we didn't improve try shortening the distance
			q_eps = q_eps / 2
			if q_eps < q_eps_min:
				STILL_OPTIMIZING = 0

	return rate, opt_q

def optimise_rate(SDP, sys, eta, q):
	"""
	Iterates between optimizing sys and optimizing q in order to optimize overall rate.
	"""

	STILL_OPTIMIZING = 1

	best_rate = compute_rate(SDP, sys, eta, q)
	best_sys = sys[:]
	best_q = q

	while STILL_OPTIMIZING:
		new_rate, new_sys = optimise_sys(SDP, best_sys[:], eta, best_q)
		new_rate, new_q = optimise_q(SDP, new_sys[:], eta, best_q)


		if (new_rate < best_rate + best_rate*EPS_M) or (new_rate < best_rate + EPS_A):
			STILL_OPTIMIZING = 0

		if new_rate > best_rate:
			best_rate = new_rate
			best_sys = new_sys[:]
			best_q = new_q

	return best_rate, best_sys, best_q

def generate_quadrature(m):
	"""
	Generates the Gaussian quadrature nodes t and weights w. Due to the way the
	package works it generates 2*M nodes and weights. Maybe consider finding a
	better package if want to compute for odd values of M.

	 	m	--	number of nodes in quadrature / 2
	"""
	t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
	t = t[0]
	return t, w

def get_subs():
	"""
	Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
	commutation relations.
	"""
	subs = {}
	# Get Alice and Bob's projective measurement constraints
	subs.update(ncp.projective_measurement_constraints(A,B))

	# Finally we note that Alice and Bob's operators should All commute with Eve's ops
	for a in ncp.flatten([A,B]):
		for z in Z:
			subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})

	return subs

def get_extra_monomials():
	"""
	Returns additional monomials to add to sdp relaxation.
	"""

	monos = []

	# Add ABZ
	ZZ = Z + [Dagger(z) for z in Z]
	Aflat = ncp.flatten(A)
	Bflat = ncp.flatten(B)
	for a in Aflat:
		for b in Bflat:
			for z in ZZ:
				monos += [a*b*z]

	# Add monos appearing in objective function
	for z in Z:
		monos += [A[0][0]*Dagger(z)*z]

	return monos[:]

"""
Now we start with setting up the ncpol2sdpa computations
"""
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
import qutip as qtp
from scipy.optimize import minimize
from ncpol2sdpa.nc_utils import ncdegree
from ncpol2sdpa.solver_common import get_xmat_value
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
from glob import glob
from joblib import Parallel, delayed, parallel_backend


LEVEL = 2						# NPA relaxation level
M = 6							# Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)	# Nodes, weights of quadrature
KEEP_M = 0						# Optimizing mth objective function?
VERBOSE = 0						# If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-4, 1e-4		# Multiplicative/Additive epsilon in iterative optimization

# Some parameters for optimizing systems for a range of eta
NUM_SAMPLES = 2					# Number of random samples
RAND_PROB = 0.2					# Probability we choose uniformly our random system
THETA_VAR = pi/128				# Variance for choosing random state angle
ANG_VAR = pi/24					# Variance for choosing random measurement angle
NUM_WORKERS = 1					# Number of workers to split parallelization over
NUM_SUBWORKERS = 4				# Number of cores each worker has access to


# number of outputs for each inputs of Alice / Bobs devices
# (Dont need to include 3rd input for Bob here as we only constrain the statistics
# for the other inputs).
A_config = [2,2]
B_config = [2,2]

# Operators in problem
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=0)

substitutions = get_subs()			# substitutions used in ncpol2sdpa
moment_ineqs = []					# moment inequalities
moment_eqs = []						# moment equalities
op_eqs = []							# operator equalities
op_ineqs = []						# operator inequalities
extra_monos = get_extra_monomials()	# extra monomials

# Defining the test sys
test_sys = [pi/4, 0, pi/2, pi/4, -pi/4, 0]
test_eta = 0.99
test_q = 0.01


ops = ncp.flatten([A,B,Z])		# Base monomials involved in problem
obj = objective(1,test_q)	# Placeholder objective function


sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
					equalities = op_eqs[:],
					inequalities = op_ineqs[:],
					momentequalities = moment_eqs[:] + score_constraints(test_sys, test_eta),
					momentinequalities = moment_ineqs[:],
					objective = obj,
					substitutions = substitutions,
					extramonomials = extra_monos)


"""
We now have an sdp relaxation of our problem for the test system introduced above.

We can test it and try to optimize our parameters
"""
# ent = compute_entropy(sdp,test_q)
# err = HAgB(test_sys, test_eta, test_q)
# print(ent, err, ent-err)
# exit()
#
# new_rate, new_sys, new_q = optimise_rate(sdp, test_sys, test_eta, test_q)
# print(new_rate)
# print(new_sys)
# print(new_q)
# exit()


"""
Generating the rate plots for a range of detection efficiencies
"""
eta_range = np.linspace(0.8,0.85,20)[:-1].tolist() + np.linspace(0.85, 0.95, 20)[:-1].tolist() + np.linspace(0.95, 1.0, 20).tolist()

# Define a function that we can distribute to a parallel processing job
def task(ETA):
	print('Starting eta ', ETA)
	fn = './data/qkd_2322_'+ str(2*M) + 'M_' + str(int(100000*ETA)) + '.csv'

	# We want good sys choices to propogate through the optimization for different
	# eta so we load the data for nearby eta and we shall also optimize their best system
	known_systems = []
	for filename in list(glob('./data/qkd_2322_'+ str(2*M) + 'M_*.csv')):
			data = np.loadtxt(filename, delimiter = ',').tolist()
			if len(data) > 0:
				known_systems += [data]
	if len(known_systems) > 0:
		# order w.r.t. eta -- orders in ascending order
		known_systems.sort(key = lambda x: x[0])
		# Grabbing the systems either side of the system currently being optimized.
		try:
			idx = [i for i, x in enumerate(known_systems) if x[0] == ETA][0]
			if idx > 0:
				previous_sys = known_systems[idx-1][:]
			else:
				previous_sys = [None]
			if idx < len(known_systems)-1:
				next_sys = known_systems[idx+1][:]
			else:
				next_sys = [None]
		except:
			previous_sys = [None]
			next_sys = [None]

	# If we have optimized before then open the data and collect best sys and hmin
	try:
		data = np.loadtxt(fn, delimiter=',').tolist()
		BEST_RATE = data[1]
		BEST_Q = data[2]
		BEST_SYS = data[3:]
		NEEDS_PRESOLVE = False
	except:
		# If we've never optimized before then try some ansatz (can be modified)
		BEST_RATE = -10
		BEST_Q = 0.0
		NEEDS_PRESOLVE = True
		BEST_SYS = [0.798381403026085,2.95356587736751,-1.79796447473952, 2.05617041931652,
					0.653912312507124,2.95155725819389]

	# Attempt 0 -- FIRST ATTEMPT AT SUPPLIED DEFAULT SYSTEM
	# if ETA < 0.92:
	# 	# Forcing it to solve a system we already know does well at low eta.
	# 	NEEDS_PRESOLVE = True
	if NEEDS_PRESOLVE:
		try:
			print('Needs presolving, no file found...')
			curr_sys = [0.150415183475178, -0.0269996391717213, 0.96107271555041, -0.565579360974102, 0.115396460799296,-0.00955679420634085]
			BEST_Q = 0.0749375
			opt_rate, opt_sys, opt_q = optimise_rate(sdp, curr_sys[:], ETA, BEST_Q)

			if opt_rate > BEST_RATE:
				print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
				BEST_RATE = opt_rate
				BEST_SYS = opt_sys[:]
				BEST_Q = opt_q
		except:
			pass


	# ATTEMPT 1 -- TRY BEST SYSTEM FROM PREVIOUS POINT
	try:
		print('trying previous point\'s current best system...')
		curr_q = previous_sys[2]
		curr_sys = previous_sys[3:]
		opt_rate, opt_sys, opt_q = optimise_rate(sdp, curr_sys[:], ETA, curr_q)

		if opt_rate > BEST_RATE:
			print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
			BEST_RATE = opt_rate
			BEST_SYS = opt_sys[:]
			BEST_Q = opt_q
	except:
		pass

	# ATTEMPT 2 -- TRY BEST SYSTEM FROM NEXT POINT
	try:
		print('trying next point\'s current best system...')
		curr_q = next_sys[2]
		curr_sys = next_sys[3:]
		opt_rate, opt_sys, opt_q = optimise_rate(sdp, curr_sys[:], ETA, curr_q)

		if opt_rate > BEST_RATE:
			print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
			BEST_RATE = opt_rate
			BEST_SYS = opt_sys[:]
			BEST_Q = opt_q
	except:
		pass

	# ATTEMPT 3 -- TRY RANDOM POINTS
	try:
		print('trying random systems')
		for _ in range(NUM_SAMPLES):
			if np.random.random() < RAND_PROB:
				curr_sys = [np.random.normal(BEST_SYS[0], THETA_VAR)] + np.random.uniform(-pi,pi, len(BEST_SYS) -1).tolist()
			else:
				curr_sys = np.random.normal(BEST_SYS, [THETA_VAR] + [ANG_VAR for _ in range(len(BEST_SYS) -1)])

			opt_rate, opt_sys, opt_q = optimise_rate(sdp, curr_sys[:], ETA, BEST_Q)

			if opt_rate > BEST_RATE:
				print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
				BEST_RATE = opt_rate
				BEST_SYS = opt_sys[:]
				BEST_Q = opt_q
	except:
		pass

	np.savetxt(fn, [ETA, BEST_RATE, BEST_Q] + np.array(BEST_SYS).tolist(), delimiter = ',')
	print('Finished eta ', ETA, ' with rate ', BEST_RATE)
	return 0

# Run the optimization.
with parallel_backend("loky"):
	results = Parallel(n_jobs = NUM_WORKERS, verbose=0)(delayed(task)(eta) for eta in reversed(eta_range))
