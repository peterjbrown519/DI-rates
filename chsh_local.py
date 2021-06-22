"""
Script to compute converging lower bounds on the rates of local randomness
extracted from two devices achieving a minimal expected CHSH score. More specifically,
computes a sequence of lower bounds on the problem

 			inf H(A|X=0,E)

where the infimum is over all quantum devices achieving a CHSH score of w. See
the accompanying paper for more details (this script was used for Figure 1).
"""


def objective(ti):
	# Note we only define the first operator in Alice and Bob's measurements
	# We then denote the second operator by 1-X, where X is the first operator
	# in the measurement.
	return (A[0][0]*(Dagger(Z[0]) + Z[0]) + (1-A[0][0])*(Dagger(Z[1]) + Z[1])) + \
		(1-ti) * (A[0][0]*Dagger(Z[0])*Z[0] + (1-A[0][0])*Dagger(Z[1])*Z[1]) + \
		ti * (Z[0]*Dagger(Z[0]) + Z[1]*Dagger(Z[1]))

def score_constraints(score):
	chsh_expr = (A[0][0]*B[0][0] + (1-A[0][0])*(1-B[0][0]) + \
				 A[0][0]*B[1][0] + (1-A[0][0])*(1-B[1][0]) + \
				 A[1][0]*B[0][0] + (1-A[1][0])*(1-B[0][0]) + \
				 A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0])/4.0

	return [chsh_expr - score]

def get_subs():
	subs = {}
	# Get Alice and Bob's projective measurement constraints
	subs.update(ncp.projective_measurement_constraints(A,B))

	# Finally we note that Alice and Bob's operators should All commute with Eve's ops
	for a in ncp.flatten([A,B]):
		for z in Z:
			subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})

	return subs

def get_extra_monomials():
	monos = []

	# It should be sufficient to consider words of length at most 2 in the Z ops
	Z2 = [z for z in get_monomials(Z,2) if ncdegree(z)>=2]
	AB = ncp.flatten([A,B])
	for a in AB:
		for z in Z2:
			monos += [a*z]
	return monos[:]

def get_extra_monomials_fast():
	# A function that defines fewer but seemingly important monomial sets
	monos = []

	ZZ = Z + [Dagger(z) for z in Z]
	Aflat = ncp.flatten(A)
	Bflat = ncp.flatten(B)
	for a in Aflat:
		for z in Z:
			monos += [a*Dagger(z)*z]
		for b in Bflat:
			for z in ZZ:
				monos += [a*b*z]

	return monos[:]


def generate_quadrature(m):
	# Returns the nodes/weights of the Gauss-Radau quadrature
	t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
	t = t[0]
	return t, w

def compute_entropy(t, w):
	# We actually compute a faster lower bound on the problem.
	# Optimizing each node/weight term individually.

	cm = 0.0	# Constant term
	ent = 0.0	# NPA term

	for k in range(len(t)):
		# Multiplicative factor for kth term
		ck = w[k]/(t[k] * log(2))
		cm += ck

		# Change the objective to fit the new nodes
		new_objective = objective(t[k])
		sdp.set_objective(new_objective)

		# Solve sdp and add result to total
		sdp.solve('mosek')
		ent += ck * sdp.dual

	return cm + ent


# Some functions to help compute the known optimal bounds for the CHSH rates
def hmin(w):
	return -log2(1/2 + sqrt(2 - (8*w - 4)**2 / 4)/2)
def h(p):
	return -p*log2(p) - (1-p)*log2(1-p)
def Hvn(w):
	return 1 - h(1/2 + sqrt((8*w - 4)**2 / 4 - 1)/2)


import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from ncpol2sdpa.nc_utils import ncdegree, get_monomials
from ncpol2sdpa.solver_common import get_xmat_value
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy

LEVEL = 2						# Level of relaxation
WMAX = 0.5 + sqrt(2)/4			# Maximum CHSH score

# Description of Alice and Bobs devices (each input has 2 outputs)
A_config = [2,2]
B_config = [2,2]

# Operators in the problem Alice, Bob and Eve
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=0)


substitutions = {}			# substitutions to be made (e.g. projections)
moment_ineqs = []			# Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []				# Moment equalities (not needed here)
op_eqs = []					# Operator equalities (not needed here)
op_ineqs = []				# Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = []			# Extra monomials to add to the relaxation beyond the level.


# Get the relevant substitutions
substitutions = get_subs()

# Define the moment inequality related to chsh score
test_score = 0.85
score_cons = score_constraints(test_score)

# Get any extra monomials we wanted to add to the problem
extra_monos = get_extra_monomials_fast()

# Define the objective function (changed later)
obj = objective(1)

# Finally defining the sdp relaxation in ncpol2sdpa
ops = ncp.flatten([A,B,Z])
sdp = ncp.SdpRelaxation(ops, verbose = 0, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
					equalities = op_eqs[:],
					inequalities = op_ineqs[:],
					momentequalities = moment_eqs[:],
					momentinequalities = moment_ineqs[:] + score_cons[:],
					objective = obj,
					substitutions = substitutions,
					extramonomials = extra_monos)


# Note that the way chaospy works this will actually be a 2*M Gauss-Radau quadrature -- (m = 2*M)
M = 3
ts, ws = generate_quadrature(M)

# Test
ent = compute_entropy(ts, ws)
print(Hvn(test_score), ent)
exit()


"""
Now let's collect some data
"""
# We'll loop over the different CHSH scores and compute lower bounds on the rate of the protocol
scores = np.linspace(0.75, 0.83, 20)[:-1].tolist() + np.linspace(0.83, WMAX-0.0001, 20).tolist()
entropy = []
for score in scores:
	sdp.process_constraints(equalities = op_eqs[:],
						inequalities = op_ineqs[:],
						momentequalities = moment_eqs[:],
						momentinequalities = moment_ineqs[:] + score_constraints(score))
	ent = compute_entropy(ts, ws)
	entropy += [ent]
	print(score, Hvn(score), ent)

np.savetxt('./data/chsh_local_' + str(2*M) + 'M.csv', [scores, entropy], delimiter = ',')

entropy_h = [Hvn(score) for score in scores]
entropy_hmin = [hmin(score) for score in scores]
# np.savetxt('./data/analytic_H_chsh_local.csv', [scores, entropy_h], delimiter = ',')
# np.savetxt('./data/analytic_Hmin_chsh_local.csv', [scores, entropy_hmin], delimiter = ',')

import matplotlib.pyplot as plt
plt.plot(scores, entropy)
plt.plot(scores, entropy_h)
plt.plot(scores, entropy_hmin)
plt.ylim(0.0, 1.0)
plt.xlim(0.75, WMAX)
plt.savefig('chsh_local_rates.png')
plt.show()
