"""
Script to compute converging lower bounds on the rates of local randomness
extracted from two devices achieving a minimal expected CHSH score. More specifically,
computes a sequence of lower bounds on the problem

    inf H(A|X=0,E)

where the infimum is over all quantum devices achieving a CHSH score of w. See
the accompanying paper for more details (this script can be used to generate data for Figure 1).
"""


def objective(ti):
    """
    Returns the objective function for the faster computations.
    Randomness generation on X=0 and only two outcomes for Alice.

        ti  --    i-th node
    """
    obj = 0.0
    F = [A[0][0], 1-A[0][0]]
    for a in range(len(F)):
    obj += F[a] * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

def score_constraints(score):
    """
    Returns CHSH score constraint
    """
    chsh_expr = (A[0][0]*B[0][0] + (1-A[0][0])*(1-B[0][0]) + \
     A[0][0]*B[1][0] + (1-A[0][0])*(1-B[1][0]) + \
     A[1][0]*B[0][0] + (1-A[1][0])*(1-B[0][0]) + \
     A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0])/4.0

    return [chsh_expr - score]

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


def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

        m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

def compute_entropy(SDP):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

        SDP -- sdp relaxation object
    """
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    if KEEP_M:
    num_opt = len(T)
    else:
    num_opt = len(T) - 1

    for k in range(num_opt):
    ck = W[k]/(T[k] * log(2))

    # Get the k-th objective function
    new_objective = objective(T[k])

    SDP.set_objective(new_objective)
    SDP.solve('mosek')

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
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy

LEVEL = 2                        # NPA relaxation level
M = 4                            # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)    # Nodes, weights of quadrature
KEEP_M = 0                        # Optimizing mth objective function?
VERBOSE = 1                        # If > 1 then ncpol2sdpa will also be verbose
WMAX = 0.5 + sqrt(2)/4            # Maximum CHSH score

# Description of Alice and Bobs devices (each input has 2 outputs)
A_config = [2,2]
B_config = [2,2]

# Operators in the problem Alice, Bob and Eve
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=0)


substitutions = {}            # substitutions to be made (e.g. projections)
moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []                # Moment equalities (not needed here)
op_eqs = []                    # Operator equalities (not needed here)
op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = []            # Extra monomials to add to the relaxation beyond the level.


# Get the relevant substitutions
substitutions = get_subs()

# Define the moment inequality related to chsh score
test_score = 0.85
score_cons = score_constraints(test_score)

# Get any extra monomials we wanted to add to the problem
extra_monos = get_extra_monomials()

# Define the objective function (changed later)
obj = objective(1)

# Finally defining the sdp relaxation in ncpol2sdpa
ops = ncp.flatten([A,B,Z])
sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
    equalities = op_eqs[:],
    inequalities = op_ineqs[:],
    momentequalities = moment_eqs[:],
    momentinequalities = moment_ineqs[:] + score_cons[:],
    objective = obj,
    substitutions = substitutions,
    extramonomials = extra_monos)

# # Test
# ent = compute_entropy(sdp)
# print("Analytical bound:", Hvn(test_score))
# print("SDP bound:" , ent)
# exit()


"""
Now let's collect some data
"""
# We'll loop over the different CHSH scores and compute lower bounds on the rate of the protocol
scores = np.linspace(0.75, WMAX-1e-4, 20).tolist()
entropy = []
for score in scores:
    # Modify the CHSH score
    sdp.process_constraints(equalities = op_eqs[:],
    inequalities = op_ineqs[:],
    momentequalities = moment_eqs[:],
    momentinequalities = moment_ineqs[:] + score_constraints(score))
    # Get the resulting entropy bound
    ent = compute_entropy(sdp)
    entropy += [ent]
    print(score, Hvn(score), ent)

# np.savetxt('./data/chsh_local_' + str(2*M) + 'M.csv', [scores, entropy], delimiter = ',')

entropy_h = [Hvn(score) for score in scores]
entropy_hmin = [hmin(score) for score in scores]
# np.savetxt('./data/analytic_H_chsh_local.csv', [scores, entropy_h], delimiter = ',')
# np.savetxt('./data/analytic_Hmin_chsh_local.csv', [scores, entropy_hmin], delimiter = ',')

"""
Plot and compare
"""
import matplotlib.pyplot as plt
plt.plot(scores, entropy)
plt.plot(scores, entropy_h)
plt.plot(scores, entropy_hmin)
plt.ylim(0.0, 1.0)
plt.xlim(0.75, WMAX)
# plt.savefig('chsh_local_rates.png')
plt.show()
