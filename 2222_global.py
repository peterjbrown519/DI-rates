"""
Script to compute converging lower bounds on the rates of global randomness
extracted from two devices that are constrained by some 2 input 2 output distribution.
More specifically, computes a sequence of lower bounds on the problem

            inf H(AB|X=0,Y=0,E)

where the infimum is over all quantum devices with some expected behaviour. See
the accompanying paper for more details (Figure 3)

Code also analyzes the scenario where we have inefficient detectors and implements
a subroutine to optimize the randomness gained from a family of two-qubit systems.
(Looks for a better two-qubit achievable distribution for a given detection efficiency)
"""

def objective(ti):
    """
    Returns the objective function for the faster computations.
    Randomness generation on X=0 and only two outcomes for Alice.

        ti  --    i-th node
    """
    return     A[0][0]*B[0][0]*(Dagger(Z[0]) + Z[0] + (1-ti)*Dagger(Z[0])*Z[0]) + \
            A[0][0]*(1-B[0][0])*(Dagger(Z[1]) + Z[1] + (1-ti)*Dagger(Z[1])*Z[1]) + \
            (1-A[0][0])*B[0][0]*(Dagger(Z[2]) + Z[2] + (1-ti)*Dagger(Z[2])*Z[2]) + \
            (1-A[0][0])*(1-B[0][0])*(Dagger(Z[3]) + Z[3] + (1-ti)*Dagger(Z[3])*Z[3]) + \
            ti * (Z[0]*Dagger(Z[0]) + Z[1]*Dagger(Z[1]) + Z[2]*Dagger(Z[2]) + Z[3]*Dagger(Z[3]))

def score_constraints(sys, Aops, Bops, eta=1.0):
    """
    Returns moment equality constraints generated by the two-qubit system specified by sys.
    In particular implements the constraints p(00|xy) and p(0|x), p(0|y) for each x,y.

        sys --  parameterization of two qubit system [theta, a0, a1, b0, b1]
                where state is
                                cos(theta) |00> + sin(theta) |11>
                and measurements are given by projectors
                                (id + cos(x) sz + sin(x) sx)/2
                defined by the remaining angles in sys with sz and sx being the Pauli z
                and Pauli x matrices.
        Aops -- Operators for Alice
        Bops -- Operators for Bob
        eta  -- Detection efficiency
    """

    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [theta, a0, a1, b0, b1] = sys[:]
    # Assume a pure two-qubit state of |00> + |11> form
    rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

    # Define the projectors for each of the measurements of Alice and Bob
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

    # Now collect the constraints subject to the inefficient detection distribution
    constraints = []
    # Constraints of form p(00|xy)
    for x in range(2):
        for y in range(2):
                   constraints += [Aops[x][0]*Bops[y][0] - (eta**2 * (rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + \
                               + eta*(1-eta)*((rho*qtp.tensor(A_meas[x][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[y][0])).tr().real) + \
                               + (1-eta)*(1-eta))]
    # Marginal constraints
    constraints += [Aops[0][0] - eta * (rho*qtp.tensor(A_meas[0][0], id)).tr().real - (1-eta)]
    constraints += [Bops[0][0] - eta * (rho*qtp.tensor(id, B_meas[0][0])).tr().real - (1-eta)]
    constraints += [Aops[1][0] - eta * (rho*qtp.tensor(A_meas[1][0], id)).tr().real - (1-eta)]
    constraints += [Bops[1][0] - eta * (rho*qtp.tensor(id, B_meas[1][0])).tr().real - (1-eta)]

    return constraints[:]

def sys2vec(sys, eta = 1.0):
    """
    Returns vector of probabilities p(00|xy), p(0|x), p(0|y) associated with sys in the same order
    as the constraints are specified in score_constraints function.

        sys --  parameterization of two qubit system [theta, a0, a1, b0, b1]
                where state is
                                cos(theta) |00> + sin(theta) |11>
                and measurements are given by projectors
                                (id + cos(x) sz + sin(x) sx)/2
                defined by the remaining angles in sys with sz and sx being the Pauli z
                and Pauli x matrices.
        eta  -- Detection efficiency
    """
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [theta, a0, a1, b0, b1] = sys[:]
    rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

    # Define the projectors for each of the measurements of Alice and Bob
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

    # Now collect the vector
    vec = []
    for x in range(2):
        for y in range(2):
            vec += [(eta**2 * (rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + \
                        + eta*(1-eta)*((rho*qtp.tensor(A_meas[x][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[y][0])).tr().real) + \
                        + (1-eta)*(1-eta))]

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
    vec = [raw_vec[2*k][0][0] - raw_vec[2*k + 1][0][0] for k in range(8)]
    return np.array(vec[:])

def dual_vec_to_improved_sys(dvec, init_sys, eta):
    """
    Given a dual vector from an sdp solution this optimizes the choice of system for a given eta.

            init_sys    --  starting system
            dvec        --  dual vector taken from SDP solved for init_sys
            eta         --  detection efficiency
    """
    # Define the function to optimize (x is sys vector)
    # we want to maximize so we minimize the negative.
    def f0(x):
        p = sys2vec(x, eta)
        return -np.dot(p, dvec)

    # Bounds on the values sys elements can take
    bounds = [[0, pi/2]] + [[-pi,pi] for _ in range(len(init_sys) - 1)]

    # Optmize qubit system using scipy
    res = minimize(f0, init_sys[:], bounds = bounds)

    # We return the opimal point found
    return res.x.tolist()[:]


def optimise_rate(SDP, sys, eta):
    """
    Function used to optimize the choice of sys for a given detection efficiency
    """
    NEEDS_IMPROVING = True            # Flag to check if we should continue optimizing
    FIRST_PASS = True                # Flag to check if we are in first loop through
    improved_sys = sys[:]            # Holds the optimized system choice
    best_sys = sys[:]                # Holds the best system choice
    dual_vec = np.zeros(8)            # Holds the dual vector of the sdp solution

    # We loop until NEEDS_IMPROVING flag is changed
    while(NEEDS_IMPROVING):

        # If it's not the initial pass then we optimize the system choice based
        # on the dual vector of the sdp from before
        if not FIRST_PASS:
            improved_sys = dual_vec_to_improved_sys(dual_vec, improved_sys[:], eta)

        # Apply the new system to the sdp
        score_cons = score_constraints(improved_sys[:], A, B, eta)
        SDP.process_constraints(equalities = op_eqs,
                            inequalities = op_ineqs,
                            momentequalities = moment_eqs[:] + score_cons[:],
                            momentinequalities = moment_ineqs)

        # Compute new dual vector and the rate
        dual_vec, new_ent = compute_dual_vector(SDP, improved_sys[:], eta)

        # Check if the result was a big enough improvement if not finalize stuff and return
        if not FIRST_PASS:
            if new_ent < current_ent + current_ent*EPS_M or new_ent < current_ent + EPS_A :
                NEEDS_IMPROVING = False
        else:
            # If first run through then this is the initial entropy
            starting_ent = new_ent
            current_ent = new_ent
            FIRST_PASS = False

        # If we improved update the best values
        if new_ent > current_ent:
            if VERBOSE > 0:
                print('Optimizing sys for eta =', eta, ' ... ', starting_ent, '->', new_ent)
            current_ent = new_ent
            best_sys = improved_sys[:]

    return current_ent, best_sys[:]

def compute_entropy(SDP):
    """
    Computes lower bound on H(AB|X=0,Y=0,E) using the fast (but less tight) method

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

def compute_dual_vector(SDP, sys, eta):
    """
    Extracts the vector from the dual problem(s) that builds into the affine function
    of the constraints that lower bounds H(A|X=0,E)

        SDP    --    sdp relaxation object
    """

    dual_vec = np.zeros(8)    # dual vector
    ck = 0.0                # kth coefficient
    ent = 0.0                # lower bound on H(A|X=0,E)

    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1
        ent = 0.0

    # Compute entropy and build dual vector from each sdp solved
    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k])

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

    Completely modifiable!
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

    for a in A[0]:
        for b in B[0]:
            for z in Z:
                monos += [a*b*z*Dagger(z), a*b*Dagger(z)*z]

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


import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import qutip as qtp
from joblib import Parallel, delayed, parallel_backend
from glob import glob
from scipy.optimize import minimize


LEVEL = 2                        # NPA relaxation level
M = 6                            # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)    # Nodes, weights of quadrature
KEEP_M = 0                        # Optimizing mth objective function?
VERBOSE = 2                        # If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-4, 1e-4        # Multiplicative/Additive epsilon in iterative optimization
# Some options for the data collection
NUM_SAMPLES = 1                    # Number of random systems we try
RAND_PROB = 0.5                    # Probability we choose the measurements uniformly
THETA_VAR = pi/32                # variance in state angle chosen
ANG_VAR = pi/16                    # variane in meansurement angles chosen
NUM_WORKERS = 1                    # Number of workers to parallelize to
NUM_SUBWORKERS = 4                # Number of subworkers to parallelize sdp to


# Description of Alice and Bobs devices (each input has 2 outputs)
A_config = [2,2]
B_config = [2,2]

# Operators in the problem Alice, Bob and Eve
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 4, hermitian=0)


substitutions = get_subs()          # substitutions to be made (e.g. projections)
moment_ineqs = []                   # Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []                     # Moment equalities (not needed here)
op_eqs = []                         # Operator equalities (not needed here)
op_ineqs = []                       # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = get_extra_monomials() # Extra monomials to add to the relaxation beyond the level.


# Define the moment equality constraints given by the distribution induced by sys and eta
test_sys = [0.519, -1.769, 0.041, -0.670, 0.875]        # Optimized system for eta = 0.96
test_eta = 0.96
score_cons = score_constraints(test_sys, A, B, test_eta)

ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
obj = objective(1)                # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_cons[:],
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)

# Test
# opt_rate, opt_sys = optimise_rate(sdp, test_sys[:], test_eta)
# print(opt_rate)
ent = compute_entropy(sdp)
print(ent)
exit()


"""
Now let's collect some data, we define the function that we will parallelize
"""
eta_range = np.linspace(0.7,0.85,10)[:-1].tolist() + \
            np.linspace(0.85, 0.95, 10)[:-1].tolist() + \
            np.linspace(0.95, 0.98, 10)[:-1].tolist() + \
            np.linspace(0.98, 1.00, 10).tolist()

# NOTE: FORMAT FOR THE SAVED DATA IS
#         [ETA, RATE, THETA, ANG0, ANG1, ...]

def task(ETA):
    """
    Given an eta try to optimize the choice of system and update the data
    """
    print('Starting eta ', ETA)
    fn = './data/kosaki_de_global_'+ str(2*M) + 'M_' + str(int(100000*ETA)) + '.csv'

    # We want good sys choices to propogate through the optimization for different
    # eta so we load the data for nearby eta and we shall also optimize their best system
    known_systems = []
    for filename in list(glob('./data/kosaki_de_global_'+ str(2*M) + 'M_*.csv')):
        if filename != fn:
            known_systems += [np.loadtxt(filename, delimiter = ',')]
    if len(known_systems) > 0:
        # order w.r.t. eta -- orders in ascending order
        known_systems.sort(key = lambda x: x[0])
        # find next and previous point
        FOUND_POINT = False
        point_count = 0
        # Find the data for the eta immediately before and after
        while not FOUND_POINT:
            if point_count > len(known_systems)-1:
                previous_sys = [None]
                next_sys = [None]
                FOUND_POINT = True
                break
            if known_systems[point_count][0] - ETA > 0:
                # now passed ETA to this should be previous system
                previous_sys = known_systems[point_count][:]
                if point_count > 0:
                    next_sys = known_systems[point_count-1][:]
                FOUND_POINT = True
            else:
                point_count += 1
    else:
        previous_sys = [None]
        next_sys = [None]

    # If we have optimized before then open the data and collect best sys and entropy
    try:
        data = np.loadtxt(fn, delimiter=',')
        BEST_RATE = data[1]
        BEST_SYS = data[2:]
        NEEDS_PRESOLVE = False
    except:
        # If we don't have any data then we should solve for some given system to start
        # things off
        BEST_RATE = -10
        NEEDS_PRESOLVE = True
        BEST_SYS = [0.795, 2.939, -2.336, 1.477, 0.444]


    # # ATTEMPT 0 -- If we need to solve for a fixed system
    # NEEDS_PRESOLVE = True
    # if NEEDS_PRESOLVE:
    #     try:
    #         print('Needs presolving, no file found...')
    #         # curr_sys = BEST_SYS[:]
    #         curr_sys = [0.3659, -1.223, 0.1757, -0.2855, 0.9986]
    #         opt_rate, opt_sys = optimise_rate(sdp, curr_sys[:], ETA, ts, ws)
    #         print(opt_rate, opt_sys)
    #
    #         if opt_rate > BEST_RATE:
    #             print('New best rate...', BEST_RATE, ' -> ', opt_rate)
    #             BEST_RATE = opt_rate
    #             BEST_SYS = opt_sys[:]
    #     except:
    #         pass

    # ATTEMPT 1 -- TRY BEST SYSTEM FROM PREVIOUS POINT
    try:
        print('trying previous point\'s current best system...')
        curr_sys = previous_sys[2:]
        opt_rate, opt_sys = optimise_rate(sdp, curr_sys[:], ETA, ts, ws)

        if opt_rate > BEST_RATE:
            print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
            BEST_RATE = opt_rate
            BEST_SYS = opt_sys[:]
    except:
        pass

    # ATTEMPT 2 -- TRY BEST SYSTEM FROM NEXT POINT
    try:
        print('trying next point\'s current best system...')
        curr_sys = next_sys[2:]
        opt_rate, opt_sys = optimise_rate(sdp, curr_sys[:], ETA, ts, ws)

        if opt_rate > BEST_RATE:
            print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
            BEST_RATE = opt_rate
            BEST_SYS = opt_sys[:]
    except:
        pass

    # ATTEMPT 3 -- TRYING RANDOM POINTS
    try:
        print('trying random systems')
        for _ in range(NUM_SAMPLES):
            # Deciding whether to sample uniformly or with so normal distribution about best system
            if np.random.random() < RAND_PROB:
                new_sys = [np.random.normal(BEST_SYS[0], THETA_VAR)] + np.random.uniform(-pi,pi, len(BEST_SYS) -1).tolist()
            else:
                new_sys = np.random.normal(BEST_SYS, [THETA_VAR] + [ANG_VAR for _ in range(len(BEST_SYS) -1)])

            opt_rate, opt_sys = optimise_rate(sdp, new_sys[:], ETA, ts, ws)

            if opt_rate > BEST_RATE:
                print('New best rate for eta', ETA, ': ', BEST_RATE, ' -> ', opt_rate)
                BEST_RATE = opt_rate
                BEST_SYS = opt_sys[:]
    except:
        pass


    np.savetxt(fn, [ETA, BEST_RATE] + np.array(BEST_SYS).tolist(), delimiter = ',')
    print('Finished eta ', ETA, ' with rate ', BEST_RATE)
    return 0

with parallel_backend("loky"):
    results = Parallel(n_jobs = NUM_WORKERS, verbose=0)(delayed(task)(eta_range[count]) for count in range(18,len(eta_range)-1))
