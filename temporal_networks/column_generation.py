#from os import getrandom
#from re import T
from audioop import mul
from curses.panel import top_panel
import numpy as np
#from scipy.stats.mvn import mvnun as rectangular
from scipy.stats import multivariate_normal as norm
from scipy import optimize
import sys
import gurobipy as gp
from gurobipy import GRB
from math import sqrt, log, exp
import numpy as np
import additional_functions as fn
import time
from timeout import timeout
import pygad
from LinearProgramParis import solveLP
from gecco_class import gecco
#from scipy.optimize import line_search

np.seterr(divide='raise')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
inf = 10000

def getStandardForm(PSTN, model, correlation=0):
    '''
    Description:    Makes matrices in the standard form of a Joint Chance Constrained Optimisation problem:

                    min     c^Tx

                    S.t.    A*vars <= b
                            P(xi <= T*vars + q ) >= 1 - alpha
                            xi = N(mu, cov)
    
    Input:          PSTN:   Instance of PSTN to be solved
    Output:         A:      m x n matrix of coefficients
                    vars:   n dimensional decision vector
                    b:      m dimensional vector of RHS values
                    c:      n dimensional vector of objective coefficients
                    T:      p x n matrix of coefficients
                    q:      p dimensional vector of RHS values
                    mu_xi:  p dimensional mean vector of xi
                    cov_xi: p x p dimensional correlation matrix of xi
    '''

    vars = PSTN.getProblemVariables()
    rvars = PSTN.getRandomVariables()
    cc = PSTN.getControllableConstraints()
    cu = PSTN.getUncontrollableConstraints()
    n = len(vars)
    m = 2 * len(cc)
    p = 2 * len(cu)
    r = len(rvars)

    corr = PSTN.correlation

    c = np.zeros(n)
    A = np.zeros((m, n))
    b = np.zeros(m)
    T = np.zeros((p, n))
    q = np.zeros((p))
    mu_X = np.zeros((r))
    D = np.zeros((r, r))
    psi = np.zeros((p, r))
    x0 = np.zeros((n))

    # Gets matrices for controllable constraints in form Ax <= b
    for i in range(len(cc)):
        ub = 2 * i
        lb = ub + 1
        start_i, end_i = vars.index(cc[i].source.id), vars.index(cc[i].sink.id)
        A[ub, start_i], A[ub, end_i], b[ub] = -1, 1, cc[i].intervals["ub"]
        A[lb, start_i], A[lb, end_i], b[lb] = 1, -1, -cc[i].intervals["lb"]

    # Gets initial solution from schedule
    tc = PSTN.getControllables()
    for i in range(len(x0)):
        x0[i] = model.getVarByName(tc[i].id).x

    # Gets matrices for joint chance constraint P(Psi omega <= T * vars + q) >= 1 - alpha
    for i in range(len(cu)):
        ub = 2 * i
        lb = ub + 1
        incoming = PSTN.incomingContingent(cu[i])
        if incoming["start"] != None:
            incoming = incoming["start"]
            start_i, end_i = vars.index(incoming.source.id), vars.index(cu[i].sink.id)
            T[ub, start_i], T[ub, end_i] = 1, -1
            T[lb, start_i], T[lb, end_i] = -1, 1
            q[ub] = cu[i].intervals["ub"]
            q[lb] = -cu[i].intervals["lb"]
            rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
            psi[ub, rvar_i] = -1
            psi[lb, rvar_i] = 1
            mu_X[rvar_i] = incoming.mu
            D[rvar_i][rvar_i] = incoming.sigma
        elif incoming["end"] != None:
            incoming = incoming["end"]
            start_i, end_i = vars.index(cu[i].source.id), vars.index(incoming.source.id)
            T[ub, start_i], T[ub, end_i] = 1, -1
            T[lb, start_i], T[lb, end_i] = -1, 1
            q[ub] = cu[i].intervals["ub"]
            q[lb] = -cu[i].intervals["lb"]
            rvar_i = rvars.index("X" + "_" + incoming.source.id + "_" + incoming.sink.id)
            psi[ub, rvar_i] = 1
            psi[lb, rvar_i] = -1
            mu_X[rvar_i] = incoming.mu
            D[rvar_i][rvar_i] = incoming.sigma
        else:
            raise AttributeError("Not an uncontrollable constraint since no incoming pstc")
    # Gets covariance matrix from correlation matrix
    cov_X = D @ corr @ np.transpose(D)

    # Performs transformation of X into eta where eta = psi X such that eta is a p dimensional random variable
    mu_eta = psi @ mu_X
    cov_eta = psi @ cov_X @ np.transpose(psi)
    # Adds regularization term to diagonals of covariance to prevent singularity
    cov_eta = cov_eta + 1e-6*np.identity(p)
    # Translates random vector eta into standard form xi = N(0, R) where R = D.eta.D^T
    # D = np.zeros((p, p))
    # for i in range(p):
    #     D[i, i] = 1/sqrt(cov_eta[i, i])
    # R = D @ cov_eta @ D.transpose()
    # T = D @ T
    # q = D @ (q - mu_eta)
    # mu_xi = np.zeros((p))
    # cov_xi = R
    z0 = T @ x0 + q
    return A, vars, b, c, T, q, mu_eta, cov_eta, z0, x0, psi
    
def Initialise(gecco, box = 6):
    '''
    Description:    Finds an initial feasible point such that the joint chance constraint is satisfied. Solves the following problem:

                    max     t

                    S.t.    A*vars <= b
                            z <= T*vars + q
                            z = 1 t
                            t <= box

                    And checks to see whether the point z satisfies the chance constraint P(xi <= z) >= 1 - alpha.
    
    Input:          gecco:   Instance of gecco class
                    box:    no of standard deviations outside which should be neglected
    Output:         m:      An instance of the Gurobi model class
    '''
    # Sets up and solves Gurobi opimisation problem
    m = gp.Model("initialisation")
    x = m.addMVar(len(gecco.vars), name=gecco.vars)
    z = m.addMVar(gecco.T.shape[0], vtype=GRB.CONTINUOUS, name="z")
    m.addConstr(gecco.A @ x <= gecco.b)
    m.addConstr(z <= gecco.T @ x + gecco.q)
    for i in range(gecco.T.shape[0]):
        m.addConstr(z[i] <= box * gecco.cov[i,i])
    m.addConstr(x[gecco.start_i] == 0)
    m.setObjective(gp.quicksum(z), GRB.MAXIMIZE)
    m.update()
    m.write("gurobi_files/initial.lp")
    m.optimize()

    # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            print("Variable {}: ".format(v.varName) + str(v.x))
    else:
        m.computeIIS()
        m.write("gurobi_files/initial.ilp")

    z_ = np.array(z.x)

    # Checks to see whether solution to z satisfies the chance constraint
    F0 = fn.prob(z_, gecco.mean, gecco.cov)
    phi = []
    #Adds p approximation points z^i = (z_1 = t,..,z_i = 0,..,z_p = t) for i = 1,2,..,p
    phi.append(-log(F0))
    z = np.c_[z_]
    for i in range(len(z_)):
        znew = np.copy(z_)
        znew[i] = 0
        z = np.hstack((z, np.c_[znew]))
        Fnew = fn.prob(znew, gecco.mean, gecco.cov)
        phi.append(-log(Fnew))

    # Initialises the matrix z and vector phi within the instance of gecco
    gecco.setZ(z)
    gecco.setPhi(np.array(phi))
    return m

def masterProblem(gecco):
    '''
    Description:    Solves the restricted master problem:

                    min.    sum_{i=0}^k{phi^i * lambda^i}

                    S.t.    A*vars <= b
                            T*vars + q >= sum_{i=0}^k{lambda^i z^i}
                            sum_{i=0}^k{lambda^i} = 1
                            lambda^i >= 0

                    And returns a Gurobi model instance containing solution and optimal objective for current iteration.
    
    Input:          gecco:   Instance of gecco class
    Output:         m:      An instance of the Gurobi model class
                    zsol:   Result sum_{i=0}^k{lambda^i z^i}
    '''
    # Sets up and solves the restricted master problem
    k = np.shape(gecco.z)[1]
    p = len(gecco.q)
    m = gp.Model("iteration_" + str(k))
    x = m.addMVar(len(gecco.vars), name=gecco.vars)
    bounds = m.addMVar(p, name="bounds")
    lam = m.addMVar(k, name="lambda")
    phi = m.addMVar(1, name = "phi")
    m.addConstr(gecco.A @ x <= gecco.b, name="cont")
    for i in range(p):
        m.addConstr(gecco.z[i, :]@lam <= gecco.T[i,:]@x + gecco.q[i], name="z{}".format(i))
    m.addConstr(bounds == gecco.T@x + gecco.q)
    m.addConstr(x[gecco.start_i] == 0, name="x0")
    m.addConstr(lam.sum() == 1, name="sum_lam")
    m.addConstr(lam @ gecco.phi == phi, name="phi")
    m.setObjective(phi, GRB.MINIMIZE)
    m.update()
    m.write("gurobi_files/rmp.lp")
    m.optimize()

    # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
    if m.status == GRB.OPTIMAL:
        print('\n objective: ', m.objVal)
        print("\n probability: ", exp(-m.objVal))
        print('\n Vars:')
        for v in m.getVars():
            if "lambda" in v.varName and v.x == 0:
                continue
            else:
                print("Variable {}: ".format(v.varName) + str(v.x))
        m.write("gurobi_files/rmp.sol")

    # Queries Gurobi to get values of dual variables and cbasis
    constraints = m.getConstrs()
    cnames = m.getAttr("ConstrName", constraints)
    mu, cb = [], []
    for i in range(len(cnames)):
        if cnames[i][0] == "z":
            mu.append(constraints[i].getAttr("Pi"))
            cb.append(constraints[i].getAttr("CBasis"))
        elif cnames[i] == "sum_lam":
            nu = constraints[i].getAttr("Pi")

    mu = np.c_[np.array(mu)]

    # Sets the dual values and cbasis within the instance of gecco to the optimal value for the current iteration
    gecco.setDuals({"mu": mu, "nu": nu})
    gecco.setCbasis(np.array(cb))

    # Gets values for variables lambda and evaluates current value of sum_{i=0}^k{lambda^i z^i} and sum(i=0)^k{lambda^i phi^i}
    lam_sol = np.array(lam.x)
    z_sol = np.array(sum([lam_sol[i]*gecco.z[:, i] for i in range(np.shape(gecco.z)[1])]))

    # Calculates the Probability using the bounds
    bounds = []
    for v in m.getVars():
        if "bounds" in v.varName:
            bounds.append(v.x)
    bounds = np.array(bounds)
    print(gecco.mean, gecco.cov)
    print("Evaluated probability: ", norm(gecco.mean, gecco.cov).cdf(bounds))
    return (m, np.c_[z_sol])

    
def column_generation_nm(z, gecco):
    '''
    Description:    Solves the column generaion problem (below) via SciPy optimize:

                    min_z.  -u^Tz - v*phi(z) - nu    

                    Every time we evaluate phi(z) we save the result to new_cols and new_phis return all new
    
    Input:          JCCP:       Instance of JCCP class
    Output:         columns:    Matrix of new columns to be added
                    values:     Vector of phi values to be added
                    f:          Final function value (reduced cost)
                    status:     Boolean stating whether the optimisation was successful
    '''
    columns_to_add = []
 
    duals = gecco.getDuals()
    mu, nu = fn.flatten(duals["mu"]), duals["nu"]
    mean, cov = gecco.mean, gecco.cov

    z = fn.flatten(z)
    start = time.time()
    print("Mean: ", mean)
    print("Covariance: ", cov)
    print("Duals:", mu, nu)
    print("z", z)

    def dualf(z):
        phi = -log(norm(mean, cov).cdf(z))
        f = phi -np.dot(mu, z) - nu
        if f < 0:
            # Keeps track of new columns in global variable and adds all points that have positive
            # reduced cost. This allows us to add multiple points at a time
            included = True
            for element in gecco.new_cols:
                if np.array_equal(z, element):
                    included = False
            if included == True:
                columns_to_add.append((np.copy(z), phi))
        return f

    # Adds bounds to prevent variables being non-negative
    bounds = []
    for i in range(len(z)):
        bound = (0.00001, inf)
        bounds.append(bound)

    res = optimize.minimize(dualf, z, method = "Nelder-Mead", bounds=bounds)
    end = time.time()
    print("Time taken: ", end - start)
    print("\n", res)
    z = res.x
    f = res.fun
    status = res.success

    for i in range(len(columns_to_add)):
        # print("Adding column: ")
        # print(columns_to_add[i])
        gecco.addColumn(np.c_[columns_to_add[i][0]], columns_to_add[i][1])
    
    return gecco, f

def column_generation_genetic(z, gecco):
    #print("Solving CG")
    # Creates a new list of new columns to add so that we can add multiple columns at once
    columns_to_add = []

    duals = gecco.getDuals()
    mu, nu = fn.flatten(duals["mu"]), duals["nu"]
    mean, cov = gecco.mean, gecco.cov
    # print("Duals:", mu, nu)
    # print("Mean: ", mean)
    # print("Covariance: ", cov)

    # Generates initial population
    z = fn.flatten(z)
    others = np.random.rand(9,len(z))
    for i in range(others.shape[0]):
        for j in range(len(z)):
            others[i, j] = others[i, j] * 6 * cov[j, j]
    initial = np.vstack((z, others))

    def genetic_dualf(x, solution_idx):
        # print("\n")
        try:
            prob = norm(mean, cov).cdf(x)
            phi = -log(prob)
            f = np.dot(mu, x) + nu - phi
        except:
            f = -inf
        if f > 0:
            #  Keeps track of new columns in global variable and adds all points that have positive
            # reduced cost. This allows us to add multiple points at a time
            included = True
            for element in columns_to_add:
                if np.array_equal(x, element):
                    included = False
            if included == True:
                # print("Dual has negative reduced cost so adding point. ")
                # print(x)
                # print("Probability, ", norm(mean, cov, allow_singular=True).cdf(x))
                # print("Phi to add", phi)
                columns_to_add.append((np.copy(x), phi))
                # for i in columns_to_add:
                #     print(i)
                # print(vals_to_add)
        return f
    
    ga = pygad.GA(num_generations=200,
                    num_parents_mating=2,
                    fitness_func=genetic_dualf,
                    initial_population=initial,
                    save_best_solutions=True,
                    mutation_by_replacement=True,
                    random_mutation_min_val=0,
                    mutation_percent_genes=20,
                    stop_criteria =  "saturate_20"
                    #stop_criteria =  "reach_0"
    )

    ga.run()
    obj = -ga.best_solutions_fitness[-1]
    print("Reduced Cost : ", )
    print(obj)
    print("Solution")
    print(ga.best_solutions_fitness)
    print(ga.best_solutions)
    # print("\nNew columns to add : ")
    # print("Here", columns_to_add)
    for i in range(len(columns_to_add)):
        # print("Adding column: ")
        # print(columns_to_add[i])
        gecco.addColumn(np.c_[columns_to_add[i][0]], columns_to_add[i][1])
    return gecco, obj


def column_generation_lbfgsb(z, gecco):
    '''
    Description:    Solves the column generaion problem (below) via SciPy optimize:

                    min_z.  -u^Tz - v*phi(z) - nu    

                    Every time we evaluate phi(z) we save the result to new_cols and new_phis return all new
    
    Input:          JCCP:       Instance of JCCP class
    Output:         columns:    Matrix of new columns to be added
                    values:     Vector of phi values to be added
                    f:          Final function value (reduced cost)
                    status:     Boolean stating whether the optimisation was successful
    '''
    columns_to_add = []
 
    duals = gecco.getDuals()
    mu, nu = fn.flatten(duals["mu"]), duals["nu"]
    mean, cov, psi = gecco.mean, gecco.cov, gecco.psi
    
    z = fn.flatten(z)
    start = time.time()

    def dualf(z):
        phi = -log(norm(mean, cov).cdf(z))
        f = phi -np.dot(mu, z) - nu
        if f < 0:
            # Keeps track of new columns in global variable and adds all points that have positive
            # reduced cost. This allows us to add multiple points at a time
            included = True
            for element in gecco.new_cols:
                if np.array_equal(z, element):
                    included = False
            if included == True:
                columns_to_add.append((np.copy(z), phi))
        print("Dual Value", f)
        return f

    def grad_prob(z):
        n = int(np.shape(mean)[0])
        #I = get_active_indices(z)
        dz = []
        for i in range(n):
            #if  I[i] != 0:
            #    dz.append(0)
            #else:
            bar_mean = np.delete(mean, i)
            bar_cov = np.delete(np.delete(cov, i, 0), i, 1)
            bar_z= np.delete(z, i)
            bar_F = norm(bar_mean, bar_cov).cdf(bar_z)
            f = norm(mean[i], sqrt(cov[i, i])).pdf(z[i])
            dz.append(f * bar_F)
        return np.array(dz)

    def get_active_indices(z):
        shape = np.shape(psi)
        m, s = shape[0], shape[1]
        I = []
        for i in range(m):
            # Sets up and solves the LP from Henrion and Moller 2012 (proposition 4.1)
            model = gp.Model("index_check_{}".format(i))
            model.setParam('OutputFlag', 0)
            u = model.addMVar(m)
            x = model.addMVar(s, lb=-GRB.INFINITY)
            model.addConstr(psi @ x + u == z - mean)
            model.setObjective(u[i], GRB.MINIMIZE)
            model.update()
            model.optimize()
            # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
            if model.status != GRB.OPTIMAL:
                model.computeIIS()
                model.write("gurobi_files/active_indices.lp")
                model.write("gurobi_files/active_indices.ilp")
            I.append(model.objVal)
        print("I", I)
        return I
    
    def gradf(z):
        print("\nZ ", z)
        f = norm(mean, cov).cdf(z)
        print("Probability ", f)
        print("Mu", mu)
        g = grad_prob(z)
        print("Gradient of probability: ", g)
        print("Jacobian", -g/f - mu)
        return -grad_prob(z)/norm(mean, cov).cdf(z) - mu
    
    def gradf_approx(z):
        return optimize.approx_fprime(z, dualf)

    # Adds bounds to prevent variables being non-negative
    bounds = []
    for i in range(len(z)):
        bound = (0.00001, inf)
        bounds.append(bound)

    res = optimize.minimize(dualf, z, jac = gradf, method = "L-BFGS-B", bounds=bounds)
    end = time.time()
    print("Time taken: ", end - start)
    print("\n", res)
    z = res.x
    f = res.fun
    status = res.success

    for i in range(len(columns_to_add)):
        # print("Adding column: ")
        # print(columns_to_add[i])
        gecco.addColumn(np.c_[columns_to_add[i][0]], columns_to_add[i][1])

    return gecco, f
    
def solve(PSTN, tolog=False, logfile = None, max_iterations = 50, column_generation_solver = column_generation_lbfgsb):
    '''
    Description:    Solves the problem of a joint chance constrained PSTN strong controllability via primal-dual column
                    generation method.
    
    Input:          PSTN:           Instance of PSTN class
                    alpha:          Allowable tolerance on risk:
                                    e.g. P(success) >= 1 - alpha
                    epsilon:        An allowable upper bound on the distance between the current solution and the global optimum
                                    e.g. (UB - LB)/LB <= epsilon    
                    log:            Boolean, whether or not to print to log file
                    logfile:        File to save log to
                    max_iteraions:  Option to set maxmimum number of iterations
                    cg_tol:         Tolerance to use with Column Generation optimisation (see: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    Output:         m:              An instance of the Gurobi model class which solves the joint chance constrained PSTN
                    problem:        An instance of the JCCP class containing problem results
    '''
    n_iterations = 0
    if tolog == True:
        saved_stdout = sys.stdout
        sys.stdout = open("logs/{}.txt".format(logfile), "w+")
    
    # Translates the PSTN to the standard form of a JCCP and stores the matrices in an instance of the JCCP class
    start = time.time()
    m, results = solveLP(PSTN, PSTN.name + "LP", pres = 15)
    matrices = getStandardForm(PSTN, m)
    A, vars, b, c, T, q, mu, cov, z0, x0, psi = matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5], matrices[6], matrices[7], matrices[8], matrices[9], matrices[10]
    problem = gecco(A, vars, b, c, T, q, mu, cov, psi)
    problem.start_i = problem.vars.index(PSTN.getStartTimepointName())
    
    # Initialises the problem with k approximation points
    m = Initialise(problem)
    F0 = fn.prob(z0, problem.mean, problem.cov)
    phi0 = -log(F0)
    problem.addColumn(np.c_[z0], phi0)
    k = len(problem.phi)

    # Solves the master problem
    print("\nSolving master problem with {} approximation points".format(k))
    m, z_m = masterProblem(problem)
    problem.add_master_time(time.time() - start, m.objVal)
    problem.addSolution(m)
    print("Current objective is: ", m.objVal)
    print("Current probability is: ",  exp(-m.objVal))

    # Solves the column generation problem
    print("\nSolving Column Generation")
    problem, obj = column_generation_solver(z_m, problem)
    k = len(problem.phi)

    while n_iterations <= max_iterations and obj < 0:
        n_iterations += 1

        print("\nSolving master problem with {} approximation points".format(k))
        m, z_m = masterProblem(problem)
        problem.add_master_time(time.time() - start, m.objVal)
        problem.addSolution(m)
        print("Current objective is: ", m.objVal)
        print("Current probability is: ",  exp(-m.objVal))

        print("\nSolving Column Generation")
        problem, obj = column_generation_solver(z_m, problem)

    end = time.time()
    solution_time = end - start
    if n_iterations <= max_iterations:
        problem.setSolved(True)

    problem.setSolutionTime(solution_time)
    print("\nFinal solution found: ")
    print("Solution time: ", solution_time)
    print("Final Probability is: ", problem.getCurrentProbability())
    print('objective: ', m.objVal)
    print('Vars:')
    # lambdas = []
    for v in m.getVars():
        # if "lambda" in v.varName and v.x == 0:
        #     continue
        # else:
        print("Variable {}: ".format(v.varName) + str(v.x))
            # if "lambda" in v.varName:
            #     m = re.search(r"\[([A-Za-z0-9_]+)\]", v.varName)
            #     lambdas.append(int(m.group(1)))
    # print("\nColumns: ")
    # print(problem.z)
    # Fs, phis = [], []
    # for i in range(np.shape(problem.z)[1]):
    #     F = fn.prob(problem.z[:, i], problem.mean, problem.cov)
    #     phi = -log(F)
    #     Fs.append(F)
    #     phis.append(phi)
    # print("\nProbabilities: ")
    # print(Fs)
    # print("\nPhis: ")
    # print(phis)
    # print([phis[i] - problem.phi[i] for i in range(len(phis))])

    if tolog == True:
        sys.stdout.close()
        sys.stdout = saved_stdout
    return m, problem
