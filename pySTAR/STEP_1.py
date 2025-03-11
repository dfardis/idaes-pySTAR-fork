# Build NLP to be solved by BARON
# NLP is formed by relaxing MINLP constraints of symbolic regression tree

from pyomo.environ import *
import numpy as np


def SRT(X, y):
    """
    Defines the symbolic regression tree (m.y relaxed)

    Parameters
    ----------
    X: matrix
       input data (n_data = X.shape[0], p = X.shape[1])

    y: vector
       target values

    Returns
    -------

       symbolic regression tree

    """
    # Hyperparameters
    depth = 3  ###
    eps = 1e-6
    c_lo = min(-100, np.min(X))
    c_up = max(100, np.max(X))
    v_lo = -100
    v_up = 100

    # Parameters given from dataset
    n_data = X.shape[0]
    n_data_set = [
        i + 1 for i in range(n_data)
    ]  # to index of instances start with 1 and end with n_data
    p = X.shape[1]
    Xposi, Xnega, Xzero = XsetGenerator(
        X
    )  # set of positive, negative, and zero values of X

    # Operators
    VARS = ["x_" + str(i + 1) for i in range(p)]
    L = ["cst"] + VARS
    B = ["+", "-", "*", "/"]
    U = ["**0.5", "log", "exp"]
    Opair = []
    Opair.append(["log", "exp"])
    O = B + U + L

    # Nodes and their allowed operators
    N, T = node_definer(depth)  # Index of the nodes
    NnotT = np.setdiff1d(N, T)
    Nperfect = NnotT
    Y = Y_define(O, NnotT, L, T)  # set of allowed pair of (operator, node index)

    ###### Model Set Up ############################################################################
    m = ConcreteModel()
    m.y = Var(Y, domain=UnitInterval)
    m.c = Var(N, domain=Reals, bounds=(c_lo, c_up))
    m.v = Var(n_data_set, N, domain=Reals, bounds=(v_lo, v_up))
    # m.eps = Var(n_data_set, N, domain=NonNegativeReals, bounds=(eps,0.5), initialize=eps)

    # Objective
    def sse(m):
        """
        Calculates the root mean square error between the actual target values and the tree's predictions

        Parameters
        ----------
        m: -
           pyomo model

        Returns
        -------
        float
           root mean square error between the actual target values and the tree's predictions

        """
        return (
            (sum((y[i - 1] - m.v[i, 1]) ** 2 for i in range(1, n_data + 1))) / n_data
        ) ** 0.5

    m.obj = Objective(rule=sse, sense=minimize)

    # Tree-defining constraints ################################################################
    @m.Constraint(NnotT)
    def tdc23a_rule(m, n):
        """
        if an operator (binary or unary) is assigned to node n, then the right node (2*n+1) must exist

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        Returns
        -------

           tree-defining equality constraint

        """
        return sum(m.y[(n, o)] for o in B + U) == sum(
            m.y[(2 * n + 1, o)] for o in O if (2 * n + 1, o) in Y
        )

    @m.Constraint(NnotT)
    def tdc23b_rule(m, n):
        """
        if a binary operator is assigned to node n, then the left node (2*n) must exist

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        Returns
        -------

           tree-defining equality constraint
        """
        return sum(m.y[(n, o)] for o in B) == sum(
            m.y[(2 * n, o)] for o in O if (2 * n, o) in Y
        )

    @m.Constraint(N)
    def tdcA1a_rule(m, n):
        """
        At most one operator or operand o can be assigned at each node n

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        Returns
        -------

           tree-defining inequality constraint
        """
        return sum(m.y[(n, o)] for o in O if (n, o) in Y) <= 1

    @m.Constraint()
    def tdcA1b_rule(m, n):
        """
        At least one variable must appear in the tree expression

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        Returns
        -------

           tree-defining inequality constraint
        """
        return sum(sum(m.y[(n, o)] for o in VARS if (n, o) in Y) for n in N) >= 1

    ### Value-defining constraints ################################################################
    @m.Constraint(n_data_set, N)
    def vdc24a_rule(m, i, n):
        """
        if a variable is assigned to node n, then return the value of the variable.
        if a variable is not assigned to node n, then let the value of the node range between the bounds v_lo and v_up.

        Parameters
        ----------
        m: -
        pyomo model

        n: natural number
        node index

        i: natural number
        sample point index

        Returns
        -------

        value-defining inequality constraint
        """
        return m.v[i, n] <= sum(
            X[i - 1, j - 1] * m.y[(n, "x_" + str(j))] for j in range(1, p + 1)
        ) + v_up * sum(m.y[(n, o)] for o in B + U + ["cst"] if (n, o) in Y)

    @m.Constraint(n_data_set, N)
    def vdc24b_rule(m, i, n):
        """
           Completing the previous constraint.
           if a variable is assigned to node n, then return the value of the variable.
           if a variable is not assigned to node n, then let the value of the node range between the bounds v_lo and v_up.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] >= sum(
            X[i - 1, j - 1] * m.y[(n, "x_" + str(j))] for j in range(1, p + 1)
        ) + v_lo * sum(m.y[(n, o)] for o in B + U + ["cst"] if (n, o) in Y)

    m.vdc24b = Constraint(n_data_set, N, rule=vdc24b_rule)

    @m.Constraint(n_data_set, N)
    def consta_rule(m, i, n):
        """
        If a constant is assigned to node n, then return the value of the constant.
        If a constant is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] - m.c[n] <= (v_up - c_lo) * (1 - m.y[(n, "cst")])

    @m.Constraint(n_data_set, N)
    def constb_rule(m, i, n):
        """
        Completing the previous constraint.
        If a constant is assigned to node n, then return the value of the constant.
        If a constant is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] - m.c[n] >= (v_lo - c_up) * (1 - m.y[(n, "cst")])

    @m.Constraint(n_data_set, NnotT)
    def adda_rule(m, i, n):
        """
        If the addition operator is assigned to node n, then return the sum of the two children nodes.
        If the addition operator is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] - (m.v[i, 2 * n] + m.v[i, 2 * n + 1]) <= (v_up - 2 * v_lo) * (
            1 - m.y[(n, "+")]
        )

    @m.Constraint(n_data_set, NnotT)
    def addb_rule(m, i, n):
        """
        Completing the previous constraint.
        """
        return m.v[i, n] - (m.v[i, 2 * n] + m.v[i, 2 * n + 1]) >= (v_lo - 2 * v_up) * (
            1 - m.y[(n, "+")]
        )

    @m.Constraint(n_data_set, NnotT)
    def suba_rule(m, i, n):
        """
        If the subtraction operator is assigned to node n, then return the difference of the two children nodes.
        If the subtraction operator is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] - (m.v[i, 2 * n] - m.v[i, 2 * n + 1]) <= (2 * v_up - v_lo) * (
            1 - m.y[(n, "-")]
        )

    @m.Constraint(n_data_set, NnotT)
    def subb_rule(m, i, n):
        """
        Completing the previous constraint.
        """
        return m.v[i, n] - (m.v[i, 2 * n] - m.v[i, 2 * n + 1]) >= (2 * v_lo - v_up) * (
            1 - m.y[(n, "-")]
        )

    @m.Constraint(n_data_set, NnotT)
    def multa_rule(m, i, n):
        """
        If the multiplication operator is assigned to node n, then return the product of the two children nodes.
        If the multiplication operator is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] - (m.v[i, 2 * n] * m.v[i, 2 * n + 1]) <= (
            v_up - min(v_lo**2, v_lo * v_up, v_up**2)
        ) * (1 - m.y[(n, "*")])

    @m.Constraint(n_data_set, NnotT)
    def multb_rule(m, i, n):
        """
        Completing the previous constraint.
        """
        return m.v[i, n] - (m.v[i, 2 * n] * m.v[i, 2 * n + 1]) >= (
            v_lo - max(v_lo**2, v_up**2)
        ) * (1 - m.y[(n, "*")])

    @m.Constraint(n_data_set, NnotT)
    def diva_rule(m, i, n):
        """
        If the division operator is assigned to node n, then return the ratio of the two children nodes.
        If the division operator is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] * m.v[i, 2 * n + 1] - (m.v[i, 2 * n]) <= (
            max(v_lo**2, v_up**2) - v_lo
        ) * (1 - m.y[(n, "/")])

    @m.Constraint(n_data_set, NnotT)
    def divb_rule(m, i, n):
        """
        Completing the previous constraint.
        """
        return m.v[i, n] * m.v[i, 2 * n + 1] - (m.v[i, 2 * n]) >= (
            min(v_lo**2, v_lo * v_up, v_up**2) - v_up
        ) * (1 - m.y[(n, "/")])

    @m.Constraint(n_data_set, NnotT)
    def divc_rule(m, i, n):
        """
        If the division operator is assigned to node n, then the denominator should not be zero.
        If the division operator is not assigned to node n, then an inequality which is always true is generated.

        In Alison's paper the constraint is written instead as: eps * m.y[(n, "/")] <= m.v[i, 2 * n] ** 2, which means that
        the square of the right node (denominator of the ratio) should be greater than a small positive number epsilon

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return ((m.v[i, 2 * n + 1]) ** 2) * m.y[(n, "/")] >= m.y[(n, "/")] - 1 + eps

    # def divd_rule(m, i, n):
    #     return eps * m.y[(n, "/")] <= m.v[i, 2 * n + 1] ** 2
    # m.divd = Constraint(n_data_set, NnotT, rule=divd_rule)

    @m.Constraint(n_data_set, NnotT)
    def sqra_rule(m, i, n):
        """
        If the square root operator is assigned to node n, then return the root of the right children node.
        If the square root operator is not assigned to node n, then let the value of the node range between bounds.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] ** 2 - (m.v[i, 2 * n + 1]) <= (
            max(v_lo**2, v_up**2) - v_lo
        ) * (1 - m.y[(n, "**0.5")])

    @m.Constraint(n_data_set, NnotT)
    def sqrb_rule(m, i, n):
        """
        Completing the previous constraint.
        """
        return m.v[i, n] ** 2 - (m.v[i, 2 * n + 1]) >= (-v_up) * (1 - m.y[(n, "**0.5")])

    @m.Constraint(n_data_set, NnotT)
    def sqrc_rule(m, i, n):
        """
        If the square root operator is assigned to node n, then argument of the square root should be nonnegative. (same as in Alison's paper)

        In Leyffer's paper the constaint is written as: eps - m.v[i, 2 * n + 1] <= (eps - v_lo) * (1 - m.y[(n, "**0.5")])

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, 2 * n + 1] * m.y[n, "**0.5"] >= 0

    @m.Constraint(n_data_set, NnotT)
    def expa_rule(m, i, n):
        """
        If the exponential operator is assigned to node n,
        then let the value of the node be lower or equal than the exponential of the right children node.
        If the exponential operator is not assigned to node n, then let the value of the node be bounded above.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, n] - exp(m.v[i, 2 * n + 1]) <= v_up * (1 - m.y[(n, "exp")])

    @m.Constraint(n_data_set, NnotT)
    def expb_rule(m, i, n):
        """
        If the exponential operator is assigned to node n,
        then let the value of the node be greater or equal than the exponential of the right children node.
        Along with the previous constraint, this means that the value of the node is the exponential of the right children node.

        If the exponential operator is not assigned to node n, then let the value of the node be bounded below by the bound exp(m.v[i, 2 * n + 1]) + (v_lo - e_up).
        If v_up is greater than 10, then set e_up = 1e5. Otherwise e_up = exp(v_up), same as in Leyffer's paper.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        if v_up >= 10:
            e_up = 1e5
        else:
            e_up = exp(v_up)
        return m.v[i, n] - exp(m.v[i, 2 * n + 1]) >= (v_lo - e_up) * (
            1 - m.y[(n, "exp")]
        )

    @m.Constraint(n_data_set, NnotT)
    def loga_rule(m, i, n):
        """
        If the logarithm operator is assigned to node n,
        then let the value of the node be lower or equal than the logarithm of the right children node.

        If the logarithm operator is not assigned to node n, then let the (exponential) value of the node be bounded above.
        If v_up - v_lo >= 10, then set e_up_lo = 1e5.
        If exp(v_up - v_lo) < -1e-5, then set e_up_lo = -1e5.
        Otherwise e_up_lo = exp(v_up - v_lo), same as in Leyffer's paper.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        if v_up - v_lo >= 10:
            e_up_lo = 1e5
        elif exp(v_up - v_lo) < -1e-5:
            e_up_lo = -1e5
        else:
            e_up_lo = exp(v_up - v_lo)
        return exp(m.v[i, n]) - m.v[i, 2 * n + 1] <= e_up_lo * (1 - m.y[(n, "log")])

    @m.Constraint(n_data_set, NnotT)
    def logb_rule(m, i, n):
        """
        If the logarithm operator is assigned to node n,
        then let the value of the node be greater or equal than the logarithm of the right children node.
        Along with the previous constraint, this means that the value of the node is the logarithm of the right children node.

        If the logarithm operator is not assigned to node n, then let the (exponential) value of the node be bounded above.
        If v_up - v_lo >= 10, then set e_up_lo = 1e5.
        If exp(v_up - v_lo) < -1e-5, then set e_up_lo = -1e5.
        Otherwise e_up_lo = exp(v_up - v_lo), same as in Leyffer's paper.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return exp(m.v[i, n]) - m.v[i, 2 * n + 1] >= -v_up * (1 - m.y[(n, "log")])

    @m.Constraint(n_data_set, NnotT)
    def logc_rule(m, i, n):
        """
        If the logarithm operator is assigned to node n, then the argument of the square root should be positive.

        In Leyffer's paper the constaint is written as: eps - m.v[i, 2 * n + 1] <= (eps - v_lo) * (1 - m.y[(n, "log")])

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        i: natural number
           sample point index

        Returns
        -------

           value-defining inequality constraint
        """
        return m.v[i, 2 * n + 1] * m.y[n, "log"] >= m.y[n, "log"] - 1 + eps

    # Redundancy-eliminating constraints ################################################################
    def rec28a_rule(m, n):
        if (2 * n + 1, "-") in Y and (n, "+") in Y:
            return m.y[(n, "+")] + m.y[(2 * n + 1, "-")] <= 1
        else:
            return Constraint.Skip

    m.rec28a = Constraint(np.setdiff1d(N, Nperfect), rule=rec28a_rule)

    def rec28b_rule(m, n):
        if (2 * n + 1, "/") in Y and (n, "*") in Y:
            return m.y[(n, "*")] + m.y[(2 * n + 1, "/")] <= 1
        else:
            return Constraint.Skip

    m.rec28b = Constraint(np.setdiff1d(N, Nperfect), rule=rec28b_rule)

    @m.Constraint(NnotT)
    def rec28c_rule(m, n):
        """
        The right child of a node n can be a constant,
        only if the addition or multiplication operand is assigned to node n.
        (13c constraint in Leyffer's paper)

        Equivalently, if the subtraction or division or any unary operand (log, exp, sqrt)
        is assigned to node n, then the right child isn't constant.

        The Leyffer's constraint substitutes the constraints 2b, 3a and 3b of Alison's paper.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        Returns
        -------

           redundancy-eliminating inequality constraint
        """
        return m.y[(2 * n + 1, "cst")] <= m.y[(n, "+")] + m.y[(n, "*")]

    @m.Constraint(NnotT)
    def recA12d_rule(m, n):
        """
        For any two children nodes of a node n,
        at most one of them can have a constant assigned. (same as in Alison's paper)
        Equivalently, no binary operation can happen between two sibling constants.

        Parameters
        ----------
        m: -
           pyomo model

        n: natural number
           node index

        Returns
        -------

           redundancy-eliminating inequality constraint
        """
        return m.y[(2 * n, "cst")] + m.y[(2 * n + 1, "cst")] <= 1

    def recA12e_rule(m, n, o1, o2):
        if (2 * n + 1, o2) in Y:
            return m.y[(n, o1)] + m.y[(2 * n + 1, o2)] <= 1
        else:
            return Constraint.Skip

    m.recA12e = Constraint(NnotT, Opair, rule=recA12e_rule)

    def recA12f_rule(m, n, o1, o2):
        if (2 * n + 1, o1) in Y:
            return m.y[(n, o2)] + m.y[(2 * n + 1, o1)] <= 1
        else:
            return Constraint.Skip

    m.recA12f = Constraint(NnotT, Opair, rule=recA12f_rule)

    # Implication cuts ################################################################
    @m.Constraint(Xzero, NnotT)
    def ic12a_rule(m, j, n):
        return m.y[(n, "/")] + m.y[(2 * n + 1, "x_" + str(j))] <= 1

    @m.Constraint(Xnega, NnotT)
    def ic12b_rule(m, j, n):
        return m.y[(n, "**0.5")] + m.y[(2 * n + 1, "x_" + str(j))] <= 1

    @m.Constraint(Xnega + Xzero, NnotT)
    def ic12c_rule(m, j, n):
        return m.y[(n, "log")] + m.y[(2 * n + 1, "x_" + str(j))] <= 1

    # symmetry breaking constraints ################################################################
    @m.Constraint(Nperfect)
    def sbc14_rule(m, n):
        return m.v[1, 2 * n] - m.v[1, 2 * n + 1] >= (v_lo - v_up) * (
            1 - m.y[(n, "+")] - m.y[(n, "*")]
        )

    return m, Y, B, U, L, NnotT, T, N, c_lo, c_up


def XsetGenerator(X):  # list of positive, negative, and zero values of X
    Xposi = []
    Xnega = []
    Xzero = []
    for var, x in enumerate(X.T):
        if np.any(x > 0):
            Xposi.append(var + 1)
        if np.any(x < 0):
            Xnega.append(var + 1)
        if np.any(x == 0):
            Xzero.append(var + 1)
    return Xposi, Xnega, Xzero


def node_definer(depth):  # Index of the nodes
    Nodes = [1]
    TerminalNodes = []
    Old_Level_Nodes = [1]
    for level in range(1, depth + 1):
        New_Level_Nodes = []
        for node in Old_Level_Nodes:
            New_Level_Nodes.append(2 * node)
            New_Level_Nodes.append(2 * node + 1)
        Nodes += New_Level_Nodes
        if level != depth:
            Old_Level_Nodes = New_Level_Nodes
        else:
            TerminalNodes += New_Level_Nodes
    return Nodes, TerminalNodes


def Y_define(O, NnotT, L, T):  # set of allowed pair of (operator, node index)
    Y = []
    for o in O:
        for n in NnotT:
            Y.append((n, o))
    for l in L:
        for t in T:
            Y.append((t, l))
    return Y
