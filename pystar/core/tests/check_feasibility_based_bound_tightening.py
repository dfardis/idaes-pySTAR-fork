import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pystar.core.symbolic_regression import SymbolicRegressionModel

num_samples = 1
tree_depth = 2
operators = ["mult", "div", "sqrt", "log", "square", "exp"] #, "square", "sum", "diff", "exp", "mult", "div", "sqrt", "log"]

# Generate signed data for: y = x1*x2
rng = np.random.default_rng(42)
data = pd.DataFrame(
    {
        "x1": rng.uniform(1.0, 2.0, num_samples),
        "x2": rng.uniform(1.0, 2.0, num_samples),
    }
)

data["y"] = data["x1"] * data["x2"]
#print(data)

input_columns = ["x1", "x2"]

m = SymbolicRegressionModel(
    data=data,
    input_columns=input_columns,
    output_column="y",
    tree_depth=tree_depth,
    operators=operators,
    var_bounds=(-10, 10),
    constant_bounds=(-10, 10),
    model_type="hull",
)

m.add_objective("sse")
#m.relax_nonconvex_constraints()

for s in m.samples:
    for n in m.non_terminal_nodes_set:
        for op in m.operators_set:
            blk = getattr(m.samples[s].node[n], f"{op}_operator")
            print(f"\n{blk.name}")

            for v in blk.component_data_objects(
                pyo.Var, descend_into=False
            ):
                fixed = f", fixed at {v.value}" if v.fixed else ""
                print(f"  {v.local_name}: {v.bounds}{fixed}")
