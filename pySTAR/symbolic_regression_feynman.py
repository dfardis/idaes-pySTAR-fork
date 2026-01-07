import numpy as np
import pandas as pd
import pyomo.environ as pyo
from symbolic_regression import SymbolicRegressionModel
import os
import sympy as sp

# from utils import get_gurobi

rng_samples_split = np.random.default_rng(42)

print()
print("--------Hyperparameters used--------")
with open("dataset_name.txt", "r") as file:
    dataset_name = file.read().strip()

with open("model_type.txt", "r") as file:
    model_type = file.read().strip()
print("Model type of the MINLP =", model_type)

with open("solver.txt", "r") as file:
    solver = file.read().strip()
print("Solver used to solve the MINLP =", solver)

with open("number_of_samples.txt", "r") as file:
    number_of_samples = int(file.read().strip())
print("Number of samples =", number_of_samples)

with open("fitness_metric.txt", "r") as file:
    fitness_metric = file.read().strip()
print("Fitness metric to minimize =", fitness_metric)

with open("max_depth.txt", "r") as file:
    depth = int(file.read().strip())
print("Maximum depth of tree =", depth)

experiment_name = f"{dataset_name}_{model_type}_{fitness_metric}_{solver}_maxdepth{depth}_samples{number_of_samples}"

df = pd.read_csv(f"Feynman_maxdepth3/{dataset_name}", sep="\t")
X_df = df.iloc[:, :-1]
y_df = df.iloc[
    :, [-1]
]  # if you use df.iloc[:, -1] it will return a Series instead of a DataFrame

# Downsampling
all_idx = np.arange(len(df))  # array of indices of all data points

if len(df) > number_of_samples:
    train_idx = rng_samples_split.choice(len(df), number_of_samples, replace=False)
    X_train_df = X_df.iloc[train_idx]
    y_train_df = y_df.iloc[train_idx]
else:
    # Use all data for training if dataset is small
    train_idx = all_idx
    X_train_df = X_df
    y_train_df = y_df

test_idx = np.setdiff1d(all_idx, train_idx)  # indices of the test points
X_test_df = X_df.iloc[test_idx]
y_test_df = y_df.iloc[test_idx]
X_test_np = X_test_df.to_numpy()
y_test_np = y_test_df.to_numpy().reshape(-1, 1)

# Save train and test data along with indices from the original dataset
# The index displayed is the datafrmame's 0-based index, not the row number in the original tsv file
# row_number = pandas_df_index + 2 (to account for header and 0-based index)
os.makedirs("Results/Train_data", exist_ok=True)
os.makedirs("Results/Test_data", exist_ok=True)

train_with_data = pd.concat([X_train_df, y_train_df], axis=1)
train_with_data.to_csv(
    f"Results/Train_data/train_data_{experiment_name}.csv", index=True
)

test_with_data = pd.concat([X_test_df, y_test_df], axis=1)
test_with_data.to_csv(f"Results/Test_data/test_data_{experiment_name}.csv", index=True)

print()
feature_names = X_df.columns.tolist()  # Preserves order
print("Dataset =", dataset_name)
print("Feature names:", feature_names)
print("Target name:", y_df.columns[0])

data = pd.concat([X_train_df, y_train_df], axis=1)
input_cols = list(X_train_df.columns)
output_col = y_train_df.columns[0]
print(data)

# Compute variable bounds from training data
v_lo = min(-100, X_train_df.min().min(), y_train_df.min().min())
v_up = max(100, X_train_df.max().max(), y_train_df.max().max())
print(f"Variable bounds: v_lo = {v_lo}, v_up = {v_up}")


def build_model():
    m = SymbolicRegressionModel(
        data=data,
        input_columns=input_cols,
        output_column=output_col,
        tree_depth=depth,
        operators=["sum", "diff", "mult", "div", "square", "sqrt"],
        var_bounds=(v_lo, v_up),
        constant_bounds=(-100, 100),
        model_type=model_type,
    )
    m.add_objective(fitness_metric)

    return m


if __name__ == "__main__":
    solver = solver
    mdl = build_model()
    # mdl.add_tree_size_constraint(3)

    if solver == "scip":
        solver = pyo.SolverFactory("scip")
        solver.solve(mdl, tee=True)

    elif solver == "gurobi":
        solver = pyo.SolverFactory("gams")
        solver.solve(mdl, solver="gurobi", tee=True)

    elif solver == "baron":
        os.makedirs("Results/baron_model_MINLP", exist_ok=True)
        mdl.write(f"Results/baron_model_MINLP/{experiment_name}.bar", format="bar")

        os.makedirs("Results/baron_res_MINLP", exist_ok=True)
        os.makedirs("Results/baron_log_MINLP", exist_ok=True)
        os.makedirs("Results/baron_summaries_MINLP", exist_ok=True)
        os.makedirs("Results/baron_timefiles_MINLP", exist_ok=True)

        solver = pyo.SolverFactory("baron")
        solver.options["MaxTime"] = 3600
        solver.options["CplexLibName"] = r"C:\GAMS\48\cplex2211.dll"
        # pyomo deactivates the creation of summary file, even though baron creates it by default
        solver.options["summary"] = 1
        solver.options["SumName"] = (
            f"Results/baron_summaries_MINLP/summary_{experiment_name}"
        )

        results = solver.solve(
            mdl,
            tee=True,
            symbolic_solver_labels=True,
            keepfiles=True,
            solnfile=f"Results/baron_res_MINLP/res_{experiment_name}",
            logfile=f"Results/baron_log_MINLP/log_{experiment_name}.log",
        )

        # Copy the tim file from temp directory to desired location
        import glob
        import shutil

        temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
        tim_pattern = os.path.join(temp_dir, "tmp*.baron.tim")
        tim_files = glob.glob(tim_pattern)
        if tim_files:
            # Get the most recently modified tim file
            latest_tim = max(tim_files, key=os.path.getmtime)
            dest_tim = f"Results/baron_timefiles_MINLP/tim_{experiment_name}"
            shutil.copy2(latest_tim, dest_tim)
            print(f"Copied tim file to {dest_tim}")

    mdl.constant_val.pprint()
    print(mdl.get_parity_plot_data())
    print("Assigned operators to nodes: ", mdl.get_selected_operators())

    parity_data = mdl.get_parity_plot_data()

    # Append predictions to train data CSV
    train_data_path = f"Results/Train_data/train_data_{experiment_name}.csv"
    train_df = pd.read_csv(train_data_path, index_col=0)
    train_df["prediction"] = parity_data["prediction"].values
    train_df["square_of_error"] = parity_data["square_of_error"].values
    train_df.to_csv(train_data_path, index=True)

    expr = mdl.selected_tree_to_expression()
    expr = expr.sympy_expression
    print("SR model:", expr)

    selected_operators = mdl.get_selected_operators()

    # Extract constant values from the model
    constant_values = [pyo.value(mdl.constant_val[i]) for i in mdl.constant_val]
    print("Constant values:", constant_values)

    # Substitute constant values into the expression
    constant_substitutions = {}
    for i in mdl.constant_val:
        constant_substitutions[sp.Symbol(f"c_{i}")] = pyo.value(mdl.constant_val[i])

    SR_model = expr.subs(constant_substitutions)
    print("SR model with constant values substituted:", SR_model)

    # Get constants that actually appear in the expression
    expr_symbols = expr.free_symbols
    constants_in_expr = []
    for i in mdl.constant_val:
        c_symbol = sp.Symbol(f"c_{i}")
        if c_symbol in expr_symbols:
            constants_in_expr.append((i, pyo.value(mdl.constant_val[i])))

    # Format constant values as a list string [c1: value, c2: value, ...] (only constants in expression)
    constant_values_str = (
        "[" + ", ".join([f"c{i}: {val}" for i, val in constants_in_expr]) + "]"
    )

    # Save to SR_model.csv (append mode)
    os.makedirs("Results", exist_ok=True)
    sr_model_csv_path = "Results/SR_model.csv"

    # Create new row with experiment data
    new_row = pd.DataFrame(
        {
            "experiment_name": [experiment_name],
            "expression": [str(expr)],
            "SR_model": [str(SR_model)],
            "constant_values": [constant_values_str],
            "nodes_assignments": [str(selected_operators)],
        }
    )

    # Append to CSV (create if doesn't exist, overwrite if experiment_name exists)
    if os.path.exists(sr_model_csv_path):
        existing_df = pd.read_csv(sr_model_csv_path)
        # Remove existing row with same experiment_name if it exists
        existing_df = existing_df[existing_df["experiment_name"] != experiment_name]
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        updated_df = new_row

    updated_df.to_csv(sr_model_csv_path, index=False)

    feature_vars = [sp.Symbol(f"x{i}") for i in range(1, len(X_test_df.columns) + 1)]
    SR_model_func = sp.lambdify(feature_vars, SR_model, modules="numpy")

    # Make predictions on the test data
    test_predictions = np.array([SR_model_func(*row) for row in X_test_df.to_numpy()])
    test_actuals = y_test_df.values.flatten()

    # Append predictions to test data CSV
    test_data_path = f"Results/Test_data/test_data_{experiment_name}.csv"
    test_df = pd.read_csv(test_data_path, index_col=0)
    test_df["prediction"] = test_predictions
    test_df["square_of_error"] = (test_actuals - test_predictions) ** 2
    test_df.to_csv(test_data_path, index=True)
