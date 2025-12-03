import numpy as np
import pandas as pd
import pyomo.environ as pyo
from symbolic_regression import SymbolicRegressionModel
import os

# from utils import get_gurobi

rng_samples_split = np.random.default_rng(42)

print()
print("--------Hyperparameters used--------")
with open("Hyperparameters/dataset_name.txt", "r") as file:
    dataset_name = file.read().strip()

with open("Hyperparameters/model_type.txt", "r") as file:
    model_type = file.read().strip()
print("Model type of the MINLP =", model_type)

with open("Hyperparameters/solver.txt", "r") as file:
    solver = file.read().strip()
print("Solver used to solve the MINLP =", solver)

with open("Hyperparameters/number_of_samples.txt", "r") as file:
    number_of_samples = int(file.read().strip())
print("Number of samples =", number_of_samples)

with open("Hyperparameters/fitness_metric.txt", "r") as file:
    fitness_metric = file.read().strip()
print("Fitness metric to minimize =", fitness_metric)

with open("Hyperparameters/max_depth.txt", "r") as file:
    depth = int(file.read().strip())
print("Maximum depth of tree =", depth)

experiment_name = f"{dataset_name}_{model_type}_{fitness_metric}_{solver}_maxdepth{depth}_samples{number_of_samples}"

df = pd.read_csv(f"Feynman_depth1trees/{dataset_name}", sep="\t")
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


test_idx = np.setdiff1d(all_idx, train_idx)  # indices of the test points
X_test_df = X_df.iloc[test_idx]
y_test_df = y_df.iloc[test_idx]
X_test_np = X_test_df.to_numpy()
y_test_np = y_test_df.to_numpy().reshape(-1, 1)

# Save train and test data along with indices from the original dataset
# The index displayed is the datafrmame's 0-based index, not the row number in the original tsv file
# row_number = pandas_df_index + 2 (to account for header and 0-based index)
os.makedirs("Train_data", exist_ok=True)
os.makedirs("Test_data", exist_ok=True)

train_with_data = pd.concat([X_train_df, y_train_df], axis=1)
train_with_data.to_csv(f"Train_data/train_data_{experiment_name}.csv", index=True)

test_with_data = pd.concat([X_test_df, y_test_df], axis=1)
test_with_data.to_csv(f"Test_data/test_data_{experiment_name}.csv", index=True)

print()
feature_names = X_df.columns.tolist()  # Preserves order
print("Dataset =", dataset_name)
print("Feature names:", feature_names)
print("Target name:", y_df.columns[0])

data = pd.concat([X_train_df, y_train_df], axis=1)
input_cols = list(X_train_df.columns)
output_col = y_train_df.columns[0]
print(data)


def build_model():
    m = SymbolicRegressionModel(
        data=data,
        input_columns=input_cols,
        output_column=output_col,
        tree_depth=depth,
        operators=["sum", "diff", "mult"],  # "div", "square", "sqrt"],
        var_bounds=(-100, 100),
        constant_bounds=(-100, 100),
        model_type=model_type,
    )
    m.add_objective()

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
        solver = pyo.SolverFactory("baron")
        solver.solve(mdl, tee=True)

    mdl.constant_val.pprint()
    print(mdl.get_parity_plot_data())
    print(mdl.get_selected_operators())
