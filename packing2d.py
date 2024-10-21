"""
2D Bin Packing problem
Simplification of: https://github.com/dwave-examples/3d-bin-packing
"""

import os
import argparse
import dimod
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet
from dwave.system import LeapHybridCQMSampler
from itertools import combinations
from tabulate import tabulate
import numpy as np
from typing import Tuple
import plotly.colors as plcolor
import plotly.graph_objects as go
import pickle


class Cases:
    """Class for representing rectangular item data in a 2D bin packing problem.

    Args:
         data: dictionary containing raw information for both bins and cases

    """

    def __init__(self, data):
        self.case_ids = np.repeat(data["case_ids"], data["quantity"])
        self.num_cases = np.sum(data["quantity"], dtype=np.int32)
        self.width = np.repeat(data["case_width"], data["quantity"])
        self.height = np.repeat(data["case_height"], data["quantity"])
        print(f"Number of cases: {self.num_cases}")


class Bins:
    """Class for representing rectangular container data in a 2D bin packing problem.

    Args:
        data: dictionary containing raw information for both bins and cases
        cases: Instance of ``Cases``, representing rectangular items packed into containers.

    """

    def __init__(self, data, cases):
        self.width = data["bin_dimensions"][0]
        self.height = data["bin_dimensions"][1]
        self.num_bins = data["num_bins"]
        self.lowest_num_bin = int(
            np.ceil(np.sum(cases.width * cases.height) / (self.width * self.height))
        )
        if self.lowest_num_bin > self.num_bins:
            raise RuntimeError(
                f"number of bins is at least {self.lowest_num_bin}, "
                + "try increasing the number of bins"
            )
        print(f"Minimum Number of bins required: {self.lowest_num_bin}")


class Variables:
    """Class that collects all CQM model variables for the 2D bin packing problem.

    Args:
        cases: Instance of ``Cases``, representing rectangular items packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.

    """

    def __init__(self, cases: Cases, bins: Bins):
        num_cases = cases.num_cases
        num_bins = bins.num_bins
        lowest_num_bin = bins.lowest_num_bin
        self.x = {
            i: Real(f"x_{i}", lower_bound=0, upper_bound=bins.width * bins.num_bins)
            for i in range(num_cases)
        }
        self.y = {
            i: Real(f"y_{i}", lower_bound=0, upper_bound=bins.height)
            for i in range(num_cases)
        }

        self.bin_height = {
            j: Real(label=f"upper_bound_{j}", upper_bound=bins.height)
            for j in range(num_bins)
        }

        # the first case always goes to the first bin
        self.bin_loc = {
            (i, j): Binary(f"case_{i}_in_bin_{j}") if num_bins > 1 else 1
            for i in range(1, num_cases)
            for j in range(num_bins)
        }

        self.bin_loc.update({(0, j): int(j == 0) for j in range(num_bins)})

        self.bin_on = {
            j: 1 if j < lowest_num_bin else Binary(f"bin_{j}_is_used")
            for j in range(num_bins)
        }

        self.o = {
            (i, k): Binary(f"o_{i}_{k}") for i in range(num_cases) for k in range(2)
        }

        self.selector = {
            (i, j, k): Binary(f"sel_{i}_{j}_{k}")
            for i, j in combinations(range(num_cases), r=2)
            for k in range(4)
        }


def add_bin_on_constraint(
    cqm: ConstrainedQuadraticModel, vars: Variables, bins: Bins, cases: Cases
):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    lowest_num_bin = bins.lowest_num_bin
    if num_bins > 1:
        for j in range(lowest_num_bin, num_bins):
            cqm.add_constraint(
                (1 - vars.bin_on[j])
                * quicksum(vars.bin_loc[i, j] for i in range(num_cases))
                <= 0,
                label=f"bin_on_{j}",
            )

        for j in range(lowest_num_bin, num_bins - 1):
            cqm.add_constraint(
                vars.bin_on[j + 1] - vars.bin_on[j] <= 0, label=f"bin_use_order_{j}"
            )


def add_orientation_constraints(
    cqm: ConstrainedQuadraticModel, vars: Variables, cases: Cases
) -> list:
    num_cases = cases.num_cases
    dx = {}
    dy = {}
    for i in range(num_cases):
        orientations = [
            [cases.width[i], cases.height[i]],
            [cases.height[i], cases.width[i]],
        ]
        dx[i] = 0
        dy[i] = 0
        for j, (a, b) in enumerate(orientations):
            dx[i] += a * vars.o[i, j]
            dy[i] += b * vars.o[i, j]

    for i in range(num_cases):
        cqm.add_discrete(
            quicksum([vars.o[i, k] for k in range(2)]), label=f"orientation_{i}"
        )
    return [dx, dy]


def add_geometric_constraints(
    cqm: ConstrainedQuadraticModel,
    vars: Variables,
    bins: Bins,
    cases: Cases,
    effective_dimensions: list,
):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    dx, dy = effective_dimensions
    # adding discrete constraints first
    if num_bins > 1:
        for i in range(1, num_cases):
            cqm.add_discrete(
                quicksum([vars.bin_loc[i, j] for j in range(num_bins)]),
                label=f"case_{i}_max_packed",
            )
    for i, k in combinations(range(num_cases), r=2):
        cqm.add_discrete(
            quicksum([vars.selector[i, k, s] for s in range(4)]),
            label=f"discrete_{i}_{k}",
        )
    for i, k in combinations(range(num_cases), r=2):
        for j in range(num_bins):
            cases_on_same_bin = vars.bin_loc[i, j] * vars.bin_loc[k, j]
            cqm.add_constraint(
                -(2 - cases_on_same_bin - vars.selector[i, k, 0])
                * num_bins
                * bins.width
                + (vars.x[i] + dx[i] - vars.x[k])
                <= 0,
                label=f"overlap_{i}_{k}_{j}_0",
            )

            cqm.add_constraint(
                -(2 - cases_on_same_bin - vars.selector[i, k, 1]) * bins.height
                + (vars.y[i] + dy[i] - vars.y[k])
                <= 0,
                label=f"overlap_{i}_{k}_{j}_1",
            )

            cqm.add_constraint(
                -(2 - cases_on_same_bin - vars.selector[i, k, 2])
                * num_bins
                * bins.width
                + (vars.x[k] + dx[k] - vars.x[i])
                <= 0,
                label=f"overlap_{i}_{k}_{j}_2",
            )

            cqm.add_constraint(
                -(2 - cases_on_same_bin - vars.selector[i, k, 3]) * bins.height
                + (vars.y[k] + dy[k] - vars.y[i])
                <= 0,
                label=f"overlap_{i}_{k}_{j}_3",
            )


def add_boundary_constraints(
    cqm: ConstrainedQuadraticModel,
    vars: Variables,
    bins: Bins,
    cases: Cases,
    effective_dimensions: list,
):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    dx, dy = effective_dimensions
    for i in range(num_cases):
        cqm.add_constraint(
            vars.y[i] + dy[i] <= bins.height,
            label=f'maxy_{i}_less')
        for j in range(num_bins):
            cqm.add_constraint(
                vars.y[i]
                + dy[i]
                - vars.bin_height[j]
                - (1 - vars.bin_loc[i, j]) * bins.height
                <= 0,
                label=f"maxx_height_{i}_{j}",
            )

            cqm.add_constraint(
                vars.x[i]
                + dx[i]
                - bins.width * (j + 1)
                - (1 - vars.bin_loc[i, j]) * num_bins * bins.width
                <= 0,
                label=f"maxx_{i}_{j}_less",
            )
            if j > 0:
                cqm.add_constraint(
                    j * bins.width * vars.bin_loc[i, j] - vars.x[i] <= 0,
                    label=f"maxx_{i}_{j}_greater",
                )


def define_objective(
    cqm: ConstrainedQuadraticModel,
    vars: Variables,
    bins: Bins,
    cases: Cases,
    effective_dimensions: list,
):
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    lowest_num_bin = bins.lowest_num_bin
    _, dy = effective_dimensions

    # First term of objective: minimize average height of cases
    first_obj_term = quicksum(vars.y[i] + dy[i] for i in range(num_cases)) / num_cases

    # Second term of objective: minimize height of the case at the top of the
    # bin
    second_obj_term = quicksum(vars.bin_height[j] for j in range(num_bins))

    # Third term of the objective: minimize the number of used bins
    third_obj_term = quicksum(vars.bin_on[j] for j in range(lowest_num_bin, num_bins))
    first_obj_coefficient = 1
    second_obj_coefficient = 1
    third_obj_coefficient = bins.height
    cqm.set_objective(
        first_obj_coefficient * first_obj_term
        + second_obj_coefficient * second_obj_term
        + third_obj_coefficient * third_obj_term
    )


def build_cqm(
    vars: Variables, bins: Bins, cases: Cases
) -> Tuple[ConstrainedQuadraticModel, list]:
    """Builds the CQM model from the problem variables and data.

    Args:
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 2D bin packing problem.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        cases: Instance of ``Cases``, representing rectangular items packed into containers.

    Returns:
        A ``dimod.CQM`` object that defines the 2D bin packing problem.
        effective_dimensions: List of case dimensions based on orientations of cases.

    """
    cqm = ConstrainedQuadraticModel()
    effective_dimensions = add_orientation_constraints(cqm, vars, cases)
    add_geometric_constraints(cqm, vars, bins, cases, effective_dimensions)
    add_bin_on_constraint(cqm, vars, bins, cases)
    add_boundary_constraints(cqm, vars, bins, cases, effective_dimensions)
    define_objective(cqm, vars, bins, cases, effective_dimensions)

    return cqm, effective_dimensions


def call_solver(
    cqm: ConstrainedQuadraticModel, time_limit: float
) -> SampleSet:
    """Helper function to call the CQM Solver.

    Args:
        cqm: A ``CQM`` object that defines the 2D bin packing problem.
        time_limit: Time limit parameter to pass on to the CQM sampler.

    Returns:
        A ``dimod.SampleSet`` that represents the best feasible solution found.

    """
    sampler = LeapHybridCQMSampler()
    res = sampler.sample_cqm(cqm, time_limit=time_limit, label="2d bin packing")

    res.resolve()
    feasible_sampleset = res.filter(lambda d: d.is_feasible)
    print(feasible_sampleset)
    try:
        best_feasible = feasible_sampleset.first.sample

        return best_feasible

    except ValueError:
        raise RuntimeError(
            "Sampleset is empty, try increasing time limit or "
            + "adjusting problem config."
        )


def print_cqm_stats(cqm: dimod.ConstrainedQuadraticModel) -> None:
    """Print some information about the CQM model defining the 2D bin packing problem.

    Args:
        cqm: A dimod constrained quadratic model.

    """
    if not isinstance(cqm, dimod.ConstrainedQuadraticModel):
        raise ValueError("input instance should be a dimod CQM model")
    num_binaries = sum(cqm.vartype(v) is dimod.BINARY for v in cqm.variables)
    num_integers = sum(cqm.vartype(v) is dimod.INTEGER for v in cqm.variables)
    num_continuous = sum(cqm.vartype(v) is dimod.REAL for v in cqm.variables)
    num_discretes = len(cqm.discrete)
    num_linear_constraints = sum(
        constraint.lhs.is_linear() for constraint in cqm.constraints.values()
    )
    num_quadratic_constraints = sum(
        not constraint.lhs.is_linear() for constraint in cqm.constraints.values()
    )
    num_le_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Le
        for constraint in cqm.constraints.values()
    )
    num_ge_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Ge
        for constraint in cqm.constraints.values()
    )
    num_equality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Eq
        for constraint in cqm.constraints.values()
    )

    assert num_binaries + num_integers + num_continuous == len(cqm.variables)

    assert num_quadratic_constraints + num_linear_constraints == len(cqm.constraints)

    print(" \n" + "=" * 35 + "MODEL INFORMATION" + "=" * 35)
    print(" " * 10 + "Variables" + " " * 20 + "Constraints" + " " * 15 + "Sensitivity")
    print("-" * 30 + " " + "-" * 28 + " " + "-" * 18)
    print(
        tabulate(
            [
                [
                    "Binary",
                    "Integer",
                    "Continuous",
                    "Quad",
                    "Linear",
                    "One-hot",
                    "EQ  ",
                    "LT",
                    "GT",
                ],
                [
                    num_binaries,
                    num_integers,
                    num_continuous,
                    num_quadratic_constraints,
                    num_linear_constraints,
                    num_discretes,
                    num_equality_constraints,
                    num_le_inequality_constraints,
                    num_ge_inequality_constraints,
                ],
            ],
            headers="firstrow",
        )
    )


def read_instance(instance_path: str) -> dict:
    """Convert instance input files into raw problem data.

    Args:
        instance_path:  Path to the bin packing problem instance file.

    Returns:
        data: dictionary containing raw information for both bins and cases.

    """

    data = {
        "num_bins": 0,
        "bin_dimensions": [],
        "quantity": [],
        "case_ids": [],
        "case_width": [],
        "case_height": [],
    }

    with open(instance_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                data["num_bins"] = int(line.split()[-1])
            elif i == 1:
                data["bin_dimensions"] = [int(i) for i in line.split()[-2:]]
            elif 2 <= i <= 4:
                continue
            else:
                case_info = list(map(int, line.split()))
                data["case_ids"].append(case_info[0])
                data["quantity"].append(case_info[1])
                data["case_width"].append(case_info[2])
                data["case_height"].append(case_info[3])

        return data


def write_solution_to_file(
    solution_file_path: str,
    cqm: dimod.ConstrainedQuadraticModel,
    vars: "Variables",
    sample: dimod.SampleSet,
    cases: "Cases",
    bins: "Bins",
    effective_dimensions: list,
):
    """Write solution to a file.

    Args:
        solution_file_path: path to the output solution file. If doesn't exist,
            a new file is created.
        cqm: A ``dimod.CQM`` object that defines the 2D bin packing problem.
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 2D bin packing problem.
        sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
        cases: Instance of ``Cases``, representing cases packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        effective_dimensions: List of case dimensions based on orientations of cases.

    """
    num_cases = cases.num_cases
    num_bins = bins.num_bins
    lowest_num_bin = bins.lowest_num_bin
    dx, dy = effective_dimensions
    if num_bins > 1:
        num_bin_used = lowest_num_bin + sum(
            [vars.bin_on[j].energy(sample) for j in range(lowest_num_bin, num_bins)]
        )
    else:
        num_bin_used = 1

    objective_value = cqm.objective.energy(sample)
    vs = [["case_id", "bin-location", "orientation", "x", "y", "x'", "y'"]]
    for i in range(num_cases):
        vs.append(
            [
                cases.case_ids[i],
                int(
                    sum(
                        (
                            int(j == 0)
                            if i == 0 or num_bins == 1
                            else (j + 1) * vars.bin_loc[i, j].energy(sample)
                        )
                        for j in range(num_bins)
                    )
                ),
                int(sum((r + 1) * vars.o[i, r].energy(sample) for r in range(2))),
                np.round(vars.x[i].energy(sample), 2),
                np.round(vars.y[i].energy(sample), 2),
                np.round(dx[i].energy(sample), 2),
                np.round(dy[i].energy(sample), 2),
            ]
        )

    with open(solution_file_path, "w") as f:
        f.write("# Number of bins used: " + str(int(num_bin_used)) + "\n")
        f.write("# Number of cases packed: " + str(int(num_cases)) + "\n")
        f.write("# Objective value: " + str(np.round(objective_value, 3)) + "\n\n")
        f.write(tabulate(vs, headers="firstrow"))
        f.close()
        print(f"Saved solution to " f"{os.path.join(os.getcwd(), solution_file_path)}")


def plot_rects(
    sample: dimod.SampleSet,
    vars: "Variables",
    cases: "Cases",
    bins: "Bins",
    effective_dimensions: list,
) -> go.Figure:
    """Visualization utility tool to view 2D bin packing solution.

    Args:
        sample: A ``dimod.SampleSet`` that represents the best feasible solution found.
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 2D bin packing problem.
        cases: Instance of ``Cases``, representing rectangular items packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        effective_dimensions: List of case dimensions based on orientations of cases.

    Returns:
        ``plotly.graph_objects.Figure`` with all cases packed according to CQM results.

    """
    dx, dy = effective_dimensions
    num_cases = cases.num_cases
    case_ids = cases.case_ids
    num_unique_ids = len(set(cases.case_ids))
    num_bins = bins.num_bins
    positions = []
    sizes = []
    colors = []
    quantities = [0] * num_bins

    # Get global coordinates for each case from sample
    for i in range(num_cases):
        positions.append((vars.x[i].energy(sample), vars.y[i].energy(sample)))
        sizes.append((dx[i].energy(sample), dy[i].energy(sample)))

    # Color by case ID
    for i in range(num_unique_ids):
        N = len(plcolor.qualitative.Plotly)
        colors.append(plcolor.qualitative.Plotly[i % N])

    # Populations of each bin
    for i in range(num_cases):
        for j in range(num_bins):
            if vars.bin_loc[i, j]:
                quantities[j] += 1

    fig = go.Figure()

    # Set axes properties
    fig.update_xaxes(range=[-5, num_bins * bins.width * 1.1], showgrid=False)
    fig.update_yaxes(range=[-5, bins.height * 1.1])

    # Draw bins
    for i in range(num_bins):
        fig.add_shape(
            type="rect",
            x0=bins.width * i,
            y0=0,
            x1=bins.width * (i + 1),
            y1=bins.height,
            line=dict(
                color="black",
                width=3,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[bins.width * (i + 0.5)],
                y=[bins.height * -0.05],
                text="Bin " + str(i + 1),
                hovertext=["Quantity: " + str(quantities[j])],
                hoverinfo="text",
                mode="text",
                textfont=dict(color="black"),
            )
        )

    # Draw cases
    for i in range(num_cases):
        for j in range(num_bins):
            if vars.bin_loc[i, j]:
                fig.add_shape(
                    type="rect",
                    x0=positions[i][0],
                    y0=positions[i][1],
                    x1=positions[i][0] + sizes[i][0],
                    y1=positions[i][1] + sizes[i][1],
                    line=dict(
                        color="RoyalBlue",
                        width=2,
                    ),
                    fillcolor=colors[case_ids[i]], # Set fill color by case ID
                )
                fig.add_trace(
                    go.Scatter(
                        x=[positions[i][0] + 0.5 * sizes[i][0]],
                        y=[positions[i][1] + 0.5 * sizes[i][1]],
                        text=[
                            f"Case ID: {case_ids[i]}<br>X: {np.round(positions[i][0], 2)}, Y: {np.round(positions[i][1], 2)}<br>Size: {sizes[i][0]} x {sizes[i][1]}"
                        ],
                        mode="markers",
                        hoverinfo="text",
                        marker=dict(opacity=0),
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=16,
                            font_color=colors[case_ids[i]],  # Set font color by case ID
                            bordercolor="gray",
                        ),
                    )
                )

    fig.update_layout(showlegend=False)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_filepath",
        type=str,
        nargs="?",
        help="Filename with path to bin-packing data file.",
        default="test_data.txt",
    )

    parser.add_argument(
        "--output_filepath",
        type=str,
        nargs="?",
        help="Path for the output solution file.",
        default=None,
    )

    parser.add_argument(
        "--time_limit",
        type=float,
        nargs="?",
        help="Time limit for the hybrid CQM Solver to run in" " seconds.",
        default=20,
    )

    parser.add_argument(
        "--html_filepath",
        type=str,
        nargs="?",
        help="Filename with path to plot html file.",
        default=None,
    )

    parser.add_argument(
        "--pickled",
        type=bool,
        nargs="?",
        help="Whether or not to use a serialized past run.",
        default=False,
    )

    args = parser.parse_args()
    output_filepath = args.output_filepath
    time_limit = args.time_limit
    html_filepath = args.html_filepath
    pickled = args.pickled

    data = read_instance(args.data_filepath)
    cases = Cases(data)
    bins = Bins(data, cases)
    vars = Variables(cases, bins)

    cqm, effective_dimensions = build_cqm(vars, bins, cases)

    # Write the CQM to a file
    cqm_file = cqm.to_file()
    with open("cqm.pickle", "wb") as outfile:
        pickle.dump(cqm_file, outfile)
        outfile.close()
    cqm_file.close()

    print_cqm_stats(cqm)

    best_feasible = None

    if not pickled:
        best_feasible = call_solver(cqm, time_limit)

        # Serialization
        with open("solution.pickle", "wb") as outfile:
            pickle.dump(best_feasible, outfile)
            outfile.close()
    else:
        # Deserialization
        with open("solution.pickle", "rb") as infile:
            best_feasible = pickle.load(infile)
            infile.close()

    if best_feasible is not None:
        if pickled:
            print("Reconstruction success")

        if output_filepath is not None:
            write_solution_to_file(
                output_filepath,
                cqm,
                vars,
                best_feasible,
                cases,
                bins,
                effective_dimensions,
            )

        fig = plot_rects(
            best_feasible, vars, cases, bins, effective_dimensions
        )

        if html_filepath is not None:
            fig.write_html(html_filepath + ".html")

        fig.show()
