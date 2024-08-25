import pandas as pd
mlr_data = pd.read_csv("")

"""
This part of code is adapted from https://matplotlib.org/stable/plot_types/3D/scatter3d_simple.html#sphx-glr-plot-types-3d-scatter3d-simple-py
and helped by other examples from the official website of matplotlib: https://matplotlib.org/stable/
"""
import matplotlib.pyplot as plt
import numpy as np
def plot(treatment_1, treatment_2,analyte):
    data = mlr_data.loc[((mlr_data["treatment_1"]==treatment_1) & (mlr_data["treatment_2"]==treatment_2)) & (mlr_data["analyte"]==analyte), ["concentration_1","concentration_2", "analyte_value"]]
    xs = data["concentration_1"]
    ys = data["concentration_2"]
    zs = data["analyte_value"]

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(xs, ys, zs)

    # ax.set(xticklabels=[],
    #         yticklabels=[],
    #         zticklabels=[])
    ax.set_xlabel(treatment_1 + " concentration")
    ax.set_ylabel(treatment_2 + " concentration")
    ax.set_zlabel(analyte + " value")
    ax.set_axis_on()
    plt.show()

plot("GEN1056","GEN1053","IL-10")
