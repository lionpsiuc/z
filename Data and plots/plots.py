import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

outdir = pathlib.Path("Figures/")
outdir.mkdir(exist_ok=True)

datafile = pathlib.Path("timing_data.csv")
full_data = pd.read_csv(datafile)

# print(full_data)

cpu_data = full_data.loc[full_data["skip_cpu"] == 0]
cpu_data = cpu_data.groupby(["height", "width"], as_index=False).median()
cpu_data = cpu_data[["height", "width", "cpu_iteration", "cpu_average"]]

data = full_data.loc[full_data["skip_cpu"] != 0]
data = data.groupby(
    ["block_size", "height", "width", "device_index"], as_index=False
).median()
data = data.drop("cpu_iteration", axis=1)
data = data.drop("cpu_average", axis=1)

data = data.merge(
    cpu_data,
    left_on=["height", "width"],
    right_on=["height", "width"],
    suffixes=(False, False),
)
data = data[
    [
        "block_size",
        "height",
        "width",
        "device_index",
        "cpu_iteration",
        "gpu_iteration",
        "gpu_iteration_comp",
        "cpu_average",
        "gpu_average",
        "gpu_average_comp",
    ]
]

data["iter_speedup_total"] = data.cpu_iteration / data.gpu_iteration
data["iter_speedup_comp"] = data.cpu_iteration / data.gpu_iteration_comp

data["avg_speedup_total"] = data.cpu_average / data.gpu_average
data["avg_speedup_comp"] = data.cpu_average / data.gpu_average_comp

# print(data)

turing_data = data.loc[(data.height == 15360) & (data.device_index == 0)]
kepler_data = data.loc[(data.height == 15360) & (data.device_index == 1)]

block_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 992]

for speedup in [
    "iter_speedup_total",
    "iter_speedup_comp",
    "avg_speedup_total",
    "avg_speedup_comp",
]:

    # print(turing_data[speedup])
    # print(kepler_data[speedup])
    # print(kepler_data['block_size'])

    fig, ax = plt.subplots()
    ax.semilogx(turing_data["block_size"], turing_data[speedup])
    ax.semilogx(kepler_data["block_size"], kepler_data[speedup])

    ax.set_xscale("log")
    ax.set_xticks(block_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(["2080 Super", "K40c"])

    outname = pathlib.Path(speedup + "_vs_block_size_plot.pdf")
    plt.savefig(outdir / outname, bbox_inches="tight", pad_inches=0)
    plt.figure().clear()
