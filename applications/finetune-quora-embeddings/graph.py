import matplotlib.pyplot as plt
import json
import math

SOURCE = "./model_perf.json"
BENCHMARK = "./output.json"
BENCHMARK_MODELS = ["text-embedding-3-small", "embed-multilingual-v3.0"]
vendors = {"text-embedding-3-small": "OpenAI", "embed-multilingual-v3.0": "Cohere"}
DATASET_SIZE = 261317
METRIC = "accuracy"
INVERSE = False
BASE_MODEL = "bge-base-en-v1.5"

with open(SOURCE, "r") as json_file:
    data = json.load(json_file)

with open(BENCHMARK, "r") as json_file:
    benchmarks = json.load(json_file)

benchmark_values = [
    (model, performance[METRIC] if not INVERSE else 1 - performance[METRIC])
    for model, performance in benchmarks.items()
    if model in BENCHMARK_MODELS
]
baseline_performance = (
    benchmarks[BASE_MODEL][METRIC]
    if not INVERSE
    else 1 - benchmarks[BASE_MODEL][METRIC]
)

data_points = [
    (
        float(dataset_pct) * DATASET_SIZE,
        data_point[METRIC] if not INVERSE else 1 - data_point[METRIC],
    )
    for dataset_pct, data_point in data.items()
]

# Sort by size
data_points.sort(key=lambda x: x[0])

X = [0] + [i[0] for i in data_points]
Y = [baseline_performance] + [i[1] for i in data_points]


plt.plot(X, Y, marker="o")
plt.xlabel("Number of Data Points")
# plt.xscale("log")
plt.ylabel(f"{METRIC.capitalize()} Score" if not INVERSE else f"{METRIC} error rate")
plt.title(
    f"{METRIC.capitalize()} Improvement with respect to size of dataset ( Epochs = 6 )"
    if not INVERSE
    else f"{METRIC.capitalize()} Error Rate against Dataset Size"
)
plt.subplots_adjust(top=0.90, bottom=0.15)

# plt.xticks(X, rotation="vertical")
plt.xticks(X[1:], [f"{math.floor(x_val)}" for x_val in X[1:]], minor=True, rotation=0)
plt.tick_params(axis="x", which="minor", length=20, width=2, color="lightblue")

plt.xlim(-10000, 250000)
# plt.show()
for idx, value in enumerate(benchmark_values):
    model, benchmark_value = value
    plt.axhline(y=benchmark_value, linestyle="dotted", color="grey")
    plt.text(
        X[4] - 38000,
        benchmark_value + 0.002,
        f"[{vendors[model]}] {model}: {round(benchmark_value,4)}",
        verticalalignment="center",
    )

ymin, ymax = plt.ylim()
for x, y in zip(X, Y):
    if x != 0:
        y_fraction = (y - ymin) / (ymax - ymin)
        plt.axvline(x=x, linestyle="dashed", ymax=y_fraction)

plt.savefig(
    f"graphs/{METRIC}_{'inverse_' if INVERSE else ''}graph.png", format="png", dpi=300
)
