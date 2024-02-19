import matplotlib.pyplot as plt
import json

baseline = {
    "accuracy": 0.8337851461857287,
    "precision": 0.592130818870998,
    "recall": 0.9122586872586873,
    "AUC": 0.8611621663864554,
}

original_performance = {
    "accuracy": 0.8311414809006386,
    "precision": 0.5887010620247596,
    "recall": 0.9042471042471042,
    "AUC": 0.8566457986589276,
}

# Initialize an empty list to store the data entries
data = [
    {
        "accuracy": 0.8534110003360591,
        "precision": 0.6358073009321853,
        "recall": 0.8624517374517374,
        "AUC": 0.856565037216022,
        "model": "BAAI/bge-base-en-v1.5",
        "dataset_size": 1000,
        "epochs": 6,
        "warmup_steps": 500,
        "scheduler": "warmupcosine",
        "batch_size": 64,
        "loss": [
            0.786621867396219,
            0.7872971821573767,
            0.7884130523672186,
            0.7900412655915894,
            0.7917944440127781,
            0.7939522746116873,
        ],
    },
    {
        "accuracy": 0.8606698778985101,
        "precision": 0.6471675314521288,
        "recall": 0.8788610038610039,
        "AUC": 0.8670162057962933,
        "model": "BAAI/bge-base-en-v1.5",
        "dataset_size": 2000,
        "epochs": 6,
        "warmup_steps": 500,
        "scheduler": "warmupcosine",
        "batch_size": 64,
        "loss": [
            0.7873146839028387,
            0.7901949982729476,
            0.7947561069240852,
            0.7996619526215534,
            0.8042983311953955,
            0.8074228142497679,
        ],
    },
    {
        "accuracy": 0.8689817407863784,
        "precision": 0.6643356643356644,
        "recall": 0.8803088803088803,
        "AUC": 0.8729334335898887,
        "model": "BAAI/bge-base-en-v1.5",
        "dataset_size": 4000,
        "epochs": 6,
        "warmup_steps": 500,
        "scheduler": "warmupcosine",
        "batch_size": 64,
        "loss": [
            0.7903197146907419,
            0.7999941228386948,
            0.8096067455629543,
            0.8145869930614067,
            0.8169794554574514,
            0.8153368349249744,
        ],
    },
    {
        "accuracy": 0.8765990814383331,
        "precision": 0.6825707405177603,
        "recall": 0.8754826254826255,
        "AUC": 0.8762095840761048,
        "model": "BAAI/bge-base-en-v1.5",
        "dataset_size": 8000,
        "epochs": 6,
        "warmup_steps": 500,
        "scheduler": "warmupcosine",
        "batch_size": 64,
        "loss": [
            0.7995496614305293,
            0.8161136835788527,
            0.8199591566340426,
            0.8237846756261787,
            0.8205980826591113,
            0.8205005653419446,
        ],
    },
    {
        "accuracy": 0.8808334266830963,
        "precision": 0.6961020773360305,
        "recall": 0.8636100386100386,
        "AUC": 0.8748247129592863,
        "model": "BAAI/bge-base-en-v1.5",
        "dataset_size": 16000,
        "epochs": 6,
        "warmup_steps": 500,
        "scheduler": "warmupcosine",
        "batch_size": 64,
        "loss": [
            0.8170246518823161,
            0.8263173567661242,
            0.8277094062084467,
            0.8263304392574006,
            0.8269694628304216,
            0.8265691784767804,
        ],
    },
    {
        "accuracy": 0.8865240282289683,
        "precision": 0.7131814155729125,
        "recall": 0.8549227799227799,
        "AUC": 0.8754993184807188,
        "model": "BAAI/bge-base-en-v1.5",
        "dataset_size": 32000,
        "epochs": 6,
        "warmup_steps": 500,
        "scheduler": "warmupcosine",
        "batch_size": 64,
        "loss": [
            0.8289369361547437,
            0.8309696091067318,
            0.8334979335164595,
            0.8297416940865507,
            0.8296396127399528,
            0.8295118543727288,
        ],
    },
]


# Open the JSONL file and read line by line

# Sort data by dataset size
data.sort(key=lambda x: x["dataset_size"])

# Extract values
metric = "precision"
dataset_sizes = [d["dataset_size"] for d in data]
performance = [1 - d[metric] for d in data]
# precisions = [d["precision"] for d in data]
# recalls = [d["recall"] for d in data]
# aucs = [d["AUC"] for d in data]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, performance, marker="o", label="Accuracy")
# plt.plot(dataset_sizes, precisions, marker="o", label="Precision")
# plt.plot(dataset_sizes, recalls, marker="o", label="Recall")
# plt.plot(dataset_sizes, aucs, marker="o", label="AUC")

# Set the x-axis to logarithmic scale
plt.xscale("log")

# Now set the ticks and labels manually
plt.xticks(dataset_sizes, labels=dataset_sizes)

plt.axhline(y=1 - baseline[metric], color="r", linestyle="-", label="OpenAI")
plt.text(
    x=max(dataset_sizes),
    y=1 - baseline[metric] - 0.003,
    s="text-embedding-3-small ( OpenAI  ) ",
    va="center",
    ha="right",
    color="r",
)

plt.xlabel("Dataset Size")
plt.ylabel(f"{metric.capitalize()} Error Rate")
plt.title(f"{metric.capitalize()} Error Rate vs Dataset Size")
plt.show()
