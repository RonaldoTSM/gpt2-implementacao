# plot_loss.py - desenha a curva de loss do log.txt gerado pelo train.py
#   python plot_loss.py

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = {"train": [], "val": []}
losses = {"train": [], "val": []}
with open("log.txt") as f:
    for line in f:
        step, kind, loss = line.split()
        steps[kind].append(int(step))
        losses[kind].append(float(loss))

plt.figure(figsize=(7, 4))
plt.plot(steps["train"], losses["train"], label="treino")
plt.plot(steps["val"], losses["val"], "o-", label="validação")
# loss de um chute uniforme entre os 50257 tokens
plt.axhline(math.log(50257), color="gray", linestyle="--", label="ln(50257) = 10,82")
plt.xlabel("passo")
plt.ylabel("loss")
plt.title("GPT tiny no Tiny Shakespeare")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss.png", dpi=150)
print("gráfico salvo em loss.png")
