import matplotlib.pyplot as plt
import numpy as np


log_fp = "log/log-20220713-0.9split.txt"
with open(log_fp) as f:
    log = f.readlines()

loss_vals = []
for ln in log:
    pat = "The train loss is "
    if pat in ln:
        loss_val = float(ln[ln.index(pat) + len(pat):ln.rindex('.')])
        loss_vals.append(loss_val)

plt.scatter(np.arange(1, 1+len(loss_vals)), loss_vals)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.show()
