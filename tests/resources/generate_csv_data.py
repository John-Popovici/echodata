import numpy as np
import pandas as pd

# reproducible random generator
rng = np.random.default_rng(42)

# ---- Training data ----
n_train = 100
X_train = np.vstack([
    rng.normal(loc=[0, 0], scale=0.3, size=(n_train // 2, 2)),   # class A
    rng.normal(loc=[2, 2], scale=0.3, size=(n_train // 2, 2))    # class B
])
y_train = np.array(["A"] * (n_train // 2) + ["B"] * (n_train // 2))

train_df = pd.DataFrame(X_train, columns=["feature1", "feature2"])
train_df["target"] = y_train
train_df.to_csv("train.csv", index=False)

# ---- Test data (with labels) ----
n_test = 40
X_test = np.vstack([
    rng.normal(loc=[0, 0], scale=0.3, size=(n_test // 2, 2)),    # class A
    rng.normal(loc=[2, 2], scale=0.3, size=(n_test // 2, 2))     # class B
])
y_test = np.array(["A"] * (n_test // 2) + ["B"] * (n_test // 2))

test_df = pd.DataFrame(X_test, columns=["feature1", "feature2"])
test_df["target"] = y_test
test_df.to_csv("test.csv", index=False)

print("Generated synthetic CSVs with labels:")
print("  - train.csv (features + target)")
print("  - test.csv  (features + target)")