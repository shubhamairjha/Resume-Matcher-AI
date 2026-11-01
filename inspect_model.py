
import joblib

# Load the trained model
model_path = "model.joblib"

artifacts = joblib.load(model_path)

print("\n===== MODEL CONTENTS =====")

# Show all saved objects
print("\nKeys inside model.joblib:")

for key in artifacts.keys():
    print("-", key)

# Show detailed information
print("\n===== DETAILS =====")

for key, value in artifacts.items():
    print(f"\n{key}:")
    print("Type:", type(value))

# Show classifier details
print("\n===== CLASSIFIER =====")

clf = artifacts["classifier"]

print("Model:", clf)
print("Classes:", clf.classes_)

# Show trained roles
print("\n===== ROLES =====")

print(artifacts["roles"])

# Show posting information
print("\n===== POSTING DATA =====")

print("Number of postings:", len(artifacts["posting_texts"]))

print("\nFirst posting text:")
print(artifacts["posting_texts"][0][:1000])