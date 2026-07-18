import pickle

max_len = 60

with open("saved_objects/max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("max_len.pkl updated successfully.")