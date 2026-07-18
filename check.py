import pickle

with open("saved_objects/max_len.pkl", "rb") as f:
    print(pickle.load(f))