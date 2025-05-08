import pandas as pd

df = pd.read_csv("data/cleaned_labeled_posts.csv")
print(df["problem_signal"].value_counts())
