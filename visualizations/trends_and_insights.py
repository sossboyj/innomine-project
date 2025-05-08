import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure visual style is consistent
sns.set(style="whitegrid")

# Load cleaned data
df = pd.read_csv("data/cleaned_labeled_posts.csv")

# Ensure date format
df["created_date"] = pd.to_datetime(df["created_date"])
df["date"] = df["created_date"].dt.date

# 1. Frequency of posts over time
plt.figure(figsize=(10, 4))
df.groupby("date").size().plot(kind="line", marker="o", title="Posts Over Time")
plt.xlabel("Date")
plt.ylabel("Post Count")
plt.tight_layout()
plt.savefig("visualizations/posts_over_time.png")
plt.clf()

#  2. Posts by subreddit
plt.figure(figsize=(8, 4))
sns.countplot(y="subreddit", data=df, order=df["subreddit"].value_counts().index)
plt.title("Post Count by Subreddit")
plt.tight_layout()
plt.savefig("visualizations/posts_by_subreddit.png")
plt.clf()

# 3. Problem signal proportion by subreddit
subreddit_group = df.groupby("subreddit")["problem_signal"].mean().sort_values(ascending=False)
subreddit_group.plot(kind="bar", color="coral", title="Problem Signal Rate by Subreddit")
plt.ylabel("Proportion of Problem Posts")
plt.tight_layout()
plt.savefig("visualizations/problem_rate_by_subreddit.png")
plt.clf()

#  4. Top words in problem-signal posts
from collections import Counter
all_words = " ".join(df[df["problem_signal"] == True]["clean_text"]).split()
common_words = Counter(all_words).most_common(20)
words, freqs = zip(*common_words)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(freqs), y=list(words))
plt.title("Top Words in Problem Posts")
plt.tight_layout()
plt.savefig("visualizations/top_words_problem_posts.png")
plt.clf()

print(" Visualizations saved in /visualizations/")
