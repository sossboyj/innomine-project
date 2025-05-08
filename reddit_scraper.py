import praw
import json
import os
from datetime import datetime, timedelta
from config import reddit  # Assumes you've stored credentials in config.py

# List of subreddits to scrape
SUBREDDITS = [
    "learnprogramming", "webdev", "reactjs", "datascience", "coding",
    "Python", "java", "cpp_questions", "programming", "AskProgramming",
    "machinelearning", "computerscience", "devops", "Frontend", "backend"
]

KEYWORDS = ["error", "issue", "stuck", "bug", "help", "can't", "frustrated", "how do I"]

DAYS_LIMIT = 30
LIMIT_PER_SUB = 100

def is_relevant(post):
    text = f"{post.title} {getattr(post, 'selftext', '')}".lower()
    return any(keyword in text for keyword in KEYWORDS)

def scrape_subreddits():
    all_data = []
    after_timestamp = datetime.utcnow() - timedelta(days=DAYS_LIMIT)

    for subreddit_name in SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"🔍 Scraping: r/{subreddit_name}")
        try:
            for post in subreddit.new(limit=LIMIT_PER_SUB):
                if datetime.utcfromtimestamp(post.created_utc) < after_timestamp:
                    continue
                if not is_relevant(post):
                    continue
                data = {
                    "subreddit": subreddit_name,
                    "title": post.title,
                    "text": getattr(post, "selftext", ""),
                    "created_utc": post.created_utc,
                    "url": post.url,
                    "score": post.score,
                    "num_comments": post.num_comments
                }
                all_data.append(data)
        except Exception as e:
            print(f"⚠️ Failed on r/{subreddit_name}: {e}")

    # Save data
    os.makedirs("data", exist_ok=True)
    with open("data/raw_data.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nFinished. Saved {len(all_data)} relevant posts to data/raw_data.json")

if __name__ == "__main__":
    scrape_subreddits()
