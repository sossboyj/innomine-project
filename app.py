from reddit_scraper import fetch_posts
from text_cleaner import clean_text
import json
import csv

# Enhanced list of problem signal keywords/phrases
problem_keywords = [
    "bug", "bugs", "error", "errors", "issue", "issues", "crash", "crashes",
    "failing", "failed", "broken", "glitch", "freeze", "lag", "not working",
    "unexpected", "unresponsive", "slow", "timeout", "disconnect", "stuck",
    "how do i", "how can i", "can't seem", "need help", "missing feature",
    "not saving", "won’t load", "fails silently", "confused", "struggling",
    "doesn't respond", "login issue", "network error", "compile error",
    "runtime error", "api fail", "authentication error", "memory leak",
    "db not syncing", "token expired", "rate limited", "invalid", "no output",
    "wrong output", "missing", "won’t build", "crashing", "segfault", "infinite loop",
    "syntax error", "logic bug", "loop doesn't end", "stack overflow",
    "environment issue", "version conflict", "ci error", "deployment error",
    "cors issue", "ssl error", "authorization denied", "index out of range"
]

def is_problem_post(text):
    text = text.lower()
    signals_found = [kw for kw in problem_keywords if kw in text]
    return len(signals_found) >= 2, signals_found

def compute_problem_score(post, signals):
    score = (post['score'] * 0.5) + (len(post['comments']) * 1) + (len(signals) * 2)
    return score

def main():
    print("Fetching Reddit posts...\n")
    posts = fetch_posts(subreddit_name="learnprogramming", limit=30)

    filtered_posts = []

    for post in posts:
        full_text = f"{post['title']} {post['selftext']} {' '.join(post['comments'])}"
        is_problem, signals = is_problem_post(full_text)
        if is_problem:
            post['problem_signals'] = signals
            post['score_rank'] = compute_problem_score(post, signals)
            filtered_posts.append(post)

    if not filtered_posts:
        print("No qualified problem-related posts found.")
        return

    # Sort by computed score descending
    filtered_posts.sort(key=lambda x: x['score_rank'], reverse=True)

    # Save filtered posts
    with open("data/raw_data.json", "w") as f:
        json.dump(filtered_posts, f, indent=2)

    # Save summary to CSV
    with open("output/ranked_problems.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Rank", "Score", "Upvotes", "Comments", "Title", "Top Comment", "Signals", "URL"])
        for idx, post in enumerate(filtered_posts[:15], 1):
            writer.writerow([
                idx,
                round(post['score_rank'], 2),
                post['score'],
                len(post['comments']),
                post['title'],
                post['comments'][0] if post['comments'] else "",
                "; ".join(post['problem_signals']),
                post['url']
            ])

    print(f"\n{len(filtered_posts)} problem-heavy posts saved to data/raw_data.json")
    print("Top 15 ranked problems saved to output/ranked_problems.csv")

if __name__ == "__main__":
    main()
