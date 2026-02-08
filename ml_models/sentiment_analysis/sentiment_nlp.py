import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize Reddit API client using PRAW
reddit = praw.Reddit(
    client_id='gT3o3Sr6dz0erhMrfZX5dg',
    client_secret='jxafYUg7Ak9DGeryoBwFci2XYn6ohw',
    user_agent='Novel-Statement-7667'
)

def analyze_sentiment(text):
    """
    Analyze sentiment of a given piece of text.
    Returns 'positive', 'neutral', or 'negative'.
    """
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "sentiment": sentiment,
        "compound_score": compound,
        "details": scores  # includes pos, neu, neg
    }

def fetch_and_analyze_reddit_posts(subreddit_name, limit=5):
    """
    Fetches posts from a specified subreddit and analyzes their sentiment.
    Returns a list of sentiment results for each post.
    """
    posts = reddit.subreddit(subreddit_name).hot(limit=limit)
    sentiment_results = []

    for post in posts:
        text = post.title + " " + post.selftext if post.selftext else post.title
        sentiment = analyze_sentiment(text)
        sentiment_results.append(sentiment)

    return sentiment_results

def get_trading_suggestion(sentiment_results):
    """
    Based on average compound score, suggest: Buy, Sell, or Hold.
    """
    if not sentiment_results:
        return "Hold (No data)"

    avg_score = sum(r['compound_score'] for r in sentiment_results) / len(sentiment_results)

    if avg_score >= 0.2:
        suggestion = "Buy"
    elif avg_score <= -0.2:
        suggestion = "Sell"
    else:
        suggestion = "Hold"

    return f"Trading Suggestion: {suggestion} (Average Compound Score: {avg_score:.3f})"
# Add this at the bottom of sentiment_nlp.py for quick testing
if __name__ == "__main__":
    data = fetch_and_analyze_reddit_posts("cryptocurrency", 3)
    print(data)
