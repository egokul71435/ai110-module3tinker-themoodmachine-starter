"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "proud",
    "sick",
    "fire",
    "wicked",
    "dope",
    "blessed",
    "wonderful",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "exhausted",
    "frustrated",
    "miserable",
    "annoyed",
    "depressed",
    "trash",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "lowkey stressed but feeling good about it 💪",
    "this is so bad im upset rn 💀",
    "just another day nothing going on",
    "tired but also excited for tomorrow :)",
    "had an amazing time with friends today 😂",
    "i hate how boring this week has been",
    "went to the store and came back",
    "love this chill weekend so relaxed rn ❤️",
    "not gonna lie this is pretty awesome",
    "i feel sad but also grateful for my friends",
    "whatever i guess it does not matter",
    "this stressed me out but the results were great",
    "honestly happy for once",
    "bad news after bad news today im done",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "lowkey stressed but feeling good about it 💪" 
    "negative",  # "this is so bad im upset rn 💀" 
    "neutral",   # "just another day nothing going on" 
    "mixed",     # "tired but also excited for tomorrow :)" 
    "positive",  # "had an amazing time with friends today 😂" 
    "negative",  # "i hate how boring this week has been" 
    "neutral",   # "went to the store and came back" 
    "positive",  # "love this chill weekend so relaxed rn ❤️"
    "positive",  # "not gonna lie this is pretty awesome"
    "mixed",     # "i feel sad but also grateful for my friends"
    "neutral",   # "whatever i guess it does not matter"
    "mixed",     # "this stressed me out but the results were great"
    "positive",  # "honestly happy for once"
    "negative",  # "bad news after bad news today im done"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
