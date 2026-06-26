# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

Preprocessing -> numeric scoring -> mood label.
"""

from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

NEGATORS = {"not", "no", "never", "dont", "doesn", "isn", "wasn", "aren"}
POSITIVE_EMOJIS = {"😂", "❤️", "😊", "🥰", "😍", "💪", "🎉", "✨", ":)", ":-)"}
NEGATIVE_EMOJIS = {"💀", "😢", "😭", "😞", "😡", "🥲", ":(", ":-("}


class MoodAnalyzer:
    """A simple rule-based mood classifier."""

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """Convert raw text into a list of tokens the model can work with."""
        cleaned = text.strip().lower()

        # Remove punctuation but preserve emojis and emoticons.
        result = []
        for char in cleaned:
            if char.isalnum() or char.isspace() or ord(char) > 127:
                result.append(char)
            elif char in ":;)(:/-":
                result.append(char)
            else:
                result.append(" ")
        cleaned = "".join(result)

        # Normalize repeated characters (e.g., "soooo" -> "soo").
        normalized = []
        for char in cleaned:
            if len(normalized) >= 2 and normalized[-1] == char and normalized[-2] == char:
                continue
            normalized.append(char)

        return "".join(normalized).split()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _analyze_tokens(self, tokens: List[str]) -> Tuple[int, bool, bool]:
        """
        Scan tokens against word lists with negation awareness.

        Returns (word_score, has_positive_signal, has_negative_signal).
        Emoji signals are NOT included — call score_text for the full score.
        """
        score = 0
        has_positive = False
        has_negative = False

        for i, token in enumerate(tokens):
            negated = i > 0 and tokens[i - 1] in NEGATORS
            if token in self.positive_words:
                if negated:
                    score -= 1
                    has_negative = True
                else:
                    score += 1
                    has_positive = True
            elif token in self.negative_words:
                if negated:
                    score += 1
                    has_positive = True
                else:
                    score -= 1
                    has_negative = True

        return score, has_positive, has_negative

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric mood score for the given text.

        Positive words increase the score, negative words decrease it.
        Negation flips the effect of the following word.
        Emojis contribute +1 or -1 directly from the raw text.
        """
        tokens = self.preprocess(text)
        word_score, _, _ = self._analyze_tokens(tokens)

        emoji_score = 0
        for emoji in POSITIVE_EMOJIS:
            if emoji in text:
                emoji_score += 1
        for emoji in NEGATIVE_EMOJIS:
            if emoji in text:
                emoji_score -= 1

        return word_score + emoji_score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn text into a mood label: 'positive', 'negative', 'neutral', or 'mixed'.

        Returns 'mixed' when both positive and negative word signals are present.
        Falls back to score thresholds for everything else.
        """
        tokens = self.preprocess(text)
        score, has_positive, has_negative = self._analyze_tokens(tokens)

        for emoji in POSITIVE_EMOJIS:
            if emoji in text:
                score += 1
        for emoji in NEGATIVE_EMOJIS:
            if emoji in text:
                score -= 1

        if has_positive and has_negative:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining why the model chose its label.

        Example: 'Score = 2 (positive: ["love", "great"], negative: [])'
        """
        tokens = self.preprocess(text)
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            negated = i > 0 and tokens[i - 1] in NEGATORS
            if token in self.positive_words:
                (negative_hits if negated else positive_hits).append(token)
            elif token in self.negative_words:
                (positive_hits if negated else negative_hits).append(token)

        score = self.score_text(text)
        return (
            f"Score = {score} "
            f"(positive: {positive_hits}, "
            f"negative: {negative_hits})"
        )
