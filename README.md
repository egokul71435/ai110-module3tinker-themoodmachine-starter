# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

---

## Repo Structure

```plaintext
├── dataset.py         # Starter word lists and example posts (you will expand these)
├── mood_analyzer.py   # Rule based classifier with TODOs to improve
├── main.py            # Runs the rule based model and interactive demo
├── ml_experiments.py  # (New) A tiny ML classifier using scikit-learn
├── model_card.md      # Template to fill out after experimenting
└── requirements.txt   # Dependencies for optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based starter:

    ```bash
    python main.py
    ```

If pieces of the analyzer are not implemented yet, you will see helpful errors that guide you to the TODOs.

To try the ML model later, run:

```bash
python ml_experiments.py
```

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Observe unusual or incorrect predictions and think about why they happen.
- Train a tiny machine learning model and compare its behavior to your rule based system.
- Complete the model card with your findings about data, behavior, limitations, and improvements.
- The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules.
- When debugging, print tokens, scores, or intermediate choices.
- Ask an AI assistant to help create edge case posts or unusual wording.
- Try examples that mislead or confuse your model. Failure cases teach you the most.

---

## Changes Made

### Step 1: Expanded the Dataset (`dataset.py`)

Added 8 new sample posts with matching true labels, covering slang, emojis, and mixed emotions:

| Post | Label |
|---|---|
| "lowkey stressed but feeling good about it 💪" | mixed |
| "this is so bad im upset rn 💀" | negative |
| "just another day nothing going on" | neutral |
| "tired but also excited for tomorrow :)" | mixed |
| "had an amazing time with friends today 😂" | positive |
| "i hate how boring this week has been" | negative |
| "went to the store and came back" | neutral |
| "love this chill weekend so relaxed rn ❤️" | positive |

Expanded word lists to include slang and additional emotional terms:
- **Positive:** `proud`, `sick`, `fire`, `wicked`, `dope`, `blessed`, `wonderful`
- **Negative:** `exhausted`, `frustrated`, `miserable`, `annoyed`, `depressed`, `trash`

### Step 2: Improved Preprocessing (`mood_analyzer.py`)

- Punctuation removal while preserving emojis and emoticons
- Repeated character normalization (`"soooo"` → `"soo"`)

### Step 3: Implemented `score_text` with Negation Handling

- Loops over tokens, scoring +1 for positive words and -1 for negative words
- Negation handling: if a token is preceded by a negator (`not`, `no`, `never`, `dont`, etc.), its effect is flipped (e.g., `"not happy"` scores -1 instead of +1)
- Emoji scoring: positive emojis (😂, ❤️, 💪, etc.) add +1, negative emojis (💀, 😢, etc.) subtract -1

### Step 4: Implemented `predict_label` with Mixed Detection

- If both positive and negative word signals are found in the text, returns `"mixed"`
- Otherwise falls back to score thresholds: `> 0` → positive, `< 0` → negative, `== 0` → neutral

---

## Performance at Each Step

| Step | Accuracy (Sample Dataset) | Accuracy (Breaker Sentences) |
|---|---|---|
| Baseline (unimplemented) | 0% (all `None`) | N/A |
| After `score_text` + `predict_label` (basic) | 79% (11/14) | 38% (3/8) |
| After vocabulary expansion | 79% (11/14) | 38% (3/8) |
| After emoji scoring + mixed detection | **93% (13/14)** | **75% (6/8)** |

---

## Breaker Sentences and Failure Analysis

Created 8 adversarial sentences to stress-test the model:

| Sentence | Predicted | Expected | Result |
|---|---|---|---|
| "I love getting stuck in traffic" | positive | negative | Failed (sarcasm) |
| "That movie was sick" | positive | positive | Passed |
| "I am fine 🙂" | neutral | mixed | Failed (emoji not scored, "fine" not in lists) |
| "I am exhausted but proud of myself" | mixed | mixed | Passed |
| "This is not bad at all" | positive | positive | Passed |
| "Great just great everything is terrible" | mixed | mixed | Passed |
| "lol this is so fire" | positive | positive | Passed |
| "I hate that I love this" | mixed | mixed | Passed |

---

## Known Limitations

- **Sarcasm is undetectable.** The model reads `"I love getting stuck in traffic"` as positive because it only sees the keyword `love`. Detecting sarcasm requires understanding context and intent, which is beyond keyword matching.
- **Vocabulary coverage gaps.** Words not in the word lists are invisible. `"Feeling tired but kind of hopeful"` predicts negative because `"hopeful"` is not in `POSITIVE_WORDS`. Adding words one by one is unsustainable — there will always be another missing word.
- **Slang is context-dependent.** Words like `"sick"` and `"fire"` are in the positive list as slang, but can be genuinely negative in other contexts (`"I feel sick"`).
- **Emojis are ambiguous.** 🙂 can be sincere or passive-aggressive. 😭 can mean sadness or overwhelming joy. The model treats them as fixed signals.

These limitations are inherent to rule-based keyword matching and motivate the use of ML models that can learn patterns from data rather than relying on manually curated lists.

---

## Model Comparison: Rule-Based vs. ML

### The Two Models

- **Rule-based (`MoodAnalyzer`):** Scores text by matching tokens against curated positive/negative word lists, with negation handling and emoji signals. Labels are derived from the net score and the presence of mixed signals.
- **ML model (`ml_experiments.py`):** Uses scikit-learn's `CountVectorizer` (bag-of-words) and `LogisticRegression` trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

### Performance

| Dataset Size | Rule-Based | ML Model |
|---|---|---|
| 14 posts | 93% (13/14) | 100% (14/14) |
| 20 posts | 90% (18/20) | 100% (20/20) |

The rule-based model **dropped from 93% to 90%** when new posts were added. The ML model **held at 100%** across both dataset sizes.

### Why the Disparity?

1. **Vocabulary coverage.** The rule-based model can only detect sentiment through words explicitly listed in `POSITIVE_WORDS` and `NEGATIVE_WORDS`. When a new post uses a word outside those lists (e.g., `"grateful"`, `"hopeful"`), the model is blind to it. The ML model learns word associations from the training data itself — it doesn't need a predefined list.

2. **Generalization from context.** The ML model sees the full bag of words in each sentence and learns which combinations map to which labels. The rule-based model evaluates each token independently without learning from patterns across sentences.

3. **Sensitivity to new data.** Adding more posts *hurts* the rule-based model by exposing more vocabulary gaps. Adding more posts *helps* the ML model by giving it more examples to learn from. This is the fundamental difference — rules are brittle to new data, while ML models benefit from it.

4. **Mixed detection.** The rule-based model can only label something `"mixed"` if it finds words from *both* word lists. The ML model learns the `"mixed"` pattern from labeled examples, so it can recognize mixed sentiment even when the specific words haven't been seen in that combination before.

### Caveats

The ML model's 100% accuracy is on its **training data**, not unseen data. With only 20 examples, it may be memorizing rather than truly generalizing. The rule-based model's logic is fully transparent and predictable; the ML model is a black box by comparison. In production, the right choice depends on whether you prioritize interpretability or accuracy.
