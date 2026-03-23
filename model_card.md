# Model Card: Mood Machine

This model card documents **two** versions of a mood classifier built for the Mood Machine lab:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

Both models were built, evaluated, and compared.

## 1. Model Overview

**Model type:**
Both models were used and compared. The rule-based model was the primary focus, with the ML model serving as a benchmark.

**Intended purpose:**
Classify short text messages (social media posts, chat messages) into one of four mood labels: **positive**, **negative**, **neutral**, or **mixed**.

**How it works (brief):**
- **Rule-based:** Tokenizes text, matches tokens against curated positive/negative word lists, applies negation handling (flipping the score when a word is preceded by "not", "no", "never", etc.), adds emoji-based scoring, and checks for mixed signals (both positive and negative words present). The final numeric score maps to a label.
- **ML model:** Converts text into numeric vectors using bag-of-words (`CountVectorizer`), then trains a `LogisticRegression` classifier on the labeled dataset. It learns word-to-label associations directly from the data.

## 2. Data

**Dataset description:**
The dataset contains **20 labeled posts** in `dataset.py`. The original starter had 6 posts. 14 new posts were added across two rounds of expansion, covering slang, emojis, mixed emotions, and neutral language.

**Labeling process:**
Labels were assigned based on the presence of positive and negative words from the word lists:
- **positive**: contains positive-list words, no negative-list words
- **negative**: contains negative-list words, no positive-list words
- **mixed**: contains words from both lists
- **neutral**: contains words from neither list

Some posts were difficult to label. For example, `"Feeling tired but kind of hopeful"` was labeled **mixed** because it expresses both fatigue and optimism, even though `"hopeful"` is not in the positive word list — making it a known challenge for the rule-based model.

**Important characteristics of the dataset:**
- Contains Gen Z slang (`"lowkey"`, `"rn"`, `"im done"`)
- Includes emojis used as emotional signals (💪, 💀, 😂, ❤️)
- Several posts express genuinely mixed feelings
- Some posts are deliberately neutral with no emotional content
- Posts are short (3–10 words), mimicking social media style

**Possible issues with the dataset:**
- **Small size** (20 posts) — not enough for the ML model to generalize beyond memorization
- **Label imbalance** — more positive and negative examples than mixed or neutral
- **English-only** — no multilingual coverage
- **Demographic bias** — slang skews toward young, English-speaking, internet-native users. Formal language, regional dialects, and non-Western emotional expression styles are absent
- **Single annotator** — labels reflect one person's interpretation with no inter-annotator agreement

## 3. How the Rule-Based Model Works

**Preprocessing:**
- Strip whitespace, lowercase all text
- Remove punctuation while preserving emojis and emoticons
- Normalize repeated characters (`"soooo"` → `"soo"`)

**Scoring rules:**
- Each positive-list word adds **+1** to the score
- Each negative-list word subtracts **-1** from the score
- **Negation handling:** if a token is preceded by a negator (`not`, `no`, `never`, `dont`, `doesn`, `isn`, `wasn`, `aren`), its effect is flipped. `"not happy"` scores -1; `"not bad"` scores +1
- **Emoji scoring:** positive emojis (😂, ❤️, 😊, 💪, 🎉, `:)`) add +1; negative emojis (💀, 😢, 😭, 😞, `:(`) subtract -1

**Label mapping:**
- If both positive and negative word signals are found → `"mixed"`
- Otherwise: score > 0 → `"positive"`, score < 0 → `"negative"`, score == 0 → `"neutral"`

**Strengths of this approach:**
- Fully transparent — you can trace exactly why any prediction was made
- Negation handling catches common patterns like `"not happy"` and `"not bad"`
- Mixed detection works well when both word lists are represented
- Fast, no training required

**Weaknesses of this approach:**
- Only recognizes words in the curated lists — `"hopeful"`, `"grateful"`, `"overwhelmed"` are invisible
- Cannot detect sarcasm — `"I love getting stuck in traffic"` reads as positive because of `love`
- Slang is context-dependent — `"sick"` is positive as slang but negative in `"I feel sick"`
- Emojis are treated as fixed signals — 🙂 can be sincere or passive-aggressive

## 4. How the ML Model Works

**Features used:**
Bag-of-words representation using `CountVectorizer`. Each unique word becomes a feature; the value is its count in the text.

**Training data:**
Trained on the 20 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
- At 14 posts: 100% training accuracy
- At 20 posts: 100% training accuracy
- The model absorbed new examples without any accuracy drop, unlike the rule-based model which went from 93% to 90%

**Strengths:**
- Learns word-to-label associations from data without needing curated word lists
- Handles mixed detection through learned patterns rather than explicit rules
- Correctly classified `"i feel sad but also grateful for my friends"` as mixed, which the rule-based model missed

**Weaknesses:**
- 100% accuracy on training data likely reflects **memorization**, not generalization — with only 20 examples, the model has enough capacity to memorize every input
- Black box — no easy way to explain *why* a specific prediction was made
- Entirely dependent on the quality and diversity of the labeled data
- Would likely struggle with any sentence containing words not seen during training

## 5. Evaluation

**How the models were evaluated:**
Both models were evaluated on the same 20 labeled posts in `dataset.py`. Additionally, 8 adversarial "breaker" sentences were crafted to stress-test the rule-based model.

**Final accuracy:**

| Model | Dataset (20 posts) | Breaker Sentences (8) |
|---|---|---|
| Rule-based | 90% (18/20) | 75% (6/8) |
| ML model | 100% (20/20) | N/A (not tested on breakers) |

**Examples of correct predictions:**
- `"I am not happy about this"` → **negative**. Negation handling correctly flipped `"happy"` from positive to negative.
- `"tired but also excited for tomorrow :)"` → **mixed**. The model found `"tired"` (negative list) and `"excited"` (positive list), triggering the mixed detection logic.
- `"went to the store and came back"` → **neutral**. No words from either list, so the score stayed at 0.

**Examples of incorrect predictions:**
- `"Feeling tired but kind of hopeful"` → predicted **negative**, true label **mixed**. The model saw `"tired"` (-1) but `"hopeful"` is not in the positive word list, so it only detected one signal and missed the mixed sentiment.
- `"i feel sad but also grateful for my friends"` → predicted **negative**, true label **mixed**. Same root cause: `"grateful"` is not in the positive word list. The ML model got this one right because it learned the pattern from data.
- (Breaker) `"I love getting stuck in traffic"` → predicted **positive**, expected **negative**. The word `"love"` scored +1 and nothing counteracted it. Sarcasm requires understanding intent, not just keywords.

## 6. Limitations

- **Small dataset.** 20 posts is not enough to evaluate real-world performance. The ML model's 100% accuracy is misleading — it's evaluated on the same data it was trained on.
- **Vocabulary gaps are the primary failure mode.** The rule-based model misclassifies any sentence whose sentiment depends on words outside its lists. Example: `"i feel sad but also grateful"` → negative instead of mixed, because `"grateful"` is not in `POSITIVE_WORDS`.
- **Sarcasm is undetectable.** `"I love getting stuck in traffic"` → positive. The model has no mechanism to understand that positive words can carry negative intent through context.
- **Context-dependent slang.** `"sick"` is in the positive list for slang usage, but `"I feel sick"` would incorrectly score as positive.
- **No generalization to longer text.** The model was designed for short posts (under 15 words). Longer text with mixed signals would likely produce unreliable scores.
- **Single-language, single-demographic.** The word lists and training data reflect English-speaking, internet-native language patterns. The model would fail on formal English, other dialects, or non-English text.

## 7. Ethical Considerations

- **Misclassifying distress.** A post like `"I'm fine 🙂"` (a common way to mask negative emotions) is classified as neutral. In a real application (mental health monitoring, content moderation), this kind of false negative could mean missing someone who needs help.
- **Cultural and linguistic bias.** The word lists and dataset are built around American English internet slang. Users who express emotion differently — through formal language, AAVE, code-switching, or non-English phrases — would be systematically misread. The model is optimized for one demographic and would misinterpret others.
- **Privacy.** Mood analysis on personal messages raises consent and surveillance concerns. Even an inaccurate model can cause harm if its predictions are acted on without the user's knowledge.
- **Labeling subjectivity.** Labels were assigned by a single person. Mood is inherently subjective — `"This is fine"` could be neutral, sarcastic, or resigned depending on who reads it. The model reflects one person's interpretation, not ground truth.
- **Overconfidence from small evaluations.** A 90% or 100% accuracy on 20 examples can create false confidence. Deploying this model based on these numbers would be irresponsible without evaluation on a much larger, more diverse dataset.

## 8. Ideas for Improvement

- **Add a held-out test set.** Split the data so the ML model is evaluated on posts it hasn't trained on. This would give a realistic accuracy estimate.
- **Expand the dataset significantly.** 50–100+ labeled posts covering more demographics, languages, and emotional styles would improve both models.
- **Use TF-IDF instead of CountVectorizer.** TF-IDF weights words by importance, which could help the ML model prioritize emotionally charged words over common filler words.
- **Add contextual slang handling.** Instead of adding `"sick"` to the positive list globally, check for patterns like `"that's sick"` (positive) vs. `"feel sick"` (negative).
- **Introduce confidence scoring.** Instead of hard labels, output a confidence level. A score of +1 is a weaker signal than +4 — the model could flag low-confidence predictions for human review.
- **Use word embeddings or a transformer model.** Pre-trained models (Word2Vec, BERT) understand word meaning in context, which would address vocabulary gaps, sarcasm, and slang ambiguity — the three biggest limitations of the current approach.
- **Multiple annotators.** Have 2–3 people label each post independently and use majority vote. This reduces individual bias and surfaces genuinely ambiguous examples.
