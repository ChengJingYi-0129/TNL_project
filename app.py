import gradio as gr


POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "enjoyed",
    "excellent",
    "fantastic",
    "good",
    "great",
    "liked",
    "love",
    "loved",
    "masterpiece",
    "nice",
    "perfect",
}

NEGATIVE_WORDS = {
    "awful",
    "bad",
    "boring",
    "disappointed",
    "disappointing",
    "hate",
    "hated",
    "poor",
    "slow",
    "terrible",
    "waste",
    "worst",
}


def predict_sentiment(review: str) -> dict[str, float]:
    text = review.lower()
    positive_score = sum(word in text for word in POSITIVE_WORDS)
    negative_score = sum(word in text for word in NEGATIVE_WORDS)

    if positive_score > negative_score:
        return {"Positive": 0.85, "Neutral": 0.10, "Negative": 0.05}

    if negative_score > positive_score:
        return {"Negative": 0.85, "Neutral": 0.10, "Positive": 0.05}

    return {"Neutral": 0.80, "Positive": 0.10, "Negative": 0.10}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Movie review",
        placeholder="Example: This movie was amazing and I loved the ending.",
        lines=5,
    ),
    outputs=gr.Label(label="Sentiment"),
    title="Cinema Sentiment Analysis",
    description="A simple Hugging Face Space test web app for movie review sentiment.",
    examples=[
        ["This movie was amazing and I loved the ending."],
        ["The story was boring and the acting was terrible."],
        ["The movie was okay."],
    ],
)


if __name__ == "__main__":
    demo.launch()
