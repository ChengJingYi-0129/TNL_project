import gradio as gr
import torch
import torch.nn as nn
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============ LOAD TOKENIZER ============
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# ============ LOAD LABEL ENCODER ============
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# ============ MODEL PARAMETERS ============
MAX_LEN = 150
vocab_size = 2103
embedding_dim = 100
hidden_size = 128
num_layers = 2
metadata_size = 3
num_classes = 3

# ============ MODEL ARCHITECTURE ============
class SentimentBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.4 if num_layers > 1 else 0,
        )
        self.metadata = nn.Sequential(
            nn.Linear(metadata_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, text_input, metadata_input):
        embedded = self.embedding(text_input)
        gru_output, _ = self.gru(embedded)
        pooled = torch.max(gru_output, dim=1).values
        meta = self.metadata(metadata_input)
        combined = torch.cat([pooled, meta], dim=1)
        return self.classifier(combined)

# ============ LOAD MODEL ============
device = torch.device("cpu")
model = SentimentBiGRU()
model.load_state_dict(torch.load('model_weights.pt', map_location=device))
model.to(device)
model.eval()

# ============ PREDICTION FUNCTION ============
def predict_sentiment(review_text):
    try:
        # Tokenize
        sequences = tokenizer.texts_to_sequences([review_text])
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        text_tensor = torch.tensor(padded, dtype=torch.long).to(device)
        
        # Metadata (zeros)
        metadata = np.zeros((1, metadata_size), dtype=np.float32)
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(text_tensor, metadata_tensor)
            probs = outputs.cpu().numpy()[0]
            class_idx = np.argmax(probs)
            sentiment = label_encoder.classes_[class_idx]
            confidence = probs[class_idx] * 100
        
        # Build result
        pos_prob = probs[0] * 100
        neu_prob = probs[1] * 100
        neg_prob = probs[2] * 100
        
        result = f"""
==================================================
ANALYSIS RESULTS
==================================================

Sentiment: {sentiment}
Confidence: {confidence:.1f}%

Score Breakdown:
  Positive: {pos_prob:.1f}%
  Neutral:  {neu_prob:.1f}%
  Negative: {neg_prob:.1f}%

Review Preview:
  {review_text[:200]}{"..." if len(review_text) > 200 else ""}

Word Count: {len(review_text.split())}
==================================================
"""
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# ============ GRADIO INTERFACE ============
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Enter Cinema Review",
        placeholder="Type your review here...",
        lines=6
    ),
    outputs=gr.Textbox(
        label="Analysis Results",
        lines=15
    ),
    title="🎬 Cinema Sentiment Analysis",
    description="Analyze cinema reviews with a PyTorch BiGRU model",
)

iface.launch()
