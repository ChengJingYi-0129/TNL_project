import gradio as gr
import torch
import torch.nn as nn
import pickle
import numpy as np
import os
import random
import csv
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============ PATH SETUP FOR HUGGING FACE ============
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
MODEL_FOLDER = os.path.join(BASE_PATH, 'best_sentiment_bigru')

# ============ LOAD MODEL FROM SPLIT FILES ============
def persistent_load(saved_id):
    if isinstance(saved_id, tuple) and len(saved_id) >= 5:
        storage_type = saved_id[0]
        storage_class = saved_id[1]
        key = saved_id[2]
        location = saved_id[3]
        size = saved_id[4]

        data_path = os.path.join(MODEL_FOLDER, 'data', str(key))
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                raw_data = f.read()

            array = np.frombuffer(raw_data, dtype=np.float32)
            storage = torch.FloatStorage.from_buffer(array.tobytes(), 'little')
            return storage
    return None

with open(os.path.join(MODEL_FOLDER, 'data.pkl'), 'rb') as f:
    unpickler = pickle.Unpickler(f)
    unpickler.persistent_load = persistent_load
    state_dict = unpickler.load()
print(f"Loaded state dict with {len(state_dict)} layers")

# ============ LOAD TOKENIZER & LABEL ENCODER ============
with open(BASE_PATH + 'word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

class SimpleTokenizer:
    def __init__(self, word_index):
        self.word_index = word_index
        self.oov_token = '<OOV>'

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in str(text).split():
                word = word.lower().strip('.,!?;:()"\'')
                if word in self.word_index:
                    seq.append(self.word_index[word])
                else:
                    seq.append(1)
            sequences.append(seq)
        return sequences

tokenizer = SimpleTokenizer(word_index)
print(f"Loaded tokenizer with {len(tokenizer.word_index)} words")

with open(BASE_PATH + 'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Label encoder loaded!")

# ============ MODEL ARCHITECTURE ============
MAX_LEN = 150
vocab_size = 2103
embedding_dim = 100
hidden_size = 128
num_layers = 2
metadata_size = 3
num_classes = 3

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

print("Model architecture defined!")

# ============ LOAD MODEL WEIGHTS ============
device = torch.device("cpu")
model = SentimentBiGRU()
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()
print("Model loaded successfully!")

# ============ HELPER FUNCTIONS ============
def tokenize_text(text, max_len=150):
    if not text or not text.strip():
        return torch.zeros(max_len, dtype=torch.long)

    sequences = tokenizer.texts_to_sequences([text])

    if sequences and sequences[0]:
        seq = sequences[0]
    else:
        seq = []

    if len(seq) > max_len:
        tokens = seq[:max_len]
    else:
        tokens = seq + [0] * (max_len - len(seq))

    return torch.tensor(tokens, dtype=torch.long)

def get_sentiment_emoji(sentiment):
    if sentiment == 'Positive':
        return '😊'
    elif sentiment == 'Neutral':
        return '😐'
    elif sentiment == 'Negative':
        return '😞'
    return ''

def get_confidence_emoji(confidence):
    if confidence >= 70:
        return '🟢'
    else:
        return '🔴'

def parse_datetime_advanced(dt_string):
    if not dt_string or not dt_string.strip():
        return None
    dt_string = dt_string.strip()

    try:
        if 'Z' in dt_string:
            dt_string_clean = dt_string.replace('Z', '+00:00')
            return datetime.fromisoformat(dt_string_clean)
        elif 'T' in dt_string and '.' in dt_string:
            base = dt_string.split('.')[0]
            if '+' in dt_string:
                tz = dt_string.split('+')[1]
                return datetime.strptime(base, '%Y-%m-%dT%H:%M:%S')
            else:
                return datetime.strptime(base, '%Y-%m-%dT%H:%M:%S')
        elif 'T' in dt_string:
            return datetime.strptime(dt_string, '%Y-%m-%dT%H:%M:%S')
    except:
        pass

    try:
        return datetime.strptime(dt_string, '%Y-%m-%d')
    except:
        pass

    formats = [
        '%m/%d/%Y %I:%M:%S %p',
        '%m/%d/%Y %I:%M %p',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_string, fmt)
        except ValueError:
            continue
    return None

def build_datetime_from_parts(date_str, time_str):
    if not date_str or not date_str.strip():
        return None
    combined = date_str.strip()
    if time_str and time_str.strip():
        combined = f"{combined} {time_str.strip()}"
    return parse_datetime_advanced(combined)

def get_time_features(dt):
    if dt is None:
        return 'Not specified', 0, 0, 0
    hour = dt.hour
    if 5 <= hour < 12:
        time_of_day = 'Morning'
    elif 12 <= hour < 17:
        time_of_day = 'Afternoon'
    elif 17 <= hour < 21:
        time_of_day = 'Evening'
    else:
        time_of_day = 'Night'
    day_of_year = dt.timetuple().tm_yday
    week_number = dt.isocalendar()[1]
    month = dt.month
    return time_of_day, day_of_year, week_number, month

print("Helper functions loaded")

# ============ ANALYSIS & PREDICTION FUNCTIONS ============
def analyze_review_with_time(review_text, dt=None):
    try:
        if not review_text or len(review_text.strip()) < 5:
            return "Neutral", 0, "Review too short to analyze.", "Not specified", "Not specified", 0, 0, 0, None, 0, 0, 0

        text_tensor = tokenize_text(review_text, MAX_LEN).unsqueeze(0).to(device)

        metadata = np.zeros((1, metadata_size), dtype=np.float32)
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(text_tensor, metadata_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pos_prob = probs[0] * 100
        neu_prob = probs[1] * 100
        neg_prob = probs[2] * 100

        class_idx = np.argmax(probs)
        if class_idx == 0:
            sentiment = "Negative"
        elif class_idx == 1:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        confidence = probs[class_idx] * 100

        if dt is None:
            timestamp_str = 'Not specified'
            time_of_day = 'Not specified'
            day_of_year = 0
            week_number = 0
            month = 0
        else:
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            time_of_day, day_of_year, week_number, month = get_time_features(dt)

        result_lines = []
        result_lines.append("-" * 50)
        result_lines.append("ANALYSIS RESULTS")
        result_lines.append("-" * 50)
        result_lines.append("")
        result_lines.append(f"Sentiment: {sentiment} {get_sentiment_emoji(sentiment)}")
        result_lines.append(f"Confidence: {confidence:.1f}% {get_confidence_emoji(confidence)}")
        result_lines.append(f"Date/Time: {timestamp_str}")
        result_lines.append("")
        result_lines.append("Score Breakdown:")
        result_lines.append(f"  Positive: {pos_prob:.1f}%")
        result_lines.append(f"  Neutral:  {neu_prob:.1f}%")
        result_lines.append(f"  Negative: {neg_prob:.1f}%")
        result_lines.append("")
        result_lines.append("Review Preview:")
        preview = review_text[:200] + "..." if len(review_text) > 200 else review_text
        result_lines.append(f"  {preview}")
        result_lines.append("")
        result_lines.append(f"Word Count: {len(review_text.split())}")
        result_lines.append("-" * 50)

        result = "\n".join(result_lines)

        return sentiment, confidence, result, timestamp_str, time_of_day, day_of_year, week_number, month, dt, pos_prob, neu_prob, neg_prob
    except Exception as e:
        print(f"Error: {e}")
        return "Neutral", 0, f"Could not analyze: {str(e)}", "Not specified", "Not specified", 0, 0, 0, None, 0, 0, 0

def random_example():
    examples = [
        {"review": "The movie was amazing! Great acting and beautiful cinematography.", "date": "6/28/2025", "time": "9:36:11 AM"},
        {"review": "Terrible service! The hall was dirty and the staff were rude.", "date": "6/27/2025", "time": "3:15:30 PM"},
        {"review": "The movie was okay. Nothing special but I didn't hate it.", "date": "6/26/2025", "time": "7:20:45 PM"},
        {"review": "Absolutely fantastic! Best movie I've seen this year. Highly recommend!", "date": "6/25/2025", "time": "10:00:00 AM"},
        {"review": "Waste of money. The plot made no sense and the acting was terrible.", "date": "6/24/2025", "time": "11:45:20 PM"}
    ]
    example = random.choice(examples)
    return example["review"], example["date"], example["time"]

print("Analysis functions loaded")

# ============ DATA STORAGE & DISPLAY FUNCTIONS ============
sentiment_history = []
all_reviews = []
all_results = []
last_analysis_results = []
sentiment_data = []
filtered_results_cache = []
filtered_flagged_cache = []
analyzed_label_to_filename = {}

FLAG_PATH = BASE_PATH + 'flagged_data/'
SAVED_PATH = BASE_PATH + 'saved_data/'

os.makedirs(FLAG_PATH, exist_ok=True)
os.makedirs(SAVED_PATH, exist_ok=True)

print(f"Flagged folder: {FLAG_PATH}")
print(f"Saved folder: {SAVED_PATH}")

def generate_summary_text():
    if not sentiment_data:
        return "No results to summarize yet."
    df = pd.DataFrame(sentiment_data)
    total = len(df)
    valid_data = df[df['sentiment'] != 'Error']
    valid_total = len(valid_data)
    if valid_total == 0:
        return "No valid results to summarize."
    pos_count = len(valid_data[valid_data['sentiment'] == 'Positive'])
    neu_count = len(valid_data[valid_data['sentiment'] == 'Neutral'])
    neg_count = len(valid_data[valid_data['sentiment'] == 'Negative'])
    avg_conf = valid_data['confidence'].mean()
    high_conf = len(valid_data[valid_data['confidence'] >= 70])
    low_conf = len(valid_data[valid_data['confidence'] < 70])
    summary = []
    summary.append("SUMMARY STATISTICS")
    summary.append("")
    summary.append(f"Total Reviews: {total}")
    summary.append(f"Valid Reviews: {valid_total}")
    summary.append("")
    summary.append("Sentiment Distribution:")
    if valid_total > 0:
        summary.append(f"  Positive: {pos_count} ({pos_count/valid_total*100:.1f}%)")
        summary.append(f"  Neutral:  {neu_count} ({neu_count/valid_total*100:.1f}%)")
        summary.append(f"  Negative: {neg_count} ({neg_count/valid_total*100:.1f}%)")
    summary.append("")
    summary.append("Confidence:")
    summary.append(f"  Average: {avg_conf:.1f}%")
    if valid_total > 0:
        summary.append(f"  High (70%+): {high_conf} ({high_conf/valid_total*100:.1f}%)")
        summary.append(f"  Low (70%-): {low_conf} ({low_conf/valid_total*100:.1f}%)")
    return "\n".join(summary)

def get_analyzed_results_display():
    global filtered_results_cache

    if not filtered_results_cache and all_results:
        filtered_results_cache = all_results.copy()

    if not all_results:
        return "No analyzed results yet.", []

    display_results = filtered_results_cache if filtered_results_cache else all_results

    html = """
    <style>
        .results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .results-table th { background: #D5C6E0; color: #000000 !important; padding: 10px; text-align: left; font-weight: 700; border: 1px solid #2A0800 !important; }
        .results-table td { padding: 8px 10px; border: 1px solid #2A0800 !important; vertical-align: top; background: #ffffff; }
        .results-table tr:hover td { background: #f5f0f7 !important; }
        .results-container { max-height: 400px; overflow-y: auto; border: 2px solid #2A0800 !important; border-radius: 8px; background: #ffffff; }
        .table-total { color: #ffffff !important; font-weight: 700; }
        .sentiment-pos { color: #2ecc71 !important; font-weight: 700; }
        .sentiment-neu { color: #f1c40f !important; font-weight: 700; }
        .sentiment-neg { color: #e74c3c !important; font-weight: 700; }
        .conf-high { color: #2ecc71 !important; font-weight: 700; }
        .conf-low { color: #e74c3c !important; font-weight: 700; }
        .review-preview { max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #000000 !important; }
        .review-preview:hover { white-space: normal; overflow: visible; background: #ffffff; position: relative; z-index: 20; box-shadow: 0 2px 8px rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px; border: 1px solid #2A0800; }
    </style>
    <div class="results-container">
        <table class="results-table">
            <thead>
                <tr>
                    <th style="width:40px;">#</th>
                    <th style="width:100px;">Sentiment</th>
                    <th style="width:100px;">Confidence</th>
                    <th style="width:80px;">Positive</th>
                    <th style="width:80px;">Neutral</th>
                    <th style="width:80px;">Negative</th>
                    <th style="width:150px;">Date/Time</th>
                    <th>Review</th>
                </tr>
            </thead>
            <tbody>
    """
    for i, res in enumerate(display_results, 1):
        sentiment = res.get('sentiment', 'Unknown')
        emoji = get_sentiment_emoji(sentiment)
        confidence = res.get('confidence', 0)
        pos_prob = res.get('pos_prob', 0)
        neu_prob = res.get('neu_prob', 0)
        neg_prob = res.get('neg_prob', 0)
        timestamp = res.get('timestamp', 'Not specified')
        review = res.get('review', '')
        preview = review[:60] + "..." if len(review) > 60 else review
        sent_class = 'sentiment-pos' if sentiment == 'Positive' else 'sentiment-neu' if sentiment == 'Neutral' else 'sentiment-neg'
        conf_class = 'conf-high' if confidence >= 70 else 'conf-low'
        html += f"""
            <tr>
                <td style="color:#000000 !important; text-align:center;">{i}</td>
                <td class="{sent_class}">{emoji} {sentiment}</td>
                <td class="{conf_class}">{confidence:.1f}%</td>
                <td style="color:#000000 !important;">{pos_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neu_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neg_prob:.1f}%</td>
                <td style="color:#000000 !important;">{timestamp}</td>
                <td class="review-preview" title="{review}" style="color:#000000 !important;">{preview}</td>
            </tr>
        """
    html += """
            </tbody>
        </table>
    </div>
    """
    total_label = "filtered" if filtered_results_cache and len(filtered_results_cache) != len(all_results) else "total"
    html += f"<br><span class='table-total'>Total: {len(display_results)} results ({total_label})</span>"

    checkbox_choices = []
    for i, res in enumerate(display_results, 1):
        timestamp = res.get('timestamp', 'No date')
        date_part = timestamp.split(' ')[0] if ' ' in timestamp else timestamp[:10]
        checkbox_choices.append(f"[{i}] {res.get('sentiment', 'Unknown')} - {date_part}")

    return html, checkbox_choices

def get_flagged_files():
    if not os.path.exists(FLAG_PATH):
        return []
    files = [f for f in os.listdir(FLAG_PATH) if f.endswith('.csv')]
    return sorted(files)

def parse_flagged_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            sentiment = "Unknown"
            sentiment_match = re.search(r'Sentiment:\s*(\w+)', content)
            if sentiment_match:
                sentiment = sentiment_match.group(1)
            confidence = 0
            conf_match = re.search(r'Confidence:\s*([\d.]+)%', content)
            if conf_match:
                confidence = float(conf_match.group(1))
            pos_prob = neu_prob = neg_prob = 0
            pos_match = re.search(r'Positive:\s*([\d.]+)%', content)
            if pos_match:
                pos_prob = float(pos_match.group(1))
            neu_match = re.search(r'Neutral:\s*([\d.]+)%', content)
            if neu_match:
                neu_prob = float(neu_match.group(1))
            neg_match = re.search(r'Negative:\s*([\d.]+)%', content)
            if neg_match:
                neg_prob = float(neg_match.group(1))
            review_preview = review_full = ""
            lines = content.strip().split('\n')
            for line in lines:
                if line.startswith('Review,') or line.startswith('"Review,'):
                    continue
                if ',' in line and not line.startswith('Review'):
                    parts = line.split(',', 1)
                    if parts:
                        review_text = parts[0].strip('"')
                        review_full = review_text
                        review_preview = review_text[:50] + "..." if len(review_text) > 50 else review_text
                        break
            date_str = ""
            date_match = re.search(r'Date/Time:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
            if date_match:
                date_str = date_match.group(1)
            return {'content': content, 'sentiment': sentiment, 'confidence': confidence, 'pos_prob': pos_prob, 'neu_prob': neu_prob, 'neg_prob': neg_prob, 'date': date_str, 'review_preview': review_preview, 'review_full': review_full, 'filename': os.path.basename(filepath)}
    except Exception as e:
        return None

def get_flagged_display_table():
    files = get_flagged_files()
    if not files:
        return "No flagged data yet.", []
    parsed_files = []
    for file in files:
        filepath = os.path.join(FLAG_PATH, file)
        parsed = parse_flagged_file(filepath)
        if parsed:
            parsed_files.append(parsed)
    html = """
    <style>
        .flagged-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .flagged-table th { background: #D5C6E0; color: #000000 !important; padding: 10px; text-align: left; font-weight: 700; border: 1px solid #2A0800 !important; }
        .flagged-table td { padding: 8px 10px; border: 1px solid #2A0800 !important; vertical-align: top; background: #ffffff; }
        .flagged-table tr:hover td { background: #f5f0f7 !important; }
        .flagged-container { max-height: 400px; overflow-y: auto; border: 2px solid #2A0800 !important; border-radius: 8px; background: #ffffff; }
        .table-total { color: #ffffff !important; font-weight: 700; }
        .sentiment-pos { color: #2ecc71 !important; font-weight: 700; }
        .sentiment-neu { color: #f1c40f !important; font-weight: 700; }
        .sentiment-neg { color: #e74c3c !important; font-weight: 700; }
        .conf-high { color: #2ecc71 !important; font-weight: 700; }
        .conf-low { color: #e74c3c !important; font-weight: 700; }
        .review-preview { max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #000000 !important; }
        .review-preview:hover { white-space: normal; overflow: visible; background: #ffffff; position: relative; z-index: 20; box-shadow: 0 2px 8px rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px; border: 1px solid #2A0800; }
    </style>
    <div class="flagged-container">
        <table class="flagged-table">
            <thead>
                <tr>
                    <th style="width:40px;">#</th>
                    <th style="width:100px;">Sentiment</th>
                    <th style="width:100px;">Confidence</th>
                    <th style="width:80px;">Positive</th>
                    <th style="width:80px;">Neutral</th>
                    <th style="width:80px;">Negative</th>
                    <th style="width:180px;">Date/Time</th>
                    <th>Review</th>
                </tr>
            </thead>
            <tbody>
    """
    for i, parsed in enumerate(parsed_files, 1):
        sentiment = parsed.get('sentiment', 'Unknown')
        emoji = get_sentiment_emoji(sentiment)
        confidence = parsed.get('confidence', 0)
        pos_prob = parsed.get('pos_prob', 0)
        neu_prob = parsed.get('neu_prob', 0)
        neg_prob = parsed.get('neg_prob', 0)
        date_display = parsed.get('date', 'Unknown')
        review = parsed.get('review_preview', '')
        review_full = parsed.get('review_full', '')
        sent_class = 'sentiment-pos' if sentiment == 'Positive' else 'sentiment-neu' if sentiment == 'Neutral' else 'sentiment-neg'
        conf_class = 'conf-high' if confidence >= 70 else 'conf-low'
        html += f"""
            <tr>
                <td style="color:#000000 !important; text-align:center;">{i}</td>
                <td class="{sent_class}">{emoji} {sentiment}</td>
                <td class="{conf_class}">{confidence:.1f}%</td>
                <td style="color:#000000 !important;">{pos_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neu_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neg_prob:.1f}%</td>
                <td style="color:#000000 !important;">{date_display}</td>
                <td class="review-preview" title="{review_full}" style="color:#000000 !important;">{review}</td>
            </tr>
        """
    html += """
            </tbody>
        </table>
    </div>
    """
    html += f"<br><span class='table-total'>Total: {len(parsed_files)} flagged files</span>"
    checkbox_choices = []
    for i, parsed in enumerate(parsed_files, 1):
        sentiment = parsed.get('sentiment', 'Unknown')
        date_display = parsed.get('date', 'Unknown')
        short_date = date_display[:10] if len(date_display) >= 10 else date_display
        checkbox_choices.append(f"[{i}] {sentiment} - {short_date}")
    return html, checkbox_choices

def create_empty_chart_figure():
    fig = go.Figure()
    fig.add_annotation(
        text="No data available yet. Start analyzing reviews!",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="white")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        height=350
    )
    return fig

# ============ CHART CREATION FUNCTIONS ============
def create_all_charts():
    if not sentiment_data:
        empty_fig = create_empty_chart_figure()
        return {
            'pie': empty_fig, 'conf_bar': empty_fig, 'trend': empty_fig,
            'time_day': empty_fig, 'day_week': empty_fig, 'month': empty_fig,
            'year': empty_fig,
            'stats_overview': "No data available yet.",
            'stats_time': "No data available yet."
        }

    valid_data = [d for d in sentiment_data if d['sentiment'] != 'Error']
    if not valid_data:
        empty_fig = create_empty_chart_figure()
        return {
            'pie': empty_fig, 'conf_bar': empty_fig, 'trend': empty_fig,
            'time_day': empty_fig, 'day_week': empty_fig, 'month': empty_fig,
            'year': empty_fig,
            'stats_overview': "No valid data available.",
            'stats_time': "No valid data available."
        }

    df = pd.DataFrame(valid_data)
    colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
    bg_color = '#1a1a2e'
    text_color = 'white'
    grid_color = 'rgba(255,255,255,0.15)'

    # Pie Chart
    pie_fig = go.Figure()
    sentiment_counts = df['sentiment'].value_counts()
    if not sentiment_counts.empty:
        pie_fig.add_trace(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=[colors.get(s, '#95a5a6') for s in sentiment_counts.index]),
            hole=0.3,
            textinfo='label+percent',
            textposition='auto',
            textfont=dict(color=text_color, size=12)
        ))
    pie_fig.update_layout(
        height=350,
        showlegend=True,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, size=12),
        legend=dict(font=dict(color=text_color))
    )

    # Confidence Bar
    conf_bar_fig = go.Figure()
    confidence_counts = df['confidence'].apply(lambda x: 'High (≥70%)' if x >= 70 else 'Low (<70%)').value_counts()
    if not confidence_counts.empty:
        conf_bar_fig.add_trace(go.Bar(
            x=confidence_counts.index,
            y=confidence_counts.values,
            marker_color=['#2ecc71' if 'High' in x else '#e74c3c' for x in confidence_counts.index],
            text=confidence_counts.values,
            textposition='auto',
            textfont=dict(color=text_color)
        ))
    conf_bar_fig.update_layout(
        height=350,
        title=dict(text="Confidence Levels", font=dict(color=text_color, size=14)),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        yaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color)
    )

    # Sentiment Trend
    trend_fig = go.Figure()
    if 'timestamp' in df.columns:
        date_df = df[df['timestamp'] != 'Not specified'].copy()
        if not date_df.empty:
            date_df['date'] = pd.to_datetime(date_df['timestamp'], errors='coerce')
            date_df = date_df.dropna(subset=['date'])
            if not date_df.empty:
                daily_counts = date_df.groupby([date_df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                daily_totals = daily_counts.sum(axis=1)
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    if sentiment in daily_counts.columns:
                        pct = (daily_counts[sentiment] / daily_totals * 100).fillna(0)
                        trend_fig.add_trace(go.Scatter(
                            x=daily_totals.index,
                            y=pct,
                            mode='lines+markers',
                            name=sentiment,
                            line=dict(color=colors.get(sentiment, '#95a5a6'), width=3),
                            marker=dict(size=6)
                        ))
    trend_fig.update_layout(
        height=300,
        title=dict(text="Sentiment Trend Over Time", font=dict(color=text_color, size=14)),
        xaxis=dict(title="Date", title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        yaxis=dict(title="Percentage %", title_font=dict(color=text_color), tickfont=dict(color=text_color), range=[0, 100], gridcolor=grid_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        legend=dict(font=dict(color=text_color))
    )

    # Time of Day
    time_day_fig = go.Figure()
    if 'time_of_day' in df.columns:
        time_df = df[df['time_of_day'] != 'Not specified']
        if not time_df.empty:
            time_sentiment = time_df.groupby(['time_of_day', 'sentiment']).size().unstack(fill_value=0)
            time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            time_sentiment = time_sentiment.reindex([t for t in time_order if t in time_sentiment.index])
            for sentiment in time_sentiment.columns:
                time_day_fig.add_trace(go.Bar(
                    name=sentiment,
                    x=time_sentiment.index,
                    y=time_sentiment[sentiment],
                    marker_color=colors.get(sentiment, '#95a5a6'),
                    text=time_sentiment[sentiment],
                    textposition='auto',
                    textfont=dict(color=text_color)
                ))
    time_day_fig.update_layout(
        height=350,
        title=dict(text="Sentiment by Time of Day", font=dict(color=text_color, size=14)),
        barmode='stack',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        yaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        legend=dict(font=dict(color=text_color))
    )

    # Day of Week
    day_week_fig = go.Figure()
    if 'timestamp' in df.columns:
        date_df = df[df['timestamp'] != 'Not specified'].copy()
        if not date_df.empty:
            date_df['day_of_week'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.day_name()
            date_df = date_df.dropna(subset=['day_of_week'])
            if not date_df.empty:
                day_data = date_df.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)
                if not day_data.empty:
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_data = day_data.reindex([d for d in day_order if d in day_data.index])
                    for sentiment in day_data.columns:
                        day_week_fig.add_trace(go.Bar(
                            name=sentiment,
                            x=day_data.index,
                            y=day_data[sentiment],
                            marker_color=colors.get(sentiment, '#95a5a6'),
                            text=day_data[sentiment],
                            textposition='auto',
                            textfont=dict(color=text_color)
                        ))
    day_week_fig.update_layout(
        height=350,
        title=dict(text="Sentiment by Day of Week", font=dict(color=text_color, size=14)),
        barmode='stack',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        yaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        legend=dict(font=dict(color=text_color))
    )

    # Month
    month_fig = go.Figure()
    if 'timestamp' in df.columns:
        date_df = df[df['timestamp'] != 'Not specified'].copy()
        if not date_df.empty:
            date_df['month'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.month_name()
            date_df = date_df.dropna(subset=['month'])
            if not date_df.empty:
                month_data = date_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                if not month_data.empty:
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                  'July', 'August', 'September', 'October', 'November', 'December']
                    month_data = month_data.reindex([m for m in month_order if m in month_data.index])
                    for sentiment in month_data.columns:
                        month_fig.add_trace(go.Bar(
                            name=sentiment,
                            x=month_data.index,
                            y=month_data[sentiment],
                            marker_color=colors.get(sentiment, '#95a5a6'),
                            text=month_data[sentiment],
                            textposition='auto',
                            textfont=dict(color=text_color)
                        ))
    month_fig.update_layout(
        height=350,
        title=dict(text="Sentiment by Month", font=dict(color=text_color, size=14)),
        barmode='stack',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        yaxis=dict(title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        legend=dict(font=dict(color=text_color))
    )

    # Year
    year_fig = go.Figure()
    if 'timestamp' in df.columns:
        date_df = df[df['timestamp'] != 'Not specified'].copy()
        if not date_df.empty:
            date_df['year'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.year
            date_df = date_df.dropna(subset=['year'])
            if not date_df.empty:
                year_data = date_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
                if not year_data.empty:
                    year_data = year_data.sort_index()
                    for sentiment in year_data.columns:
                        year_fig.add_trace(go.Bar(
                            name=sentiment,
                            x=year_data.index.astype(str),
                            y=year_data[sentiment],
                            marker_color=colors.get(sentiment, '#95a5a6'),
                            text=year_data[sentiment],
                            textposition='auto',
                            textfont=dict(color=text_color)
                        ))
    year_fig.update_layout(
        height=350,
        title=dict(text="Sentiment by Year", font=dict(color=text_color, size=14)),
        barmode='stack',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(title="Year", title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        yaxis=dict(title="Number of Reviews", title_font=dict(color=text_color), tickfont=dict(color=text_color), gridcolor=grid_color),
        legend=dict(font=dict(color=text_color))
    )

    stats_overview = generate_detailed_summary_stats(df, "Overview")
    stats_time = generate_detailed_summary_stats(df, "Time")

    return {
        'pie': pie_fig,
        'conf_bar': conf_bar_fig,
        'trend': trend_fig,
        'time_day': time_day_fig,
        'day_week': day_week_fig,
        'month': month_fig,
        'year': year_fig,
        'stats_overview': stats_overview,
        'stats_time': stats_time
    }

# ============ SUMMARY STATS FUNCTIONS ============
def generate_detailed_summary_stats(df, tab_name="Overview"):
    if df.empty:
        return "No data available yet."

    total = len(df)
    pos_count = len(df[df['sentiment'] == 'Positive'])
    neu_count = len(df[df['sentiment'] == 'Neutral'])
    neg_count = len(df[df['sentiment'] == 'Negative'])
    avg_conf = df['confidence'].mean()
    high_conf = len(df[df['confidence'] >= 70])

    extra_rows = ""

    if tab_name == "Overview":
        dominant = max([('Positive', pos_count), ('Neutral', neu_count), ('Negative', neg_count)], key=lambda x: x[1])[0]
        extra_rows = f"""
        <tr>
            <td class="stats-label">Dominant Sentiment:</td>
            <td class="stats-value" colspan="3">{dominant}</td>
        </tr>
        """

    elif tab_name == "Time":
        time_df = df[df['time_of_day'] != 'Not specified']
        time_counts = time_df['time_of_day'].value_counts()
        top_time = time_counts.index[0] if len(time_counts) > 0 else "N/A"

        most_active_day = "N/A"
        most_active_day_count = 0
        if 'timestamp' in df.columns:
            date_df = df[df['timestamp'] != 'Not specified'].copy()
            if not date_df.empty:
                date_df['day_of_week'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.day_name()
                date_df = date_df.dropna(subset=['day_of_week'])
                day_counts = date_df['day_of_week'].value_counts()
                if not day_counts.empty:
                    most_active_day = day_counts.index[0]
                    most_active_day_count = day_counts.values[0]

        most_active_month = "N/A"
        most_active_month_count = 0
        if 'timestamp' in df.columns:
            date_df = df[df['timestamp'] != 'Not specified'].copy()
            if not date_df.empty:
                date_df['month'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.month_name()
                date_df = date_df.dropna(subset=['month'])
                month_counts = date_df['month'].value_counts()
                if not month_counts.empty:
                    most_active_month = month_counts.index[0]
                    most_active_month_count = month_counts.values[0]

        most_active_year = "N/A"
        most_active_year_count = 0
        if 'timestamp' in df.columns:
            date_df = df[df['timestamp'] != 'Not specified'].copy()
            if not date_df.empty:
                date_df['year'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.year
                date_df = date_df.dropna(subset=['year'])
                year_counts = date_df['year'].value_counts()
                if not year_counts.empty:
                    most_active_year = str(int(year_counts.index[0]))
                    most_active_year_count = year_counts.values[0]

        best_time = "N/A"
        best_time_ratio = 0
        for time_period in time_counts.index:
            time_data = time_df[time_df['time_of_day'] == time_period]
            if len(time_data) > 0:
                pos_ratio = len(time_data[time_data['sentiment'] == 'Positive']) / len(time_data) * 100
                if pos_ratio > best_time_ratio:
                    best_time_ratio = pos_ratio
                    best_time = time_period

        best_day = "N/A"
        best_day_ratio = 0
        if 'timestamp' in df.columns:
            date_df = df[df['timestamp'] != 'Not specified'].copy()
            if not date_df.empty:
                date_df['day_of_week'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.day_name()
                date_df = date_df.dropna(subset=['day_of_week'])
                for day in date_df['day_of_week'].unique():
                    day_data = date_df[date_df['day_of_week'] == day]
                    if len(day_data) > 0:
                        pos_ratio = len(day_data[day_data['sentiment'] == 'Positive']) / len(day_data) * 100
                        if pos_ratio > best_day_ratio:
                            best_day_ratio = pos_ratio
                            best_day = day

        best_month = "N/A"
        best_month_ratio = 0
        if 'timestamp' in df.columns:
            date_df = df[df['timestamp'] != 'Not specified'].copy()
            if not date_df.empty:
                date_df['month'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.month_name()
                date_df = date_df.dropna(subset=['month'])
                for month in date_df['month'].unique():
                    month_data = date_df[date_df['month'] == month]
                    if len(month_data) > 0:
                        pos_ratio = len(month_data[month_data['sentiment'] == 'Positive']) / len(month_data) * 100
                        if pos_ratio > best_month_ratio:
                            best_month_ratio = pos_ratio
                            best_month = month

        best_year = "N/A"
        best_year_ratio = 0
        if 'timestamp' in df.columns:
            date_df = df[df['timestamp'] != 'Not specified'].copy()
            if not date_df.empty:
                date_df['year'] = pd.to_datetime(date_df['timestamp'], errors='coerce').dt.year
                date_df = date_df.dropna(subset=['year'])
                for year in date_df['year'].unique():
                    year_data = date_df[date_df['year'] == year]
                    if len(year_data) > 0:
                        pos_ratio = len(year_data[year_data['sentiment'] == 'Positive']) / len(year_data) * 100
                        if pos_ratio > best_year_ratio:
                            best_year_ratio = pos_ratio
                            best_year = str(int(year))

        extra_rows = f"""
        <tr>
            <td class="stats-label">Most Active Time:</td>
            <td class="stats-value">{top_time}</td>
            <td class="stats-label">Best Performing Time:</td>
            <td class="stats-value">{best_time} ({best_time_ratio:.1f}% positive)</td>
        </tr>
        <tr>
            <td class="stats-label">Most Active Day:</td>
            <td class="stats-value">{most_active_day} ({most_active_day_count} reviews)</td>
            <td class="stats-label">Best Performing Day:</td>
            <td class="stats-value">{best_day} ({best_day_ratio:.1f}% positive)</td>
        </tr>
        <tr>
            <td class="stats-label">Most Active Month:</td>
            <td class="stats-value">{most_active_month} ({most_active_month_count} reviews)</td>
            <td class="stats-label">Best Performing Month:</td>
            <td class="stats-value">{best_month} ({best_month_ratio:.1f}% positive)</td>
        </tr>
        <tr>
            <td class="stats-label">Most Active Year:</td>
            <td class="stats-value">{most_active_year} ({most_active_year_count} reviews)</td>
            <td class="stats-label">Best Performing Year:</td>
            <td class="stats-value">{best_year} ({best_year_ratio:.1f}% positive)</td>
        </tr>
        """

    return f"""
    <style>
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
        }}
        .stats-table th {{
            background: #D5C6E0;
            color: #000000 !important;
            padding: 10px;
            text-align: left;
            font-weight: 700;
            border: 1px solid #2A0800 !important;
        }}
        .stats-table td {{
            padding: 10px;
            border: 1px solid #2A0800 !important;
            vertical-align: top;
            background: #ffffff;
            color: #000000 !important;
        }}
        .stats-table tr:hover td {{
            background: #f5f0f7 !important;
        }}
        .stats-container {{
            border: 2px solid #2A0800 !important;
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
            margin-top: 16px;
        }}
        .stats-header {{
            background: #D5C6E0 !important;
            color: #000000 !important;
            font-weight: 700;
            text-align: center;
            font-size: 15px;
        }}
        .stats-positive {{ color: #2ecc71 !important; font-weight: 400; }}
        .stats-neutral {{ color: #f1c40f !important; font-weight: 400; }}
        .stats-negative {{ color: #e74c3c !important; font-weight: 400; }}
        .stats-high {{ color: #2ecc71 !important; font-weight: 400; }}
        .stats-label {{
            color: #000000 !important;
            font-weight: 700;
        }}
        .stats-value {{
            color: #000000 !important;
            font-weight: 400;
        }}
    </style>
    <div class="stats-container">
        <table class="stats-table">
            <thead>
                <tr>
                    <th colspan="4" class="stats-header">{tab_name} Statistics</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="stats-label">Total Reviews:</td>
                    <td class="stats-value">{total}</td>
                    <td class="stats-label">Positive:</td>
                    <td class="stats-positive">{pos_count} ({pos_count/total*100:.1f}%)</td>
                </tr>
                <tr>
                    <td class="stats-label">Neutral:</td>
                    <td class="stats-neutral">{neu_count} ({neu_count/total*100:.1f}%)</td>
                    <td class="stats-label">Negative:</td>
                    <td class="stats-negative">{neg_count} ({neg_count/total*100:.1f}%)</td>
                </tr>
                <tr>
                    <td class="stats-label">Avg Confidence:</td>
                    <td class="stats-value">{avg_conf:.1f}%</td>
                    <td class="stats-label">High Confidence (≥70%):</td>
                    <td class="stats-high">{high_conf} ({high_conf/total*100:.1f}%)</td>
                </tr>
                {extra_rows}
            </tbody>
        </table>
    </div>
    """

# ============ ACTION FUNCTIONS ============

def select_all_analyzed():
    _, choices = get_analyzed_results_display()
    return gr.update(value=choices)

def deselect_all_analyzed():
    return gr.update(value=[])

def select_all_flagged():
    _, choices = get_flagged_display_table()
    return gr.update(value=choices)

def deselect_all_flagged():
    return gr.update(value=[])

def delete_selected_from_table(selected_labels):
    global sentiment_data, all_results, all_reviews, sentiment_history, last_analysis_results, filtered_results_cache
    if not selected_labels or not filtered_results_cache:
        return "No results selected or no results to delete."
    indices_to_delete = []
    for label in selected_labels:
        match = re.search(r'\[(\d+)\]', label)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(filtered_results_cache):
                review_text = filtered_results_cache[idx].get('review', '')
                for i, res in enumerate(all_results):
                    if res.get('review') == review_text:
                        indices_to_delete.append(i)
                        break
    if not indices_to_delete:
        return "No valid results selected."
    indices_to_delete.sort(reverse=True)
    deleted_count = 0
    for idx in indices_to_delete:
        if 0 <= idx < len(all_results):
            try:
                del all_results[idx]
                if idx < len(sentiment_history):
                    del sentiment_history[idx]
                if idx < len(all_reviews):
                    del all_reviews[idx]
                if idx < len(sentiment_data):
                    del sentiment_data[idx]
                deleted_count += 1
            except:
                pass
    if len(last_analysis_results) > 0 and len(all_results) == 0:
        last_analysis_results = []
    filtered_results_cache = all_results.copy()
    return f"Deleted {deleted_count} result(s)."

def delete_all_analyzed():
    global sentiment_data, last_analysis_results, filtered_results_cache
    if not all_results:
        return "No results to delete."
    count = len(all_results)
    sentiment_history.clear()
    all_reviews.clear()
    all_results.clear()
    sentiment_data = []
    last_analysis_results = []
    filtered_results_cache = []
    return f"Deleted all {count} results."

# ============ SAVE FUNCTIONS ============

def save_selected_results_from_table(selected_labels):
    if not selected_labels or not filtered_results_cache:
        return None

    selected_results = []
    for label in selected_labels:
        match = re.search(r'\[(\d+)\]', label)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(filtered_results_cache):
                selected_results.append(filtered_results_cache[idx])

    if not selected_results:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exported_results_{timestamp}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Review', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%', 'Date/Time'])
        for res in selected_results:
            writer.writerow([
                res.get('review', ''),
                res.get('sentiment', ''),
                f"{res.get('confidence', 0):.1f}%",
                f"{res.get('pos_prob', 0):.1f}%",
                f"{res.get('neu_prob', 0):.1f}%",
                f"{res.get('neg_prob', 0):.1f}%",
                res.get('timestamp', '')
            ])

    return filename

def save_all_results():
    if not all_results:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exported_all_results_{timestamp}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Review', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%', 'Date/Time'])
        for res in all_results:
            writer.writerow([
                res.get('review', ''),
                res.get('sentiment', ''),
                f"{res.get('confidence', 0):.1f}%",
                f"{res.get('pos_prob', 0):.1f}%",
                f"{res.get('neu_prob', 0):.1f}%",
                f"{res.get('neg_prob', 0):.1f}%",
                res.get('timestamp', '')
            ])

    return filename

def save_flagged_selected(selected_labels):
    if not selected_labels:
        return None

    selected_files = []
    for label in selected_labels:
        if label in analyzed_label_to_filename:
            selected_files.append(analyzed_label_to_filename[label])
        else:
            match = re.search(r'\[(\d+)\]', label)
            if match:
                idx = int(match.group(1)) - 1
                files = get_flagged_files()
                if 0 <= idx < len(files):
                    selected_files.append(files[idx])

    if not selected_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flagged_export_{timestamp}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Review', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%', 'Date/Time'])

        for file in selected_files:
            filepath = os.path.join(FLAG_PATH, file)
            try:
                parsed = parse_flagged_file(filepath)
                if parsed:
                    writer.writerow([
                        parsed.get('review_full', ''),
                        parsed.get('sentiment', 'Unknown'),
                        f"{parsed.get('confidence', 0):.1f}%",
                        f"{parsed.get('pos_prob', 0):.1f}%",
                        f"{parsed.get('neu_prob', 0):.1f}%",
                        f"{parsed.get('neg_prob', 0):.1f}%",
                        parsed.get('date', '')
                    ])
            except:
                pass

    return filename

def save_flagged_all():
    files = get_flagged_files()
    if not files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flagged_all_export_{timestamp}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Review', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%', 'Date/Time'])

        for file in files:
            filepath = os.path.join(FLAG_PATH, file)
            try:
                parsed = parse_flagged_file(filepath)
                if parsed:
                    writer.writerow([
                        parsed.get('review_full', ''),
                        parsed.get('sentiment', 'Unknown'),
                        f"{parsed.get('confidence', 0):.1f}%",
                        f"{parsed.get('pos_prob', 0):.1f}%",
                        f"{parsed.get('neu_prob', 0):.1f}%",
                        f"{parsed.get('neg_prob', 0):.1f}%",
                        parsed.get('date', '')
                    ])
            except:
                pass

    return filename

def flag_selected_from_table(selected_labels):
    if not selected_labels or not filtered_results_cache:
        return "No results selected."
    selected_results = []
    for label in selected_labels:
        match = re.search(r'\[(\d+)\]', label)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(filtered_results_cache):
                selected_results.append(filtered_results_cache[idx])
    if not selected_results:
        return "No valid results selected."
    existing_reviews = get_existing_flagged_reviews()
    flagged_count = 0
    skipped_count = 0
    for res in selected_results:
        review_text = res.get('review', '').strip()
        timestamp = res.get('timestamp', 'Not specified')
        if not review_text:
            continue
        date_only = 'No date'
        if timestamp and timestamp != 'Not specified':
            date_only = timestamp[:10] if len(timestamp) >= 10 else timestamp
        key = (review_text, date_only)
        if key in existing_reviews:
            skipped_count += 1
            continue
        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"flag_{timestamp_now}.csv"
        filepath = os.path.join(FLAG_PATH, filename)
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Review', 'Result', 'Timestamp', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%'])
                writer.writerow([res.get('review', ''), res.get('full_result', ''), res.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")), res.get('sentiment', 'Unknown'), f"{res.get('confidence', 0):.1f}%", f"{res.get('pos_prob', 0):.1f}%", f"{res.get('neu_prob', 0):.1f}%", f"{res.get('neg_prob', 0):.1f}%"])
            flagged_count += 1
            existing_reviews[key] = filename
        except Exception as e:
            print(f"Error flagging review: {e}")
    if flagged_count == 0 and skipped_count > 0:
        return f"Flagged 0 reviews, {skipped_count} skipped (already exist)"
    elif flagged_count > 0 and skipped_count > 0:
        return f"Flagged {flagged_count} review(s), {skipped_count} skipped (already exist)"
    else:
        return f"Flagged {flagged_count} review(s)!"

def flag_all_from_table():
    if not all_results:
        return "No results to flag."
    existing_reviews = get_existing_flagged_reviews()
    flagged_count = 0
    skipped_count = 0
    for res in all_results:
        review_text = res.get('review', '').strip()
        timestamp = res.get('timestamp', 'Not specified')
        if not review_text:
            continue
        date_only = 'No date'
        if timestamp and timestamp != 'Not specified':
            date_only = timestamp[:10] if len(timestamp) >= 10 else timestamp
        key = (review_text, date_only)
        if key in existing_reviews:
            skipped_count += 1
            continue
        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"flag_{timestamp_now}.csv"
        filepath = os.path.join(FLAG_PATH, filename)
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Review', 'Result', 'Timestamp', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%'])
                writer.writerow([res.get('review', ''), res.get('full_result', ''), res.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")), res.get('sentiment', 'Unknown'), f"{res.get('confidence', 0):.1f}%", f"{res.get('pos_prob', 0):.1f}%", f"{res.get('neu_prob', 0):.1f}%", f"{res.get('neg_prob', 0):.1f}%"])
            flagged_count += 1
            existing_reviews[key] = filename
        except Exception as e:
            print(f"Error flagging review: {e}")
    if flagged_count == 0 and skipped_count > 0:
        return f"Flagged 0 reviews, {skipped_count} skipped (already exist)"
    elif flagged_count > 0 and skipped_count > 0:
        return f"Flagged {flagged_count} review(s), {skipped_count} skipped (already exist)"
    elif flagged_count > 0:
        return f"Flagged all {flagged_count} review(s) from current results!"
    else:
        return "No new reviews to flag (all already exist)"

def flag_low_confidence(selected_labels):
    if not selected_labels or not filtered_results_cache:
        return "No results selected."
    THRESHOLD = 70
    selected_results = []
    for label in selected_labels:
        match = re.search(r'\[(\d+)\]', label)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(filtered_results_cache):
                selected_results.append(filtered_results_cache[idx])
    if not selected_results:
        return "No valid results selected."
    existing_reviews = get_existing_flagged_reviews()
    flagged_count = 0
    skipped_count = 0
    low_conf_count = 0
    for res in selected_results:
        review_text = res.get('review', '').strip()
        timestamp = res.get('timestamp', 'Not specified')
        confidence = res.get('confidence', 0)
        if not review_text:
            continue
        if confidence >= THRESHOLD:
            continue
        low_conf_count += 1
        date_only = 'No date'
        if timestamp and timestamp != 'Not specified':
            date_only = timestamp[:10] if len(timestamp) >= 10 else timestamp
        key = (review_text, date_only)
        if key in existing_reviews:
            skipped_count += 1
            continue
        timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"flag_{timestamp_now}.csv"
        filepath = os.path.join(FLAG_PATH, filename)
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Review', 'Result', 'Timestamp', 'Sentiment', 'Confidence', 'Positive%', 'Neutral%', 'Negative%'])
                writer.writerow([res.get('review', ''), res.get('full_result', ''), res.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")), res.get('sentiment', 'Unknown'), f"{res.get('confidence', 0):.1f}%", f"{res.get('pos_prob', 0):.1f}%", f"{res.get('neu_prob', 0):.1f}%", f"{res.get('neg_prob', 0):.1f}%"])
            flagged_count += 1
            existing_reviews[key] = filename
        except Exception as e:
            print(f"Error flagging review: {e}")
    if low_conf_count == 0:
        return f"No low confidence reviews in selection (threshold: {THRESHOLD}%)"
    elif flagged_count == 0 and skipped_count > 0:
        return f"Found {low_conf_count} low confidence reviews, but {skipped_count} already flagged"
    elif flagged_count > 0 and skipped_count > 0:
        return f"Flagged {flagged_count} low confidence reviews, {skipped_count} already flagged (threshold: {THRESHOLD}%)"
    else:
        return f"Flagged {flagged_count} low confidence reviews (threshold: {THRESHOLD}%)"

def delete_flagged_results(selected_labels):
    if not selected_labels:
        return "No files selected.", [], ""
    selected_files = []
    for label in selected_labels:
        if label in analyzed_label_to_filename:
            selected_files.append(analyzed_label_to_filename[label])
        else:
            match = re.search(r'\[(\d+)\]', label)
            if match:
                idx = int(match.group(1)) - 1
                files = get_flagged_files()
                if 0 <= idx < len(files):
                    selected_files.append(files[idx])
    if not selected_files:
        return "No valid files selected.", [], ""
    deleted_count = 0
    for file in selected_files:
        filepath = os.path.join(FLAG_PATH, file)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_count += 1
        except:
            pass
    display, checkbox_choices = get_flagged_display_table()
    return f"Deleted {deleted_count} flagged file(s).", gr.update(choices=checkbox_choices, value=[]), display

def delete_all_flagged():
    files = get_flagged_files()
    if not files:
        return "No flagged files to delete.", [], ""
    deleted_count = 0
    for file in files:
        filepath = os.path.join(FLAG_PATH, file)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_count += 1
        except:
            pass
    display, checkbox_choices = get_flagged_display_table()
    return f"Deleted all {deleted_count} flagged files.", gr.update(choices=checkbox_choices, value=[]), display

def refresh_flagged():
    display, checkbox_choices = get_flagged_display_table()
    return gr.update(choices=checkbox_choices, value=[]), display

def refresh_analyzed():
    display, checkbox_choices = get_analyzed_results_display()
    return display, gr.update(choices=checkbox_choices, value=[]), "Refreshed successfully!"

def get_existing_flagged_reviews():
    existing_reviews = {}
    files = get_flagged_files()
    for file in files:
        filepath = os.path.join(FLAG_PATH, file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                date_text = 'No date'
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
                if date_match:
                    date_text = date_match.group(1)
                lines = content.strip().split('\n')
                for line in lines:
                    if line and not line.startswith('Review,') and not line.startswith('"Review,'):
                        review_text = None
                        try:
                            if line.startswith('"'):
                                end = line.find('"', 1)
                                if end > 0:
                                    review_text = line[1:end]
                            else:
                                parts = line.split(',', 1)
                                if parts:
                                    review_text = parts[0].strip()
                            if review_text:
                                key = (review_text.strip(), date_text)
                                existing_reviews[key] = file
                                break
                        except:
                            pass
        except:
            pass
    return existing_reviews

# ============ CSV IMPORT FUNCTIONS ============

def detect_columns(df):
    review_col = None
    datetime_col = None
    date_col = None
    time_col = None
    columns = df.columns.tolist()
    review_keywords = ['review', 'text', 'comment', 'feedback', 'message', 'content']
    datetime_keywords = ['datetime', 'timestamp', 'created_at', 'posted_at']
    date_keywords = ['date', 'review_date', 'posted_date', 'created_date']
    time_keywords = ['time', 'review_time', 'posted_time']
    
    for col in columns:
        col_lower = col.lower().strip()
        if 'sentiment' in col_lower or 'label' in col_lower:
            continue
        if review_col is None:
            for keyword in review_keywords:
                if col_lower == keyword or keyword in col_lower:
                    review_col = col
                    break
        if datetime_col is None:
            for keyword in datetime_keywords:
                if col_lower == keyword or keyword in col_lower:
                    datetime_col = col
                    break
        if date_col is None:
            for keyword in date_keywords:
                if col_lower == keyword or keyword in col_lower:
                    date_col = col
                    break
        if time_col is None:
            for keyword in time_keywords:
                if col_lower == keyword or keyword in col_lower:
                    time_col = col
                    break
    
    if date_col is None and datetime_col is not None:
        date_col = datetime_col
        datetime_col = None
    
    if review_col is None:
        for col in columns:
            if 'sentiment' in col.lower() or 'label' in col.lower():
                continue
            if df[col].dtype == 'object':
                sample = df[col].dropna()
                if len(sample) > 0:
                    sample_str = str(sample.iloc[0])
                    if len(sample_str.split()) > 2 or len(sample_str) > 20:
                        review_col = col
                        break
    
    return review_col, datetime_col, date_col, time_col

def display_imported_csv(file):
    if file is None:
        return "No file uploaded yet."
    
    try:
        df = pd.read_csv(file.name)
        if df.empty:
            return "CSV file is empty."
        
        preview_html = """
        <style>
            .import-preview-table {
                width: 100%;
                border-collapse: collapse;
                font-family: "Play", sans-serif;
                font-size: 13px;
                background: #ffffff;
            }
            .import-preview-table th {
                background: #D5C6E0;
                color: #000000 !important;
                padding: 10px;
                text-align: left;
                font-weight: 700;
                border: 1px solid #2A0800 !important;
            }
            .import-preview-table td {
                padding: 8px 10px;
                border: 1px solid #2A0800 !important;
                vertical-align: top;
                background: #ffffff !important;
                color: #000000 !important;
            }
            .import-preview-table tr:hover td {
                background: #f5f0f7 !important;
            }
            .import-preview-container {
                max-height: 400px;
                overflow-y: auto;
                border: 2px solid #2A0800 !important;
                border-radius: 8px;
                background: #ffffff;
            }
            .import-info {
                color: #ffffff !important;
                font-weight: 700;
                padding: 8px 0;
            }
        </style>
        """
        
        preview_html += f"<div class='import-info'>Imported {len(df)} rows, {len(df.columns)} columns</div>"
        
        preview_html += '<div class="import-preview-container"><table class="import-preview-table"><thead><tr>'
        for col in df.columns[:10]:
            preview_html += f'<th>{col}</th>'
        preview_html += '</tr></thead><tbody>'
        
        for i, row in df.head(10).iterrows():
            preview_html += '<tr>'
            for col in df.columns[:10]:
                val = str(row[col])[:50] + "..." if len(str(row[col])) > 50 else str(row[col])
                preview_html += f'<td>{val}</td>'
            preview_html += '</tr>'
        
        preview_html += '</tbody></table></div>'
        
        if len(df) > 10:
            preview_html += f"<div style='color:#ffffff; padding-top: 4px;'>... and {len(df) - 10} more rows</div>"
        
        return preview_html
    except Exception as e:
        return f"Error previewing CSV: {str(e)}"

def import_csv_batch(file):
    """Import and analyze CSV file with reviews"""
    global last_analysis_results, sentiment_data, all_results, all_reviews, sentiment_history, filtered_results_cache
    
    if file is None:
        empty_fig = create_empty_chart_figure()
        charts = create_all_charts()
        analyzed_display, analyzed_choices = get_analyzed_results_display()
        return (
            "No file uploaded.",
            gr.update(choices=analyzed_choices, value=[]),
            analyzed_display,
            empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig, empty_fig,
            "No file uploaded.",
            charts['stats_overview'], charts['stats_time']
        )
    
    try:
        df = pd.read_csv(file.name)
        if df.empty:
            empty_fig = create_empty_chart_figure()
            charts = create_all_charts()
            analyzed_display, analyzed_choices = get_analyzed_results_display()
            return (
                "CSV file is empty.",
                gr.update(choices=analyzed_choices, value=[]),
                analyzed_display,
                empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig,
                "CSV file is empty.",
                charts['stats_overview'], charts['stats_time']
            )
        
        review_col, datetime_col, date_col, time_col = detect_columns(df)
        if review_col is None:
            empty_fig = create_empty_chart_figure()
            charts = create_all_charts()
            analyzed_display, analyzed_choices = get_analyzed_results_display()
            preview = display_imported_csv(file)
            return (
                "Could not find review column. Please check your CSV headers.",
                gr.update(choices=analyzed_choices, value=[]),
                analyzed_display,
                empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig,
                preview,
                charts['stats_overview'], charts['stats_time']
            )
        
        imported_count = 0
        last_analysis_results = []
        
        for idx, row in df.iterrows():
            review_text = str(row[review_col]).strip() if review_col in df.columns else ""
            if not review_text or review_text == 'nan':
                continue
            
            dt = None
            if datetime_col and datetime_col in df.columns:
                dt_str = str(row[datetime_col])
                if dt_str and dt_str != 'nan':
                    dt = parse_datetime_advanced(dt_str)
            
            if dt is None and date_col and date_col in df.columns:
                date_str = str(row[date_col])
                if date_str and date_str != 'nan':
                    time_str = ""
                    if time_col and time_col in df.columns:
                        time_str = str(row[time_col])
                        if time_str and time_str == 'nan':
                            time_str = ""
                    dt = build_datetime_from_parts(date_str, time_str)
            
            if dt is None:
                for col in df.columns:
                    if col != review_col and col not in [datetime_col, date_col, time_col]:
                        try:
                            val = str(row[col])
                            if val and val != 'nan':
                                parsed = parse_datetime_advanced(val)
                                if parsed:
                                    dt = parsed
                                    break
                        except:
                            pass
            
            try:
                sentiment, confidence, full_result, timestamp_str, time_of_day, day_of_year, week_number, month_num, parsed_dt, pos_prob, neu_prob, neg_prob = analyze_review_with_time(review_text, dt)
                
                data_entry = {
                    'review': review_text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': timestamp_str,
                    'time_of_day': time_of_day,
                    'day_of_year': day_of_year,
                    'week_number': week_number,
                    'month': month_num,
                    'datetime': parsed_dt,
                    'emoji': get_sentiment_emoji(sentiment),
                    'full_result': full_result,
                    'pos_prob': pos_prob,
                    'neu_prob': neu_prob,
                    'neg_prob': neg_prob
                }
                
                sentiment_data.append(data_entry)
                all_results.append(data_entry.copy())
                all_reviews.append(review_text[:50] + "..." if len(review_text) > 50 else review_text)
                sentiment_history.append(f"[{timestamp_str}] {sentiment} ({confidence:.1f}%)")
                last_analysis_results.append(data_entry.copy())
                imported_count += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        analyzed_display, analyzed_choices = get_analyzed_results_display()
        filtered_results_cache = all_results.copy()
        charts = create_all_charts()
        preview = display_imported_csv(file)
        
        return (
            f"Imported and analyzed {imported_count} reviews from CSV!",
            gr.update(choices=analyzed_choices, value=[]),
            analyzed_display,
            charts['pie'], charts['conf_bar'], charts['trend'],
            charts['time_day'], charts['day_week'], charts['month'], charts['year'],
            preview,
            charts['stats_overview'], charts['stats_time']
        )
        
    except Exception as e:
        empty_fig = create_empty_chart_figure()
        charts = create_all_charts()
        analyzed_display, analyzed_choices = get_analyzed_results_display()
        return (
            f"Error reading CSV: {str(e)}",
            gr.update(choices=analyzed_choices, value=[]),
            analyzed_display,
            empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig, empty_fig,
            f"Error: {str(e)}",
            charts['stats_overview'], charts['stats_time']
        )

def clear_history():
    global sentiment_data, all_results, all_reviews, sentiment_history, last_analysis_results, filtered_results_cache

    sentiment_history.clear()
    all_reviews.clear()
    all_results.clear()
    last_analysis_results = []
    sentiment_data = []
    filtered_results_cache = []

    analyzed_display, analyzed_choices = get_analyzed_results_display()
    charts = create_all_charts()

    return (
        "Cleared all history!",
        "Recent History:\nNo history yet.",
        "History cleared!",
        analyzed_display,
        gr.update(choices=analyzed_choices),
        charts['pie'], charts['conf_bar'], charts['trend'],
        charts['time_day'], charts['day_week'], charts['month'], charts['year'],
        charts['stats_overview'], charts['stats_time']
    )

def refresh_all_charts():
    charts = create_all_charts()
    return (
        charts['pie'], charts['conf_bar'], charts['trend'],
        charts['time_day'], charts['day_week'], charts['month'], charts['year'],
        charts['stats_overview'], charts['stats_time']
    )

def execute_reset():
    """Execute reset and return all outputs"""
    global last_analysis_results, sentiment_history, all_reviews, all_results, sentiment_data, filtered_results_cache

    # Clear all data
    sentiment_history.clear()
    all_reviews.clear()
    all_results.clear()
    last_analysis_results = []
    sentiment_data = []
    filtered_results_cache = []

    # Delete flagged files
    files = get_flagged_files()
    for f in files:
        try:
            os.remove(os.path.join(FLAG_PATH, f))
        except:
            pass

    # Get updated displays
    analyzed_display, analyzed_choices = get_analyzed_results_display()
    flagged_display, flagged_choices = get_flagged_display_table()
    charts = create_all_charts()

    # Return all outputs in the correct order
    return (
        "All data has been reset successfully!",
        "Recent History:\nNo history yet.",
        flagged_display,
        gr.update(choices=flagged_choices, value=[]),
        analyzed_display,
        gr.update(choices=analyzed_choices, value=[]),
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        charts['pie'],
        charts['conf_bar'],
        charts['trend'],
        charts['time_day'],
        charts['day_week'],
        charts['month'],
        charts['year'],
        charts['stats_overview'],
        charts['stats_time']
    )

def confirm_reset():
    """Show confirmation row"""
    return gr.update(visible=True), gr.update(visible=False)

def cancel_reset():
    """Hide confirmation row"""
    return gr.update(visible=False), gr.update(visible=True)

def filter_analyzed_results(sentiment_filter):
    global filtered_results_cache
    if not all_results:
        return "No analyzed results yet.", []
    if sentiment_filter == "All":
        filtered_results_cache = all_results.copy()
    else:
        filtered_results_cache = [r for r in all_results if r.get('sentiment') == sentiment_filter]
    if not filtered_results_cache:
        return f"No {sentiment_filter} results found.", []
    
    html = """
    <style>
        .results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .results-table th { background: #D5C6E0; color: #000000 !important; padding: 10px; text-align: left; font-weight: 700; border: 1px solid #2A0800 !important; }
        .results-table td { padding: 8px 10px; border: 1px solid #2A0800 !important; vertical-align: top; background: #ffffff; }
        .results-table tr:hover td { background: #f5f0f7 !important; }
        .results-container { max-height: 400px; overflow-y: auto; border: 2px solid #2A0800 !important; border-radius: 8px; background: #ffffff; }
        .table-total { color: #ffffff !important; font-weight: 700; }
        .sentiment-pos { color: #2ecc71 !important; font-weight: 700; }
        .sentiment-neu { color: #f1c40f !important; font-weight: 700; }
        .sentiment-neg { color: #e74c3c !important; font-weight: 700; }
        .conf-high { color: #2ecc71 !important; font-weight: 700; }
        .conf-low { color: #e74c3c !important; font-weight: 700; }
        .review-preview { max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #000000 !important; }
        .review-preview:hover { white-space: normal; overflow: visible; background: #ffffff; position: relative; z-index: 20; box-shadow: 0 2px 8px rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px; border: 1px solid #2A0800; }
    </style>
    <div class="results-container">
        <table class="results-table">
            <thead>
                <tr>
                    <th style="width:40px;">#</th>
                    <th style="width:100px;">Sentiment</th>
                    <th style="width:100px;">Confidence</th>
                    <th style="width:80px;">Positive</th>
                    <th style="width:80px;">Neutral</th>
                    <th style="width:80px;">Negative</th>
                    <th style="width:150px;">Date/Time</th>
                    <th>Review</th>
                </tr>
            </thead>
            <tbody>
    """
    for i, res in enumerate(filtered_results_cache, 1):
        sentiment = res.get('sentiment', 'Unknown')
        emoji = get_sentiment_emoji(sentiment)
        confidence = res.get('confidence', 0)
        pos_prob = res.get('pos_prob', 0)
        neu_prob = res.get('neu_prob', 0)
        neg_prob = res.get('neg_prob', 0)
        timestamp = res.get('timestamp', 'Not specified')
        review = res.get('review', '')
        preview = review[:60] + "..." if len(review) > 60 else review
        sent_class = 'sentiment-pos' if sentiment == 'Positive' else 'sentiment-neu' if sentiment == 'Neutral' else 'sentiment-neg'
        conf_class = 'conf-high' if confidence >= 70 else 'conf-low'
        html += f"""
            <tr>
                <td style="color:#000000 !important; text-align:center;">{i}</td>
                <td class="{sent_class}">{emoji} {sentiment}</td>
                <td class="{conf_class}">{confidence:.1f}%</td>
                <td style="color:#000000 !important;">{pos_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neu_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neg_prob:.1f}%</td>
                <td style="color:#000000 !important;">{timestamp}</td>
                <td class="review-preview" title="{review}" style="color:#000000 !important;">{preview}</td>
            </tr>
        """
    html += """
            </tbody>
        </table>
    </div>
    """
    html += f"<br><span class='table-total'>Total: {len(filtered_results_cache)} results (filtered: {sentiment_filter})</span>"
    checkbox_choices = []
    for i, res in enumerate(filtered_results_cache, 1):
        timestamp = res.get('timestamp', 'No date')
        date_part = timestamp.split(' ')[0] if ' ' in timestamp else timestamp[:10]
        checkbox_choices.append(f"[{i}] {res.get('sentiment', 'Unknown')} - {date_part}")
    return html, gr.update(choices=checkbox_choices, value=checkbox_choices)

def filter_flagged_results(sentiment_filter):
    global filtered_flagged_cache
    files = get_flagged_files()
    if not files:
        return "No flagged data yet.", []
    parsed_files = []
    for file in files:
        filepath = os.path.join(FLAG_PATH, file)
        parsed = parse_flagged_file(filepath)
        if parsed:
            parsed_files.append(parsed)
    if not parsed_files:
        return "No flagged data yet.", []
    if sentiment_filter == "All":
        filtered_flagged_cache = parsed_files
    else:
        filtered_flagged_cache = [f for f in parsed_files if f.get('sentiment') == sentiment_filter]
    if not filtered_flagged_cache:
        return f"No {sentiment_filter} flagged files found.", []
    
    html = """
    <style>
        .flagged-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .flagged-table th { background: #D5C6E0; color: #000000 !important; padding: 10px; text-align: left; font-weight: 700; border: 1px solid #2A0800 !important; }
        .flagged-table td { padding: 8px 10px; border: 1px solid #2A0800 !important; vertical-align: top; background: #ffffff; }
        .flagged-table tr:hover td { background: #f5f0f7 !important; }
        .flagged-container { max-height: 400px; overflow-y: auto; border: 2px solid #2A0800 !important; border-radius: 8px; background: #ffffff; }
        .table-total { color: #ffffff !important; font-weight: 700; }
        .sentiment-pos { color: #2ecc71 !important; font-weight: 700; }
        .sentiment-neu { color: #f1c40f !important; font-weight: 700; }
        .sentiment-neg { color: #e74c3c !important; font-weight: 700; }
        .conf-high { color: #2ecc71 !important; font-weight: 700; }
        .conf-low { color: #e74c3c !important; font-weight: 700; }
        .review-preview { max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #000000 !important; }
        .review-preview:hover { white-space: normal; overflow: visible; background: #ffffff; position: relative; z-index: 20; box-shadow: 0 2px 8px rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px; border: 1px solid #2A0800; }
    </style>
    <div class="flagged-container">
        <table class="flagged-table">
            <thead>
                <tr>
                    <th style="width:40px;">#</th>
                    <th style="width:100px;">Sentiment</th>
                    <th style="width:100px;">Confidence</th>
                    <th style="width:80px;">Positive</th>
                    <th style="width:80px;">Neutral</th>
                    <th style="width:80px;">Negative</th>
                    <th style="width:180px;">Date/Time</th>
                    <th>Review</th>
                </tr>
            </thead>
            <tbody>
    """
    for i, parsed in enumerate(filtered_flagged_cache, 1):
        sentiment = parsed.get('sentiment', 'Unknown')
        emoji = get_sentiment_emoji(sentiment)
        confidence = parsed.get('confidence', 0)
        pos_prob = parsed.get('pos_prob', 0)
        neu_prob = parsed.get('neu_prob', 0)
        neg_prob = parsed.get('neg_prob', 0)
        date_display = parsed.get('date', 'Unknown')
        review = parsed.get('review_preview', '')
        review_full = parsed.get('review_full', '')
        sent_class = 'sentiment-pos' if sentiment == 'Positive' else 'sentiment-neu' if sentiment == 'Neutral' else 'sentiment-neg'
        conf_class = 'conf-high' if confidence >= 70 else 'conf-low'
        html += f"""
            <tr>
                <td style="color:#000000 !important; text-align:center;">{i}</td>
                <td class="{sent_class}">{emoji} {sentiment}</td>
                <td class="{conf_class}">{confidence:.1f}%</td>
                <td style="color:#000000 !important;">{pos_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neu_prob:.1f}%</td>
                <td style="color:#000000 !important;">{neg_prob:.1f}%</td>
                <td style="color:#000000 !important;">{date_display}</td>
                <td class="review-preview" title="{review_full}" style="color:#000000 !important;">{review}</td>
            </tr>
        """
    html += """
            </tbody>
        </table>
    </div>
    """
    html += f"<br><span class='table-total'>Total: {len(filtered_flagged_cache)} flagged files (filtered: {sentiment_filter})</span>"
    checkbox_choices = []
    for i, parsed in enumerate(filtered_flagged_cache, 1):
        sentiment = parsed.get('sentiment', 'Unknown')
        date_display = parsed.get('date', 'Unknown')
        short_date = date_display[:10] if len(date_display) >= 10 else date_display
        checkbox_choices.append(f"[{i}] {sentiment} - {short_date}")
    return html, gr.update(choices=checkbox_choices, value=checkbox_choices)

def predict_with_time_wrapper(review_text, date_input, time_input):
    try:
        if not review_text or not review_text.strip():
            charts = create_all_charts()
            default_table, default_choices = get_analyzed_results_display()
            return ("Please enter a review.", "No history.", "", default_table, gr.update(choices=default_choices),
                    charts['pie'], charts['conf_bar'], charts['trend'],
                    charts['time_day'], charts['day_week'], charts['month'], charts['year'],
                    charts['stats_overview'], charts['stats_time'])

        review = review_text.strip()
        dt = None
        if date_input and date_input.strip():
            dt = build_datetime_from_parts(date_input, time_input)

        sentiment, confidence, full_result, timestamp_str, time_of_day, day_of_year, week_number, month_num, parsed_dt, pos_prob, neu_prob, neg_prob = analyze_review_with_time(review, dt)

        sentiment_history.append(f"[{timestamp_str}] {sentiment} ({confidence:.1f}%)")
        preview = review[:50] + "..." if len(review) > 50 else review
        all_reviews.append(preview)

        data_entry = {
            'review': review,
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': timestamp_str,
            'time_of_day': time_of_day,
            'day_of_year': day_of_year,
            'week_number': week_number,
            'month': month_num,
            'datetime': parsed_dt,
            'emoji': get_sentiment_emoji(sentiment),
            'full_result': full_result,
            'pos_prob': pos_prob,
            'neu_prob': neu_prob,
            'neg_prob': neg_prob
        }

        sentiment_data.append(data_entry)
        all_results.append(data_entry.copy())
        last_analysis_results.append(data_entry.copy())

        output = full_result

        if len(sentiment_data) > 1:
            summary = generate_summary_text()
            output = summary + "\n\n" + output

        history_text = "Recent History:\n"
        if sentiment_history:
            history_text += "\n".join(sentiment_history[-20:])
            history_text += f"\n\nTotal: {len(sentiment_history)} reviews"
        else:
            history_text += "No history yet."

        analyzed_display, analyzed_choices = get_analyzed_results_display()
        filtered_results_cache = all_results.copy()
        charts = create_all_charts()

        return (output, history_text, "", analyzed_display, gr.update(choices=analyzed_choices),
                charts['pie'], charts['conf_bar'], charts['trend'],
                charts['time_day'], charts['day_week'], charts['month'], charts['year'],
                charts['stats_overview'], charts['stats_time'])

    except Exception as e:
        empty_fig = create_empty_chart_figure()
        return (f"Error: {str(e)}", "Error occurred.", "", "", gr.update(choices=[]),
                empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig,
                "Error loading stats", "Error loading stats")

# ============ CUSTOM CSS ============
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Play:wght@400;700&display=swap');
* { font-family: "Play", sans-serif !important; }
body { background: #967AA1 !important; }
.gradio-container { max-width: 1400px; width: 95%; margin: 20px auto; background: #967AA1; padding: 16px; border-radius: 12px; }
.gradio-interface { background: #967AA1; border-radius: 12px !important; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
h1 { color: #D8DCFF !important; font-family: "Bebas Neue", sans-serif !important; font-weight: 400 !important; font-size: 3rem !important; text-align: center !important; margin-bottom: 2px; letter-spacing: 2px; width: 100% !important; }
h1 + p { color: #AEADF0 !important; text-align: center !important; font-size: 1rem !important; margin-top: -2px; font-weight: 700 !important; width: 100% !important; }
.gr-textbox .gr-textbox-input, .gr-textbox textarea, .gradio-textbox textarea { background: #ffffff !important; border: 1.5px solid #D8DCFF !important; border-radius: 8px !important; padding: 12px !important; font-size: 14px !important; color: #000000 !important; }
.gr-textbox .gr-textbox-input:focus, .gr-textbox textarea:focus { border-color: #D8DCFF !important; box-shadow: 0 0 0 3px rgba(216, 220, 255, 0.2) !important; }
textarea { background: #ffffff !important; border: 1.5px solid #D8DCFF !important; border-radius: 8px !important; padding: 10px !important; font-size: 14px !important; color: #000000 !important; min-height: 80px !important; }
textarea:focus { border-color: #D8DCFF !important; box-shadow: 0 0 0 3px rgba(216, 220, 255, 0.2) !important; }
textarea::placeholder { color: #999999 !important; }
.gr-html { background: #ffffff !important; border: 1.5px solid #D8DCFF !important; border-radius: 8px !important; padding: 0 !important; font-size: 14px !important; color: #000000 !important; line-height: 1.6 !important; max-height: 500px !important; overflow-y: auto !important; }
.gr-html pre { margin: 0 !important; padding: 12px !important; background: #ffffff !important; color: #000000 !important; font-family: "Play", monospace !important; font-size: 13px !important; white-space: pre-wrap !important; word-wrap: break-word !important; }
.gr-textbox .gr-textbox-input { background: #ffffff !important; border: 1.5px solid #D8DCFF !important; border-radius: 8px !important; padding: 12px !important; font-size: 14px !important; color: #000000 !important; }
#history_output textarea, #analyzed_display textarea, #flagged_output textarea { overflow-y: auto !important; resize: vertical !important; max-height: 300px !important; min-height: 150px !important; }
.gr-textbox textarea::-webkit-scrollbar, .gr-textbox .gr-textbox-input::-webkit-scrollbar { width: 8px; }
.gr-textbox textarea::-webkit-scrollbar-track, .gr-textbox .gr-textbox-input::-webkit-scrollbar-track { background: #e9ecef; border-radius: 4px; }
.gr-textbox textarea::-webkit-scrollbar-thumb, .gr-textbox .gr-textbox-input::-webkit-scrollbar-thumb { background: #D5C6E0; border-radius: 4px; }
.gr-textbox textarea::-webkit-scrollbar-thumb:hover, .gr-textbox .gr-textbox-input::-webkit-scrollbar-thumb:hover { background: #C38D94; }
.gr-textbox[data-testid="textbox"] { background: transparent !important; border: none !important; box-shadow: none !important; padding: 4px 0 !important; font-size: 13px !important; color: #000000 !important; min-height: auto !important; max-height: none !important; }
.gr-textbox[data-testid="textbox"] textarea, .gr-textbox[data-testid="textbox"] input { background: transparent !important; border: none !important; box-shadow: none !important; color: #000000 !important; padding: 4px 0 !important; font-size: 13px !important; min-height: auto !important; }
.status-container { margin-top: 8px; padding: 4px 0; }
button { border: none !important; padding: 8px 18px !important; font-size: 14px !important; font-weight: 700 !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
button.primary, button[data-testid="primary"], button#analyze_btn { background: #D5C6E0 !important; color: #2A0800 !important; }
button.primary:hover, button[data-testid="primary"]:hover { background: #D8DCFF !important; color: #2A0800 !important; }
button.secondary, button[data-testid="secondary"] { background: #565676 !important; color: #D8DCFF !important; }
button.secondary:hover, button[data-testid="secondary"]:hover { background: #D8DCFF !important; color: #565676 !important; }
button[id*="delete"], button[id*="Delete"], button:contains("Delete"), button:contains("Delete Selected"), button:contains("Delete All"), button:contains("Reset"), button:contains("Reset All") { background: #e74c3c !important; color: #ffffff !important; }
button[id*="delete"]:hover, button[id*="Delete"]:hover, button:contains("Delete"):hover, button:contains("Delete Selected"):hover, button:contains("Delete All"):hover, button:contains("Reset"):hover, button:contains("Reset All"):hover { background: #c0392b !important; color: #ffffff !important; }
button[id*="save"], button[id*="Save"], button:contains("Save"), button:contains("Save Selected"), button:contains("Save All") { background: #27ae60 !important; color: #ffffff !important; }
button[id*="save"]:hover, button[id*="Save"]:hover, button:contains("Save"):hover, button:contains("Save Selected"):hover, button:contains("Save All"):hover { background: #229954 !important; color: #ffffff !important; }
button:contains("Flag"), button:contains("Flag Selected"), button:contains("Flag All") { background: #D5C6E0 !important; color: #2A0800 !important; }
button:contains("Flag"):hover, button:contains("Flag Selected"):hover, button:contains("Flag All"):hover { background: #D8DCFF !important; color: #2A0800 !important; }
button:contains("Clear"), button:contains("Refresh"), button:contains("Random"), button:contains("Select"), button:contains("Deselect"), button:contains("Cancel") { background: #D5C6E0 !important; color: #2A0800 !important; }
button:contains("Clear"):hover, button:contains("Refresh"):hover, button:contains("Random"):hover, button:contains("Select"):hover, button:contains("Deselect"):hover, button:contains("Cancel"):hover { background: #D8DCFF !important; color: #2A0800 !important; }
label { font-weight: 700 !important; color: #D8DCFF !important; font-size: 13px !important; margin-bottom: 4px !important; }
.gr-checkbox-group { max-height: 250px !important; overflow-y: auto !important; display: block !important; border: 1.5px solid #AEADF0 !important; border-radius: 10px !important; padding: 12px !important; background: #ffffff !important; }
.gr-checkbox-group::-webkit-scrollbar { width: 6px; }
.gr-checkbox-group::-webkit-scrollbar-track { background: #e9ecef; border-radius: 4px; }
.gr-checkbox-group::-webkit-scrollbar-thumb { background: #D5C6E0; border-radius: 4px; }
.gr-checkbox-group::-webkit-scrollbar-thumb:hover { background: #C38D94; }
.gr-checkbox-group label { padding: 4px 8px !important; margin: 2px 0 !important; font-size: 13px !important; color: #2A0800 !important; }
.gr-checkbox-group input[type="checkbox"] { margin-right: 8px !important; }
hr { border-color: #AEADF0 !important; opacity: 0.2 !important; }
footer { text-align: center !important; margin-top: 16px !important; color: #AEADF0 !important; font-size: 12px !important; }
.gr-row { gap: 16px !important; }
.gr-column { padding: 2px !important; }
.gr-box { padding: 10px !important; }
.gr-textbox .gr-textbox-input, .gr-textbox textarea, .gr-textbox .gr-textbox-input *, .gr-textbox textarea * { color: #000000 !important; }
.gr-markdown p, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4, .gr-markdown h5, .gr-markdown h6, .gr-markdown { color: #000000 !important; }
.gr-markdown h1 { font-family: "Bebas Neue", sans-serif !important; font-weight: 400 !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
.js-plotly-plot .plotly .main-svg .bg { fill: transparent !important; }
.results-table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; }
.results-table th { background: #D5C6E0; color: #000000 !important; padding: 10px; text-align: left; font-weight: 700; border: 1px solid #2A0800 !important; }
.results-table td { padding: 10px; border: 1px solid #2A0800 !important; background: #ffffff; }
.results-table tr:hover td { background: #f5f0f7 !important; }
.sentiment-pos { color: #2ecc71 !important; font-weight: 700; }
.sentiment-neu { color: #f1c40f !important; font-weight: 700; }
.sentiment-neg { color: #e74c3c !important; font-weight: 700; }
.conf-high { color: #2ecc71 !important; font-weight: 700; }
.conf-low { color: #e74c3c !important; font-weight: 700; }
.flagged-table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; }
.flagged-table th { background: #D5C6E0; color: #000000 !important; padding: 10px; text-align: left; font-weight: 700; border: 1px solid #2A0800 !important; }
.flagged-table td { padding: 10px; border: 1px solid #2A0800 !important; background: #ffffff; }
.flagged-table tr:hover td { background: #f5f0f7 !important; }
.results-container, .flagged-container { max-height: 400px; overflow-y: auto; border: 2px solid #2A0800 !important; border-radius: 8px; background: #ffffff; }
.table-total { color: #ffffff !important; font-weight: 700; }
.click-hint { text-align: center; color: #000000 !important; font-size: 12px; padding: 4px; font-style: italic; margin-bottom: 8px; }
.selection-row { display: flex; gap: 16px; align-items: flex-start; }
.selection-left { flex: 2; min-width: 0; }
.selection-right { flex: 1; min-width: 150px; }
.reset-status-text { color: #000000 !important; background: transparent !important; border: none !important; }
.date-input .gr-textbox-input, .time-input .gr-textbox-input, .date-input textarea, .time-input textarea { min-height: 30px !important; max-height: 30px !important; padding: 4px 10px !important; font-size: 13px !important; border: 1.5px solid #D8DCFF !important; border-radius: 8px !important; background: #ffffff !important; color: #000000 !important; overflow-y: auto !important; resize: vertical !important; line-height: 1.3 !important; }
.date-input .gr-textbox-input:focus, .time-input .gr-textbox-input:focus, .date-input textarea:focus, .time-input textarea:focus { border-color: #D8DCFF !important; box-shadow: 0 0 0 3px rgba(216, 220, 255, 0.2) !important; }
.date-input textarea::-webkit-scrollbar, .time-input textarea::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
.date-input textarea, .time-input textarea { scrollbar-width: none !important; -ms-overflow-style: none !important; }

.status-textbox {
    background: #000000 !important;
    border: 2px solid #D5C6E0 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    min-height: 50px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

.status-textbox textarea,
.status-textbox .gr-textbox-input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding: 4px 0 !important;
    min-height: auto !important;
    max-height: none !important;
}

.status-textbox-small {
    background: #000000 !important;
    border: 1.5px solid #D5C6E0 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    min-height: 36px !important;
}

.status-textbox-small textarea,
.status-textbox-small .gr-textbox-input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: center !important;
    padding: 2px 0 !important;
    min-height: auto !important;
}

.tab-nav button { cursor: pointer !important; }
.tabs { display: block !important; }
.gr-checkbox-group { transition: none !important; }
"""

# ============ BUILD THE GRADIO UI ============

if all_results and not filtered_results_cache:
    filtered_results_cache = all_results.copy()

initial_flagged_display, initial_flagged_choices = get_flagged_display_table()
initial_analyzed_display, initial_analyzed_choices = get_analyzed_results_display()
initial_charts = create_all_charts()

with gr.Blocks(title="Cinema Sentiment Analysis", css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown("# 🎥 Cinema Sentiment Analysis 🎬")
            gr.Markdown("<p style='text-align: center; font-weight: bold; font-size: 1.1rem;'>Lights, Camera, Analyze! Track review sentiment across time!</p>")
        with gr.Column(scale=1, min_width=100):
            reset_btn = gr.Button("Reset All", variant="secondary", size="sm", elem_id="reset-btn")
            reset_status = gr.Textbox(visible=False)
            with gr.Row(visible=False) as confirm_row:
                confirm_yes_btn = gr.Button("Yes", variant="primary", size="sm")
                confirm_no_btn = gr.Button("Cancel", variant="secondary", size="sm")

    with gr.Tabs() as tabs:
        # TAB 1: Main Analysis
        with gr.TabItem("Main Analysis", id="tab_main"):
            gr.Markdown("<b>Paste a review below and get instant sentiment analysis</b>")
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Enter Cinema Review", placeholder="Enter a review to analyze", lines=6)
                    with gr.Row():
                        date_input = gr.Textbox(label="Date", placeholder="e.g., 6/28/2025 or 2025-06-28", lines=1, scale=2, elem_classes="date-input")
                        time_input = gr.Textbox(label="Time", placeholder="e.g., 9:36:11 AM or 09:36:11", lines=1, scale=1, elem_classes="time-input")
                    with gr.Row():
                        submit_btn = gr.Button("Analyze", variant="primary")
                        random_btn = gr.Button("Random Example")
                        clear_btn = gr.Button("Clear History")
                    output = gr.Textbox(label="Analysis Results", lines=15, interactive=False)
                with gr.Column(scale=1):
                    history_output = gr.Textbox(label="Recent History", lines=8, interactive=False, max_lines=999, elem_id="history_output")

        # TAB 2: CSV Import
        with gr.TabItem("Batch Upload", id="tab_csv"):
            gr.Markdown("<b>Upload a CSV file with reviews to analyze in batch</b>")
            with gr.Row():
                with gr.Column(scale=2):
                    csv_file = gr.File(label="Upload CSV File", file_types=[".csv"])
                    import_btn = gr.Button("Import and Analyze", variant="primary")
                with gr.Column(scale=1):
                    csv_status = gr.Textbox(label="Status", lines=2, interactive=False, max_lines=2, elem_classes="status-textbox")
            gr.Markdown("---")
            gr.Markdown("### CSV Preview")
            csv_preview = gr.HTML(label="CSV Preview", value="Upload a CSV file to see a preview here.")
            gr.Markdown("""
            **CSV Format Guide:**
            - Column headers can be: 'review', 'text', 'comment', 'feedback', etc.
            - Optional datetime column: 'datetime', 'timestamp', 'date', 'time', etc.
            """)

        # TAB 3: Manage Analyzed Results
        with gr.TabItem("Manage Analyzed Results", id="tab_analyzed"):
            gr.Markdown("**All analyzed reviews with their sentiment scores**")
            with gr.Row():
                with gr.Column(scale=2):
                    filter_dropdown = gr.Dropdown(label="Filter by Sentiment", choices=["All", "Positive", "Neutral", "Negative"], value="All", interactive=True)
            analyzed_table = gr.HTML(label="Analyzed Results Table", value=initial_analyzed_display)
            gr.Markdown("---")
            gr.Markdown("**Select rows below to manage (check the boxes)**")
            with gr.Row():
                with gr.Column(scale=2):
                    analyzed_checkbox = gr.CheckboxGroup(label="Select results to manage", choices=initial_analyzed_choices, interactive=True, elem_id="analyzed_checkbox")
                    with gr.Row():
                        select_all_analyzed_btn = gr.Button("Select All", size="sm", scale=1)
                        deselect_all_analyzed_btn = gr.Button("Deselect All", size="sm", scale=1)
                with gr.Column(scale=1):
                    gr.Markdown("**Actions**")
                    with gr.Row():
                        delete_selected_btn = gr.Button("Delete Selected", variant="secondary", scale=1)
                    with gr.Row():
                        delete_all_btn = gr.Button("Delete All", variant="secondary", scale=1)
                        save_selected_btn = gr.Button("Save Selected", variant="secondary", scale=1)
                    with gr.Row():
                        save_all_btn = gr.Button("Save All", variant="secondary", scale=1)
                        flag_selected_btn = gr.Button("Flag Selected", variant="secondary", scale=1)
                    with gr.Row():
                        flag_all_btn = gr.Button("Flag All", variant="secondary", scale=1)
                        flag_low_btn = gr.Button("Flag Low Conf", variant="secondary", scale=1)
                    analyzed_status = gr.Textbox(label="", lines=1, placeholder="Check boxes below the table, then use the buttons above", interactive=False, max_lines=1, elem_classes="status-textbox")
                    download_output = gr.File(label="Download CSV", visible=True)

        # TAB 4: Flagged Data
        with gr.TabItem("Flagged Data", id="tab_flagged"):
            gr.Markdown("**All analyzed reviews with their sentiment scores**")
            with gr.Row():
                with gr.Column(scale=2):
                    flagged_filter_dropdown = gr.Dropdown(label="Filter by Sentiment", choices=["All", "Positive", "Neutral", "Negative"], value="All", interactive=True)
            flagged_table = gr.HTML(label="Flagged Data Table", value=initial_flagged_display)
            gr.Markdown("---")
            gr.Markdown("**Select rows below to manage (check the boxes)**")
            with gr.Row():
                with gr.Column(scale=2):
                    flagged_checkbox = gr.CheckboxGroup(label="Select flagged files to manage", choices=initial_flagged_choices, interactive=True, elem_id="flagged_checkbox")
                    with gr.Row():
                        select_all_flagged_btn = gr.Button("Select All", size="sm", scale=1)
                        deselect_all_flagged_btn = gr.Button("Deselect All", size="sm", scale=1)
                with gr.Column(scale=1):
                    gr.Markdown("**Actions**")
                    with gr.Row():
                        delete_flagged_btn = gr.Button("Delete Selected", variant="secondary", scale=1)
                    with gr.Row():
                        delete_all_flagged_btn = gr.Button("Delete All", variant="secondary", scale=1)
                        save_flagged_selected_btn = gr.Button("Save Selected", variant="secondary", scale=1)
                    with gr.Row():
                        save_flagged_all_btn = gr.Button("Save All", variant="secondary", scale=1)
                    flagged_status = gr.Textbox(label="", lines=1, placeholder="Check boxes below the table, then use the buttons above", interactive=False, max_lines=1, elem_classes="status-textbox")
                    flagged_download_output = gr.File(label="Download CSV", visible=True)

        # TAB 5: Summary & Visualizations
        with gr.TabItem("Data Visualization", id="tab_summary"):
            gr.Markdown("<b>Interactive charts organized by category for better insights</b>")

            with gr.Tabs() as summary_sub_tabs:
                with gr.TabItem("Sentiment Overview", id="subtab_overview"):
                    gr.Markdown("### Sentiment Distribution & Trends")
                    gr.Markdown("Overview of sentiment patterns across your reviews.")

                    with gr.Row():
                        with gr.Column(scale=1):
                            sentiment_pie = gr.Plot(label="Sentiment Distribution", value=initial_charts['pie'])
                        with gr.Column(scale=1):
                            confidence_bar = gr.Plot(label="Confidence Levels", value=initial_charts['conf_bar'])

                    with gr.Row():
                        sentiment_trend = gr.Plot(label="Sentiment Trends Over Time", value=initial_charts['trend'])

                    stats_overview = gr.HTML(label="Overview Statistics", value=initial_charts['stats_overview'])

                with gr.TabItem("Time Analysis", id="subtab_time"):
                    gr.Markdown("### Time-Based Sentiment Analysis")
                    gr.Markdown("How sentiment varies across different time periods.")

                    with gr.Row():
                        with gr.Column(scale=1):
                            time_of_day_chart = gr.Plot(label="Sentiment by Time of Day", value=initial_charts['time_day'])
                        with gr.Column(scale=1):
                            day_of_week_chart = gr.Plot(label="Sentiment by Day of Week", value=initial_charts['day_week'])

                    with gr.Row():
                        with gr.Column(scale=1):
                            month_chart = gr.Plot(label="Sentiment by Month", value=initial_charts['month'])
                        with gr.Column(scale=1):
                            year_chart = gr.Plot(label="Sentiment by Year", value=initial_charts['year'])

                    stats_time = gr.HTML(label="Time Statistics", value=initial_charts['stats_time'])

    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    Group 8 - TNL6323 Natural Language Processing
    | Cheng Jing Yi 242UT244B7
    | Thomas Tan Yi Wei 243UT245LB
    | Enzo Joaquin Itona Erfe 1231300995
    | Michelle Theba 1231302828

    **Domain:** Movie/Entertainment Reviews (Malaysia Context)
    **Project:** Cinema Sentiment Analysis System
    """)
    gr.Markdown("---")
    gr.Markdown("2026 Group 8 | TNL6323 Natural Language Processing")

    # ============ EVENT HANDLERS ============

    # Submit button
    submit_btn.click(
        fn=predict_with_time_wrapper,
        inputs=[text_input, date_input, time_input],
        outputs=[
            output, history_output, analyzed_status, analyzed_table, analyzed_checkbox,
            sentiment_pie, confidence_bar, sentiment_trend,
            time_of_day_chart, day_of_week_chart, month_chart, year_chart,
            stats_overview, stats_time
        ]
    )

    # Random button
    random_btn.click(fn=random_example, outputs=[text_input, date_input, time_input])

    # Clear button
    clear_btn.click(
        fn=clear_history,
        outputs=[
            output, history_output, analyzed_status, analyzed_table, analyzed_checkbox,
            sentiment_pie, confidence_bar, sentiment_trend,
            time_of_day_chart, day_of_week_chart, month_chart, year_chart,
            stats_overview, stats_time
        ]
    )

    # Import button
    import_btn.click(
        fn=import_csv_batch,
        inputs=[csv_file],
        outputs=[
            csv_status, analyzed_checkbox, analyzed_table,
            sentiment_pie, confidence_bar, sentiment_trend,
            time_of_day_chart, day_of_week_chart, month_chart, year_chart,
            csv_preview, stats_overview, stats_time
        ]
    )

    csv_file.change(fn=display_imported_csv, inputs=[csv_file], outputs=[csv_preview])

    filter_dropdown.change(fn=filter_analyzed_results, inputs=[filter_dropdown], outputs=[analyzed_table, analyzed_checkbox])

    select_all_analyzed_btn.click(fn=select_all_analyzed, outputs=[analyzed_checkbox])
    deselect_all_analyzed_btn.click(fn=deselect_all_analyzed, outputs=[analyzed_checkbox])

    delete_selected_btn.click(
        fn=delete_selected_from_table,
        inputs=[analyzed_checkbox],
        outputs=[analyzed_status]
    ).then(fn=refresh_analyzed, outputs=[analyzed_table, analyzed_checkbox, analyzed_status])

    delete_all_btn.click(
        fn=delete_all_analyzed,
        outputs=[analyzed_status]
    ).then(fn=refresh_analyzed, outputs=[analyzed_table, analyzed_checkbox, analyzed_status])

    # SAVE BUTTONS (Downloads CSV)
    save_selected_btn.click(
        fn=save_selected_results_from_table,
        inputs=[analyzed_checkbox],
        outputs=[download_output]
    )

    save_all_btn.click(
        fn=save_all_results,
        outputs=[download_output]
    )

    flag_selected_btn.click(
        fn=flag_selected_from_table,
        inputs=[analyzed_checkbox],
        outputs=[analyzed_status]
    ).then(fn=refresh_flagged, outputs=[flagged_checkbox, flagged_table])

    flag_all_btn.click(
        fn=flag_all_from_table,
        outputs=[analyzed_status]
    ).then(fn=refresh_flagged, outputs=[flagged_checkbox, flagged_table])

    flag_low_btn.click(
        fn=flag_low_confidence,
        inputs=[analyzed_checkbox],
        outputs=[analyzed_status]
    ).then(fn=refresh_flagged, outputs=[flagged_checkbox, flagged_table])

    flagged_filter_dropdown.change(
        fn=filter_flagged_results,
        inputs=[flagged_filter_dropdown],
        outputs=[flagged_table, flagged_checkbox]
    )

    select_all_flagged_btn.click(fn=select_all_flagged, outputs=[flagged_checkbox])
    deselect_all_flagged_btn.click(fn=deselect_all_flagged, outputs=[flagged_checkbox])

    delete_flagged_btn.click(
        fn=delete_flagged_results,
        inputs=[flagged_checkbox],
        outputs=[flagged_status, flagged_checkbox, flagged_table]
    )

    delete_all_flagged_btn.click(
        fn=delete_all_flagged,
        outputs=[flagged_status, flagged_checkbox, flagged_table]
    )

    # SAVE FLAGGED BUTTONS (Downloads CSV)
    save_flagged_selected_btn.click(
        fn=save_flagged_selected,
        inputs=[flagged_checkbox],
        outputs=[flagged_download_output]
    )

    save_flagged_all_btn.click(
        fn=save_flagged_all,
        outputs=[flagged_download_output]
    )

    # ============ TAB REFRESH HANDLERS ============

    tabs.select(
        fn=refresh_all_charts,
        inputs=None,
        outputs=[
            sentiment_pie, confidence_bar, sentiment_trend,
            time_of_day_chart, day_of_week_chart, month_chart, year_chart,
            stats_overview, stats_time
        ],
        show_progress=False
    )

    summary_sub_tabs.select(
        fn=refresh_all_charts,
        inputs=None,
        outputs=[
            sentiment_pie, confidence_bar, sentiment_trend,
            time_of_day_chart, day_of_week_chart, month_chart, year_chart,
            stats_overview, stats_time
        ],
        show_progress=False
    )

    # ============ RESET HANDLERS ============
    reset_btn.click(
        fn=confirm_reset,
        outputs=[confirm_row, reset_btn]
    )

    confirm_no_btn.click(
        fn=cancel_reset,
        outputs=[confirm_row, reset_btn]
    )

    confirm_yes_btn.click(
        fn=execute_reset,
        outputs=[
            reset_status, history_output, flagged_table, flagged_checkbox,
            analyzed_table, analyzed_checkbox, reset_status, confirm_row, reset_btn,
            sentiment_pie, confidence_bar, sentiment_trend,
            time_of_day_chart, day_of_week_chart, month_chart, year_chart,
            stats_overview, stats_time
        ]
    )

# ============ LAUNCH THE APP ============
if __name__ == "__main__":
    demo.launch()
