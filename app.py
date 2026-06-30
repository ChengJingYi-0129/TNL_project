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

# ============ PATHS ============

MODEL_FOLDER = "best_sentiment_bigru"
FLAG_PATH = "flagged_data/"
os.makedirs(FLAG_PATH, exist_ok=True)

# ============ LOAD SPLIT MODEL FILES ============

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

print("Loading model from split files...")
with open(os.path.join(MODEL_FOLDER, 'data.pkl'), 'rb') as f:
    unpickler = pickle.Unpickler(f)
    unpickler.persistent_load = persistent_load
    state_dict = unpickler.load()
print(f"Loaded state dict with {len(state_dict)} layers")

# ============ LOAD TOKENIZER & LABEL ENCODER ============

with open('word_index.pkl', 'rb') as f:
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
print(f"Loaded NEW tokenizer with {len(tokenizer.word_index)} words")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Label encoder loaded!")

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
    formats = [
        '%m/%d/%Y %I:%M:%S %p',
        '%m/%d/%Y %I:%M %p',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
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

# ============ DATA STORAGE & DISPLAY FUNCTIONS ============

sentiment_history = []
all_reviews = []
all_results = []
last_analysis_results = []
sentiment_data = []
filtered_results_cache = []
filtered_flagged_cache = []
analyzed_label_to_filename = {}

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
    if not all_results:
        return "No analyzed results yet.", []
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
    for i, res in enumerate(all_results, 1):
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
    html += f"<br><span class='table-total'>Total: {len(all_results)} results</span>"
    checkbox_choices = []
    for i, res in enumerate(all_results, 1):
        timestamp = res.get('timestamp', 'No date')
        date_part = timestamp.split(' ')[0] if ' ' in timestamp else timestamp[:10]
        checkbox_choices.append(f"[{i}] {res.get('sentiment', 'Unknown')} - {date_part}")
    return html, checkbox_choices

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

def create_empty_chart():
    fig = go.Figure()
    fig.add_annotation(text="No data available yet. Start analyzing reviews!", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="#2A0800"))
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
    return fig.to_html(full_html=False)

def create_visualization_charts():
    if not sentiment_data:
        return create_empty_chart()
    valid_data = [d for d in sentiment_data if d['sentiment'] != 'Error']
    if not valid_data:
        return create_empty_chart()
    df = pd.DataFrame(valid_data)
    fig = go.Figure()
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
    if not sentiment_counts.empty:
        fig.add_trace(go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, marker=dict(colors=[colors.get(s, '#95a5a6') for s in sentiment_counts.index]), hole=0.4, textinfo='label+percent'))
    fig.update_layout(height=500, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#2A0800'))
    return fig.to_html(full_html=False)

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

def get_readable_flagged_list():
    global analyzed_label_to_filename
    analyzed_label_to_filename = {}
    files = get_flagged_files()
    if not files:
        return []
    readable_list = []
    for file in files:
        filepath = os.path.join(FLAG_PATH, file)
        parsed = parse_flagged_file(filepath)
        if parsed:
            date_display = parsed['date'][:10] if parsed['date'] else 'Unknown'
            label = f"{parsed['sentiment']} - {parsed['review_preview']} ({date_display})"
            readable_list.append(label)
            analyzed_label_to_filename[label] = file
        else:
            readable_list.append(file)
            analyzed_label_to_filename[file] = file
    return readable_list

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

# ============ SAVE FUNCTIONS (DOWNLOAD TO COMPUTER) ============

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

def clear_history():
    global last_analysis_results, sentiment_data, filtered_results_cache
    sentiment_history.clear()
    all_reviews.clear()
    all_results.clear()
    last_analysis_results = []
    sentiment_data = []
    filtered_results_cache = []
    analyzed_display, analyzed_choices = get_analyzed_results_display()
    chart_html = create_visualization_charts()
    return "History cleared!", "Recent History:\nNo history yet.", "History cleared!", analyzed_display, gr.update(choices=analyzed_choices, value=[]), chart_html

def refresh_analyzed():
    display, checkbox_choices = get_analyzed_results_display()
    return display, gr.update(choices=checkbox_choices, value=[]), "Refreshed successfully!"

def reset_all():
    global last_analysis_results, sentiment_history, all_reviews, all_results, sentiment_data, filtered_results_cache
    sentiment_history.clear()
    all_reviews.clear()
    all_results.clear()
    last_analysis_results = []
    sentiment_data = []
    filtered_results_cache = []
    files = get_flagged_files()
    for f in files:
        try:
            os.remove(os.path.join(FLAG_PATH, f))
        except:
            pass
    analyzed_display, analyzed_choices = get_analyzed_results_display()
    flagged_display, flagged_choices = get_flagged_display_table()
    chart_html = create_visualization_charts()
    return "All data has been reset successfully!", "Recent History:\nNo history yet.", flagged_display, gr.update(choices=flagged_choices, value=[]), analyzed_display, gr.update(choices=analyzed_choices, value=[]), "", gr.update(visible=False), gr.update(visible=True), chart_html

def confirm_reset():
    return gr.update(visible=True), gr.update(visible=False)

def cancel_reset():
    return gr.update(visible=False), gr.update(visible=True)

def execute_reset():
    msg, history, flagged_display, flagged_checkbox, analyzed_display, analyzed_checkbox, reset_status, confirm_row, reset_btn, chart_html = reset_all()
    return msg, history, flagged_display, flagged_checkbox, analyzed_display, analyzed_checkbox, confirm_row, reset_btn, msg, chart_html

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
        .conf-high ... (40 KB left)
