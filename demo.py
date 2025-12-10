# https://drive.google.com/drive/folders/1DbZe37JSWVFaVUAgyRY7fFrzLSDsY_xq?usp=sharing
from flask import Flask, render_template, request
import io, base64
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from utils import load_data, accuracy, edge_index_from_adj
from models import R_GATModel

app = Flask(__name__)

# Load adjacency and derive edge_index and edge_type
adj = load_data()
row_idx, col_idx = edge_index_from_adj(adj)
edge_index = torch.stack([row_idx, col_idx], dim=0)  # shape (2, E)
edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

# Instantiate relation-aware GAT
tfid = 3
num_relations = 1
model = R_GATModel(
    nfeat=tfid,
    nhid=64,
    nclass=2,
    dropout=0.38,
    nheads=8,
    stock_num=adj.size(0),
    num_relations=num_relations
)
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

COMPANIES = [
    'AAPL','AMZN','BA','BAC','C','CAT','CELG','CSCO','CVX','D','DIS','FB',
    'GE','GOOG','HD','INTC','JNJ','JPM','KO','MCD','MRK','MSFT','PCLN','PFE',
    'T','V','VZ','WFC','WMT','XOM'
]

RAW_DATA_DIR = './Data/raw_data'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', companies=COMPANIES)

def get_date_range_data(company, start, end):
    text_dir = os.path.join(RAW_DATA_DIR, 'text', company)
    price_dir = os.path.join(RAW_DATA_DIR, 'price', company)
    dates = sorted([f[:-4] for f in os.listdir(text_dir) if f.endswith('.npy') and start <= f[:-4] <= end])

    price_feats, text_feats = [], []
    for d in dates:
        price_arr = np.load(os.path.join(price_dir, d + '.npy'))
        text_arr = np.load(os.path.join(text_dir, d + '.npy'))
        price_feats.append(price_arr)
        text_feats.append(text_arr)
    return dates, price_feats, text_feats

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    start = request.form['start_date']
    end = request.form['end_date']

    dates, price_feats, text_feats = get_date_range_data(company, start, end)
    labels = []

    for price_arr, text_arr in zip(price_feats, text_feats):
        p = torch.tensor(price_arr, dtype=torch.float32)
        t = torch.tensor(text_arr, dtype=torch.float32)
        price_list = [p] * adj.size(0)
        text_list  = [t] * adj.size(0)
        logits = model(text_list, price_list, edge_index, edge_type)

        idx = COMPANIES.index(company)
        pred = logits[idx].argmax().item()
        labels.append('UP' if pred == 1 else 'DOWN')

    predictions = list(zip(dates, labels))

    # Plot: UP = 1, DOWN = 0
    binary_labels = [1 if label == 'UP' else 0 for label in labels]
    fig, ax = plt.subplots()
    ax.plot(dates, binary_labels, marker='o', linestyle='-', color='blue')
    ax.set_title(f'Predicted Trend for {company}')
    ax.set_ylabel('Trend (UP=1, DOWN=0)')
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return render_template('index.html', companies=COMPANIES, predictions=predictions, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
