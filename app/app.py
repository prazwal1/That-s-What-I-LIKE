"""
Task 3: Web Application for Similar Context Search
A Flask web application that uses trained embeddings to find similar contexts.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Global variables for loaded data
embeddings = None
word2index = None
index2word = None
corpus = None
corpus_raw = None
doc_ids = None


def load_data(model_name='skipgram_neg'):
    """Load embeddings and corpus data."""
    global embeddings, word2index, index2word, corpus, corpus_raw, doc_ids
    
    # Load embeddings
    emb_path = f'exports/{model_name}_embeddings.npy'
    embeddings = torch.FloatTensor(np.load(emb_path))
    
    # Load vocabulary
    vocab_path = f'exports/{model_name}_vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    word2index = vocab_data['word2index']
    index2word = vocab_data['index2word']
    
    # Load corpus
    corpus_path = 'exports/reuters_corpus.pkl'
    with open(corpus_path, 'rb') as f:
        corpus_data = pickle.load(f)
    corpus = corpus_data['tokenized']
    corpus_raw = corpus_data['raw']
    doc_ids = corpus_data['doc_ids']
    
    print(f"Loaded {model_name} embeddings: {embeddings.shape}")
    print(f"Loaded corpus: {len(corpus)} documents")


def find_similar_words(word, top_k=10):
    """Find similar words using cosine similarity."""
    word = word.lower()
    
    if word not in word2index:
        return None
    
    word_idx = word2index[word]
    query_vec = embeddings[word_idx]
    
    # Compute cosine similarities: (A·B) / (||A|| * ||B||)
    query_norm = query_vec / (torch.norm(query_vec) + 1e-8)
    emb_norms = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-8)
    similarities = emb_norms @ query_norm
    similarities[word_idx] = -1e9  # Exclude self
    
    top_indices = torch.topk(similarities, top_k).indices
    
    results = []
    for idx in top_indices:
        idx = idx.item()
        results.append({
            'word': index2word[idx],
            'similarity': float(similarities[idx].item())
        })
    
    return results


def find_contexts(query_text, window=5, top_k=20):
    """Find contexts containing the word(s) or similar words."""
    # Split query into words and normalize
    words = [w.lower().strip() for w in query_text.strip().split() if w.strip()]
    
    if not words:
        return {'error': 'Please enter at least one word'}
    
    # Check which words are in vocabulary
    valid_words = [w for w in words if w in word2index]
    missing_words = [w for w in words if w not in word2index]
    
    if not valid_words:
        return {'error': f"None of the words found in vocabulary: {', '.join(words)}"}
    
    # Compute average embedding for multi-word query
    query_indices = [word2index[w] for w in valid_words]
    query_vec = embeddings[query_indices].mean(dim=0)
    
    # Find similar words based on averaged embedding using cosine similarity
    query_norm = query_vec / (torch.norm(query_vec) + 1e-8)
    emb_norms = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-8)
    similarities = emb_norms @ query_norm
    # Exclude query words from similar words
    for idx in query_indices:
        similarities[idx] = -1e9
    
    top_indices = torch.topk(similarities, 15).indices
    
    similar = []
    for idx in top_indices:
        idx = idx.item()
        similar.append({
            'word': index2word[idx],
            'similarity': float(similarities[idx].item())
        })
    
    similar_word_set = {w['word'] for w in similar}
    for w in valid_words:
        similar_word_set.add(w)
    
    # Build similarity lookup
    sim_lookup = {w['word']: w['similarity'] for w in similar}
    for w in valid_words:
        sim_lookup[w] = 1.0
    
    # Search corpus
    contexts = []
    
    for doc_idx, doc in enumerate(corpus):
        for word_pos, token in enumerate(doc):
            if token in similar_word_set:
                start = max(0, word_pos - window)
                end = min(len(doc), word_pos + window + 1)
                context_words = doc[start:end]
                target_pos = word_pos - start
                
                contexts.append({
                    'doc_id': doc_ids[doc_idx],
                    'target_word': token,
                    'context': ' '.join(context_words),
                    'target_position': target_pos,
                    'similarity': sim_lookup.get(token, 0.0),
                    'snippet': corpus_raw[doc_idx][:300]
                })
    
    # Sort and limit
    contexts = sorted(contexts, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    return {
        'query': ' '.join(valid_words),
        'num_words': len(valid_words),
        'missing_words': missing_words if missing_words else None,
        'similar_words': similar[:10],
        'num_found': len(contexts),
        'contexts': contexts
    }


@app.route('/')
def home():
    """Home page with search interface."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """API endpoint for searching similar contexts."""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter one or more words to search'}), 400
    
    results = find_contexts(query, window=5, top_k=15)
    return jsonify(results)


@app.route('/similar/<word>')
def similar(word):
    """Get similar words for a given word."""
    results = find_similar_words(word, top_k=10)
    if results is None:
        return jsonify({'error': f"Word '{word}' not found"}), 404
    return jsonify({'word': word, 'similar': results})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'vocab_size': len(word2index) if word2index else 0,
        'corpus_size': len(corpus) if corpus else 0
    })


if __name__ == '__main__':
    print("Starting NewsFind Application...")
    
    # Load data
    load_data(model_name='skipgram_neg')  
    
    print("\nApplication ready!")
    print("Open: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
