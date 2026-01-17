# A1: That's What I LIKE - Semantic Search with Word Embeddings

## Introduction

This assignment, **A1: That's What I LIKE**, focuses on developing a semantic search system that retrieves the top paragraphs from a corpus with the most similar context to a user query (e.g., "Harry Potter"). The core of the system relies on word embeddings, which capture semantic and syntactic relationships between words in a vector space. By training custom implementations of **Word2Vec** (Mikolov et al., 2013) and **GloVe** (Pennington et al., 2014) from scratch, building upon class code, we explore predictive and count-based approaches to embedding learning. These models are evaluated on training efficiency, word analogy tasks, and correlation with human similarity judgments, before deploying a simple web interface for querying.

#### Key Summary of the Word2Vec Paper  
**Efficient Estimation of Word Representations in Vector Space** (Mikolov et al., 2013) introduces the Skip-gram model, which efficiently learns high-quality distributed representations of words by predicting surrounding context words given a target word. Unlike earlier models that used dense softmax over the entire vocabulary (computationally expensive), the paper proposes **negative sampling**—an approximation that samples a small number of negative (incorrect) context words and uses sigmoid loss to distinguish positive from negative pairs. This speeds up training dramatically while preserving quality. The embeddings enable arithmetic analogies (e.g., king - man + woman ≈ queen) due to their linear substructure. Key contributions include scalability to large corpora, superior performance on word similarity and analogy tasks, and the insight that local context prediction captures rich semantics.

#### Key Summary of the GloVe Paper  
**GloVe: Global Vectors for Word Representation** (Pennington et al., 2014) presents a global matrix factorization approach that leverages co-occurrence statistics across the entire corpus, rather than local windows alone. It constructs a co-occurrence matrix and minimizes a weighted least-squares objective on the log of co-occurrence probabilities, with weights decaying for rare pairs. This captures global statistics (e.g., ratios like P(ice|solid) / P(water|solid)) while remaining efficient. The resulting embeddings combine the advantages of predictive models (like Word2Vec) and matrix factorization methods, achieving state-of-the-art results on word analogy and similarity benchmarks. The paper emphasizes that global co-occurrence ratios encode more semantic information than raw probabilities.

By implementing and comparing these models on a real-world corpus (NLTK Reuters news dataset), this assignment demonstrates their strengths, limitations (e.g., sensitivity to corpus size), and practical deployment in a search engine.

## Project Overview

This repository contains the implementation for AT82.05 Artificial Intelligence: Natural Language Understanding (NLU) Assignment 1. It includes:
- Custom modifications to Word2Vec (Skip-gram with and without negative sampling) and GloVe models, with dynamic window size support.
- Training on the NLTK Reuters corpus.
- Evaluation on training loss/time, word analogies (semantic: capital-common-countries; syntactic: past-tense), and Spearman correlation with human similarity judgments (WordSim-353 dataset).
- A simple Flask-based web application for semantic search, retrieving the top-10 similar paragraphs based on query embeddings (using averaged word vectors and dot product similarity).

Key parameters used:
- Vector size: 32
- Epochs: 10000
- Window size: 2
- Batch size: 64
- Low accuracies are expected due to small corpus size, as noted in the assignment.

**Dataset Credits:**
- Corpus: NLTK Reuters dataset (public domain, available at https://www.nltk.org/).
- Word Analogies: From https://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt (Mikolov et al.).
- WordSim-353: From http://alfonseca.org/eng/research/wordsim353.html (Finkelstein et al.).

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/prazwal1/That-s-What-I-LIKE.git
   cd A1_Thats_What_I_LIKE
   ```

2. **Install Dependencies**:
   Create a virtual environment and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**:
   ```
   python -c "import nltk; nltk.download('reuters')"
   ```

4. **Run the Notebook** (A1_notebook.ipynb):
   - Launch Jupyter: `jupyter notebook`
   - Open and run `A1_notebook.ipynb` to train models, generate embeddings (saved in `data/`), and evaluate.
   - Note: Training may take time; use a subset of the corpus for quick testing (e.g., `corpus_sentences[:1000]`).

5. **Run the Web App**:
   ```
   cd app
   python app.py
   ```
   - Access at http://localhost:5000.
   - Enter a query (e.g., "Harry Potter") to see top-10 similar paragraphs from the corpus (using GloVe embeddings by default).

## Experiments and Results

### Model Training and Comparison
Models were trained with window size=2, vector size=50, epochs=5. Skip-gram NEG was fastest, while basic Skip-gram had higher loss due to full softmax. GloVe captured global stats but had higher loss from co-occurrence weighting. Gensim pre-trained GloVe (wiki-gigaword-50) serves as a benchmark.

| Model            | Window Size | Training Loss | Training Time (s) | Semantic Accuracy | Syntactic Accuracy |
|------------------|-------------|---------------|-------------------|-------------------|--------------------|
| Skipgram        | 2          | 13.726706    | 3783.448128      | 0.000000         | 0.000000          |
| Skipgram (NEG)  | 2          | 4.279615     | 230.880975       | 0.000000         | 0.000000          |
| GloVe           | 2          | 698.010717   | 212.801008       | 0.000000         | 0.000000          |
| Gensim (GloVe)  | -          | -            | -                | 0.894433         | 0.554487          |

**Discussion**: Custom models showed 0% accuracy on analogies due to limited corpus size (as expected per assignment note). Gensim performed well, highlighting the need for larger data. Experiment: With window=5, training time increased ~1.5x for Skip-gram, but semantic accuracy remained low; correlations improved slightly (e.g., +0.02 for NEG).

### Similarity Correlation
Used dot product for model similarities and Spearman rank correlation with human judgments (WordSim-353). MSE not computed here, but correlations assess alignment.

| Model            | Spearman Correlation |
|------------------|----------------------|
| Skipgram        | 0.010459            |
| Skipgram (NEG)  | 0.059556            |
| GloVe           | -0.029026           |
| Gensim (GloVe)  | 0.532735            |

**Discussion**: Low/negative correlations for custom models reflect corpus limitations (small vocab, domain-specific news). Gensim's higher value shows pre-trained benefits. Dot product was used as per task; cosine similarity yielded similar trends.

## Web Application Usage

### 🌐 Live Demo
**The NewsFind application is live at:** [https://newsfind.ambitiousisland-1be3b1ed.southeastasia.azurecontainerapps.io/](https://newsfind.ambitiousisland-1be3b1ed.southeastasia.azurecontainerapps.io/)

### Features
- The app uses Flask and loads pre-trained embeddings (Skipgram Negative by default) from `data/`.
- Paragraph embeddings: Averaged word vectors.
- Similarity: Cosine similarity between query and paragraph vectors.
- Example: Query "oil prices" returns news-related paragraphs from Reuters.
- Limitations: Results are unreliable due to training on a very small dataset and for an insufficient number of epochs, which prevents the model from learning stable and generalizable vector embeddings.


