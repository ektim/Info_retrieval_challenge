import os
import re
import json
import random
import string
import itertools
from pathlib import Path
from tqdm.auto import tqdm
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#########################################################
#data loading and analysis
def load_json_data(file_path):
    with open(file_path, "r") as file:
        contents = json.load(file)
    return contents

def analyze_content(content_dict, flag='claims'):
    """
    Analyze the content dictionary based on the specified flag.

    Parameters:
      content_dict (dict): The content dictionary.
      flag (str): The analysis type. Allowed values:
          'claims' - Count keys starting with "c-en-"
          'descr'  - Count keys matching the pattern 'p<digits>'
          'total'  - Compute the total word count in all string values

    Returns:
      int: The result of the analysis.

    Raises:
      ValueError: If the flag is invalid.
    """
    if not isinstance(content_dict, dict):
        return 0

    if flag == 'claims':
        return sum(1 for k in content_dict.keys() if k.startswith('c-en-'))
    elif flag == 'descr':
        return sum(1 for k in content_dict.keys() if re.match(r'^p\d+$', k))
    elif flag == 'total':
        total_len = 0
        for value in content_dict.values():
            if isinstance(value, str):
                total_len += len(value.split())
        return total_len
    else:
        raise ValueError("Invalid flag. Allowed values: 'claims', 'descr', 'total'")

def get_mapping_dict(mapping_df):
    """
    Creates dictionary of citing ids to non-citing id based on given dataframe (which is based on providedjson)

    Parameters:
    mapping_df (DataFrame): DataFrame containing mapping between citing and cited patents
    Returns:
    dict: dictionary of unique citing patent ids to list of cited patent ids
    """
    mapping_dict = {}

    for _, row in mapping_df.iterrows():
        key = row[0]  # Value from column 0
        value = row[2]  # Value from column 2
        if key in mapping_dict:
            mapping_dict[key].append(value)
        else:
            mapping_dict[key] = [value]

    return mapping_dict

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)    

def create_weighted_corpus(corpus, weights, apply_cleaning=False):
    """
    Generate a weighted corpus from patent documents according to section weights.

    Parameters:
      corpus (iterable): Iterable of document dicts containing 'Content', 'Application_Number', and 'Application_Category'.
      weights (dict): Mapping of section names to weights: 'title', 'abstract', 'claims', 'description'.
      apply_cleaning (bool): Whether to apply clean_text to the final text.

    Returns:
      list: List of dicts with keys 'id' and 'text'.
    """
    weighted_corpus = []
    #for doc in tqdm(corpus, desc="Creating weighted corpus"):
    for doc in corpus:
        content = doc.get('Content', {})
        weighted_parts = []
        w = weights.get

        w_title = w('title', 1)
        if 'title' in content and w_title > 0:
            weighted_parts.extend([content['title']] * int(round(w_title)))

        w_abstract = w('abstract', 1)
        if 'pa01' in content and w_abstract > 0:
            weighted_parts.extend([content['pa01']] * int(round(w_abstract)))

        w_claims = w('claims', 1)
        if w_claims > 0:
            claims = [text for k, text in content.items() if k.startswith('c-en-')]
            weighted_parts.extend(claims * int(round(w_claims)))

        w_description = w('description', 1)
        if w_description > 0:
            desc = [text for k, text in content.items() if k.startswith('p') and k != 'pa01']
            weighted_parts.extend(desc * int(round(w_description)))

        text = " ".join(weighted_parts)
        if apply_cleaning:
            text = clean_text(text)

        patent_id = doc.get('Application_Number', '') + doc.get('Application_Category', '')
        weighted_corpus.append({'id': patent_id, 'text': text})

    return weighted_corpus


#########################################################
#TF-IDF methods

def create_tfidf_matrix(citing_texts, nonciting_texts, vectorizer, test=False):
    """
    Transform or fit+transform TF-IDF matrices for citing and nonciting texts.

    Parameters:
      citing_texts (list): Texts for query documents.
      nonciting_texts (list): Texts for candidate documents.
      vectorizer (TfidfVectorizer): A preconfigured TF-IDF vectorizer.
      test (bool): If True, call transform; if False, call fit_transform.

    Returns:
      tuple:
        tfidf_citing (sparse matrix): TF-IDF for citing_texts,
        tfidf_nonciting (sparse matrix): TF-IDF for nonciting_texts,
        vectorizer (TfidfVectorizer): The fitted or used vectorizer.
    """
    all_texts = citing_texts + nonciting_texts
    if test:
        tfidf_matrix = vectorizer.transform(tqdm(all_texts, desc="TF-IDF"))
    else:
        tfidf_matrix = vectorizer.fit_transform(tqdm(all_texts, desc="TF-IDF"))

    split_idx = len(citing_texts)
    tfidf_citing    = tfidf_matrix[:split_idx]
    tfidf_nonciting = tfidf_matrix[split_idx:]
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    return tfidf_citing, tfidf_nonciting, vectorizer

def prepare_tfidf_data(citing_docs, cited_docs, weights,
                       vectorizer, apply_cleaning=False, test=False):
    """
    Build weighted corpora and produce TF-IDF matrices, ID lists, and the vectorizer.

    Parameters:
      citing_docs (list): Documents to use as queries.
      cited_docs (list): Documents to use as candidates.
      weights (dict): Section weights for create_weighted_corpus.
      vectorizer (TfidfVectorizer): A preconfigured TF-IDF vectorizer.
      apply_cleaning (bool): Whether to apply clean_text to weighted text.
      test (bool): If True, only transform using existing vectorizer; else fit+transform.

    Returns:
      tuple:
        tfidf_query (sparse matrix): TF-IDF for citing_docs,
        tfidf_docs  (sparse matrix): TF-IDF for cited_docs,
        query_ids   (list): IDs of citing_docs,
        doc_ids     (list): IDs of cited_docs,
        vectorizer  (TfidfVectorizer): The fitted or used vectorizer.
    """

    weighted_citing = create_weighted_corpus(citing_docs, weights, apply_cleaning)
    weighted_cited  = create_weighted_corpus(cited_docs,  weights, apply_cleaning)

    query_ids   = [doc['id']   for doc in weighted_citing]
    doc_ids     = [doc['id']   for doc in weighted_cited]
    query_texts = [doc['text'] for doc in weighted_citing]
    doc_texts   = [doc['text'] for doc in weighted_cited]

    tfidf_query, tfidf_docs, vectorizer = create_tfidf_matrix(
        query_texts, doc_texts, vectorizer, test=test
    )

    return tfidf_query, tfidf_docs, query_ids, doc_ids, vectorizer

def get_tfidf_recommendations(
    query_ids,
    candidate_ids,
    tfidf_query,
    tfidf_docs,
    k=100,
    return_scores=False
):
    """
    For each query in query_ids, compute cosine similarities to tfidf_docs,
    then return the top-k recommendations.

    Parameters:
      query_ids (list): List of query identifiers (length N).
      candidate_ids (list): List of candidate identifiers (length M).
      tfidf_query (sparse matrix): TF-IDF matrix for queries, shape [N, D].
      tfidf_docs (sparse matrix): TF-IDF matrix for candidates, shape [M, D].
      k (int): Number of top candidates to return per query.
      return_scores (bool): If True, return list of (candidate_id, score) tuples;
                            if False, return only candidate_id list.

    Returns:
      dict: Mapping each query_id to either a list of candidate_ids
            or a list of (candidate_id, score) tuples.
    """
    # compute cosine similarity matrix [N × M]
    sim_matrix = linear_kernel(tfidf_query, tfidf_docs)

    recommendations = {}
    for qi, qid in enumerate(query_ids):
        top_idxs = np.argsort(sim_matrix[qi])[::-1][:k]
        if return_scores:
            recommendations[qid] = [
                (candidate_ids[i], float(sim_matrix[qi, i])) for i in top_idxs
            ]
        else:
            recommendations[qid] = [candidate_ids[i] for i in top_idxs]
    return recommendations

def prepare_training_data_tfidf(recommendations, gold_mapping):
    """
    Build training feature matrix X and label vector y from recommendation scores,
    using optimized comprehension and precomputed gold sets.

    Parameters:
      recommendations (dict): Mapping from query_id to list of (candidate_id, score) tuples.
      gold_mapping (dict): Mapping from query_id to list of true relevant candidate_ids.

    Returns:
      tuple:
        X (np.ndarray): Array of shape [n_samples, 1] with similarity scores.
        y (np.ndarray): Array of shape [n_samples,] with binary labels (1=relevant, 0=non-relevant).
    """
    gold_sets = {qid: set(cids) for qid, cids in gold_mapping.items()}

    data = [
        (score, int(cid in gold_sets.get(qid, ())))
        for qid, recs in recommendations.items()
        for cid, score in recs
    ]
    
    if not data:
        return np.empty((0, 1)), np.empty((0,), dtype=int)
    scores, labels = zip(*data)
    X = np.array(scores, dtype=float).reshape(-1, 1)
    y = np.array(labels, dtype=int)
    return X, y


def re_rank_candidates_tfidf(recommendations, re_rank_model, top_k_candidates=100):
    """
    Re-rank candidates for each query using a trained re-rank model,
    based on the original similarity scores stored in recommendations.

    Parameters:
      recommendations (dict): Mapping from query_id to list of (candidate_id, score) tuples,
                              already sorted by descending score.
      re_rank_model: A classifier with predict_proba(X) -> [P(neg), P(pos)].
      top_k_candidates (int): Number of top candidates from the original recommendations to consider.

    Returns:
      dict: Mapping each query_id to a list of candidate_ids sorted by descending P(pos).
    """
    re_ranked = {}
    for qid, cand_list in recommendations.items():
        top_cands = cand_list[:top_k_candidates]
        ids, scores = zip(*top_cands) if top_cands else ([], [])
        
        if not ids:
            re_ranked[qid] = []
            continue
        
        X = np.array(scores).reshape(-1, 1)
        pos_probas = re_rank_model.predict_proba(X)[:, 1]
        order = np.argsort(pos_probas)[::-1]
        re_ranked[qid] = [ids[i] for i in order]
    return re_ranked
    
#########################################################
# grid search tf-idf

def grid_search_tfidf(
    citing_docs,
    nonciting_docs,
    gold_mapping,
    weights_grid,
    vectorizer_params_grid,
    k=100,
    sample_fraction=1.0,
    random_state=None
):
    """
    Grid search over section weights and TF-IDF parameters, sampling fractions of both
    citing (queries) and nonciting (candidates) docs.

    Parameters:
      citing_docs (list of dict): JSON-loaded citing patents.
      nonciting_docs (list of dict): JSON-loaded nonciting patents.
      gold_mapping (dict): Mapping from citing_id to list of relevant nonciting_ids.
      weights_grid (dict): Grid of section weights.
      vectorizer_params_grid (dict): Grid of kwargs for TfidfVectorizer.
      k (int): Number of top candidates per query.
      sample_citing_fraction (float): Fraction of citing_docs to sample (0<≤1).
      sample_non_citing_fraction (float): Fraction of nonciting_docs to sample (0<≤1).
      random_state (int|None): Seed for sampling.

    Returns:
      (best_params, best_score)
    """
    rng = random.Random(random_state)
    if sample_fraction < 1.0:
        n_cite = int(len(citing_docs) * sample_fraction)
        citing_docs = rng.sample(citing_docs, n_cite)
        n_noncite = int(len(nonciting_docs) * sample_fraction)
        nonciting_docs = rng.sample(nonciting_docs, n_noncite)

    best_score = -np.inf
    best_params = None

    for weight_vals in itertools.product(*weights_grid.values()):
        weights = dict(zip(weights_grid.keys(), weight_vals))
        for vect_vals in itertools.product(*vectorizer_params_grid.values()):
            vect_params = dict(zip(vectorizer_params_grid.keys(), vect_vals))
            vectorizer = TfidfVectorizer(**vect_params)
            tfidf_query, tfidf_docs, query_ids, doc_ids, _ = prepare_tfidf_data(
                citing_docs,
                nonciting_docs,
                weights,
                vectorizer,
                apply_cleaning=False,
                test=False
            )
            sim_matrix = linear_kernel(tfidf_query, tfidf_docs)
            recs = get_tf_idf_recommendations(
                query_ids, doc_ids, sim_matrix, k=k, return_scores=False
            )

            recall, mAP = evaluate_recommendations(gold_mapping, recs, k=k)
            composite = 0.5 * recall + 0.5 * mAP
            print(f"Grid params: weights={weights}, vect_params={vect_params}, "
                  f"Recall@{k}={recall:.4f}, mAP@{k}={mAP:.4f}, composite={composite:.4f}")
            
            if composite > best_score:
                best_score = composite
                best_params = {
                    'weights':           weights,
                    'vectorizer_params': vect_params,
                    'recall':            recall,
                    'mAP':               mAP,
                    'composite_score':   composite
                }
    return best_params, best_score
    
#########################################################
#dense embeddings functions

def load_embeddings_and_ids(embedding_file, app_ids_file):
    """
    Load the embeddings and application IDs from saved files
    """
    print(f"Loading embeddings from {embedding_file}")
    embeddings = torch.from_numpy(np.load(embedding_file))

    print(f"Loading app_ids from {app_ids_file}")
    with open(app_ids_file, 'r') as f:
        app_ids = json.load(f)

    print(f"Loaded {len(embeddings)} embeddings and {len(app_ids)} app_ids")
    return embeddings, app_ids


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def citation_to_citing_to_cited_dict_embb(citations):
    """
    Put a citation mapping in a dict format
    """
    # Initialize an empty dictionary to store the results
    citing_to_cited_dict = {}

    # Iterate over the items in the JSON list
    for citation in citations:
        # Check if the citing id already exists in the resulting dictionary
        if citation[0] in citing_to_cited_dict:
            # If the citing id exists, append the cited id to the existing list
            citing_to_cited_dict[citation[0]].append(citation[2])
        else:
            # If the citing id doesn't exist, create a new list with the cited id for that citing id
            citing_to_cited_dict[citation[0]] = [citation[2]]

    return citing_to_cited_dict

def get_true_and_predicted(citing_to_cited_dict, recommendations_dict):
    """
    Get the true and predicted labels for the metrics calculation.
    """
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    not_in_citation_mapping = 0

    # Iterate over the items in both dictionaries
    for citing_id in recommendations_dict.keys():
        # Check if the citing_id is present in both dictionaries
        if citing_id in citing_to_cited_dict:
            # If yes, append the recommended items from both dictionaries to the respective lists
            true_labels.append(citing_to_cited_dict[citing_id])
            predicted_labels.append(recommendations_dict[citing_id])
        else:
            print(citing_id, "not in citation mapping")
            not_in_citation_mapping += 1

    return true_labels, predicted_labels, not_in_citation_mapping

def get_dense_recommendations(query_embeddings, query_app_ids, doc_embeddings, doc_app_ids, top_k_candidates=100):
    """
    Generate dense similarity recommendations based on cosine similarity.

    Parameters:
      query_embeddings (iterable): Iterable of query embeddings.
      query_app_ids (iterable): Iterable of query identifiers.
      doc_embeddings (torch.Tensor): Tensor of document embeddings.
      doc_app_ids (iterable): Iterable of document identifiers.
      TOP_N (int): Number of top candidates per query (default is 100).

    Returns:
      dict: A dictionary mapping each query identifier to a list of tuples (doc_app_id, cosine_score).
    """
    dense_results = {}
    for query_embedding, query_id in tqdm(zip(query_embeddings, query_app_ids), total=len(query_embeddings), desc="Dense embeddings similarity"):
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float)
        query_embedding = query_embedding.unsqueeze(0)
        cos_scores = pytorch_cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
        top_indices = np.argsort(cos_scores)[::-1][:top_k_candidates]
        dense_results[query_id] = [(doc_app_ids[idx], cos_scores[idx]) for idx in top_indices]
    return dense_results

def grid_search_embedd_comp(params, BASE_DIR):
    """
    Evaluate retrieval performance for a single embedding configuration.

    This function:
      1. Constructs file paths for document and query embeddings based on the
         provided MODEL_NAME and CONTENT_TYPE in params and the BASE_DIR.
      2. Loads precomputed embeddings and their corresponding app IDs.
      3. Computes cosine similarities between each query embedding and all
         document embeddings, retrieves the top-N candidates.
      4. Loads the gold citation mapping and computes Recall@100 and mAP@100.

    Parameters:
      params (dict): Must contain:
        - 'MODEL_NAME'   (str): Embedding model name, e.g. 'all-MiniLM-L6-v2'
        - 'CONTENT_TYPE' (str): Section key, e.g. 'TA', 'claims', 'TAC'
      BASE_DIR (str):   Root directory where embedding folders and citation
                        JSON live.

    Returns:
      tuple:
        recall_at_k (float): Recall@100 for this configuration.
        map_score   (float): Mean Average Precision@100 for this configuration.
    """
    doc_dir   = os.path.join(BASE_DIR, "embeddings_precalculated_docs")
    query_dir = os.path.join(BASE_DIR, "embeddings_precalculated_train")
    citation_file = os.path.join(BASE_DIR, "Citation_JSONs", "Citation_Train.json")

    POOLING = "mean"
    TOP_N   = 100
    K      = 100

    model = params['MODEL_NAME']
    ctype = params['CONTENT_TYPE']
    doc_emb_file = os.path.join(doc_dir,   f"embeddings_{model}_{POOLING}_{ctype}.npy")
    doc_ids_file = os.path.join(doc_dir,   f"app_ids_{model}_{POOLING}_{ctype}.json")
    qry_emb_file = os.path.join(query_dir, f"embeddings_{model}_{POOLING}_{ctype}.npy")
    qry_ids_file = os.path.join(query_dir, f"app_ids_{model}_{POOLING}_{ctype}.json")

    doc_embeddings, doc_app_ids     = load_embeddings_and_ids(doc_emb_file, doc_ids_file)
    query_embeddings, query_app_ids = load_embeddings_and_ids(qry_emb_file, qry_ids_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    doc_embeddings   = doc_embeddings.to(device)
    query_embeddings = query_embeddings.to(device)

    results = {}
    for q_emb, q_id in tqdm(zip(query_embeddings, query_app_ids),
                             total=len(query_embeddings),
                             desc=f"Retrieval [{model}-{ctype}]"):
        scores = pytorch_cos_sim(q_emb.unsqueeze(0), doc_embeddings)[0].cpu()
        top_idx = torch.argsort(scores, descending=True)[:TOP_N].numpy()
        results[q_id] = [doc_app_ids[i] for i in top_idx]

    with open(citation_file, 'r') as f:
        citations = json.load(f)
    gold_map = citation_to_citing_to_cited_dict_embb(citations)

    true_lbls, pred_lbls, _ = get_true_and_predicted(gold_map, results)
    recall_at_k = mean_recall_at_k(true_lbls, pred_lbls, k=K)
    map_score   = mean_average_precision(true_lbls, pred_lbls, k=K)

    return recall_at_k, map_score

def combine_embed(emb1: torch.Tensor, emb2: torch.Tensor, method: str) -> torch.Tensor:
    """
    Align dimensions if needed and combine two embedding tensors.

    Parameters:
      emb1, emb2 (Tensor): [N, D1] and [M, D2] — document or query embeddings.
      method (str): "sum", "avg", or "concat".

    Returns:
      Tensor: Combined embeddings.
    """
    # align dim for sum/avg
    if method in ("sum", "avg") and emb1.shape[1] != emb2.shape[1]:
        in_dim, out_dim = min(emb1.shape[1], emb2.shape[1]), max(emb1.shape[1], emb2.shape[1])
        lin = torch.nn.Linear(in_dim, out_dim).to(emb1.device)
        if emb1.shape[1] < emb2.shape[1]:
            emb1 = lin(emb1)
        else:
            emb2 = lin(emb2)

    # truncate to common length
    m = min(emb1.shape[0], emb2.shape[0])
    e1, e2 = emb1[:m], emb2[:m]

    if method == "sum":
        return e1 + e2
    if method == "avg":
        return (e1 + e2) / 2
    # concat
    return torch.cat([e1, e2], dim=1)


def grid_search_embedd_combinations(
    model_names,
    content_types,
    methods,
    citation_file,
    doc_dir,
    query_dir,
    top_n=100,
    k_value=100,
    pooling="mean"
):
    """
    Load precomputed embeddings, combine them by each method, retrieve and evaluate,
    printing metrics for each combination as we go.

    Parameters:
      model_names (list[str]): e.g. ["all-MiniLM-L6-v2", "PatentSBERTa"]
      content_types (list[str]): e.g. ["TA", "claims", "TAC"]
      methods (list[str]): Combination methods, e.g. ["sum","avg","concat"]
      citation_file (str): Path to JSON with citation mapping for evaluation.
      doc_dir (str): Directory of doc embeddings and app_ids.
      query_dir (str): Directory of query embeddings and app_ids.
      top_n (int): Number of candidates to retrieve per query.
      k_value (int): K for Recall@K and mAP@K.
      pooling (str): Pooling strategy in filenames

    Returns:
      tuple: (best_combo, best_score) where
        best_combo = (emb1, emb2, method, recall, mAP)
        best_score = recall + mAP
    """
    # load all embeddings once
    all_doc_embeddings   = {}
    all_query_embeddings = {}

    for m in model_names:
        for ctype in content_types:
            key = f"{m}_{ctype}"
            d_file = os.path.join(doc_dir,   f"embeddings_{m}_{pooling}_{ctype}.npy")
            dids   = os.path.join(doc_dir,   f"app_ids_{m}_{pooling}_{ctype}.json")
            q_file = os.path.join(query_dir, f"embeddings_{m}_{pooling}_{ctype}.npy")
            qids   = os.path.join(query_dir, f"app_ids_{m}_{pooling}_{ctype}.json")

            d_emb, doc_ids = load_embeddings_and_ids(d_file, dids)
            q_emb, qry_ids = load_embeddings_and_ids(q_file, qids)

            all_doc_embeddings[key]   = d_emb
            all_query_embeddings[key] = q_emb

    with open(citation_file) as f:
        citations = json.load(f)
    gold_map = citation_to_citing_to_cited_dict_embb(citations)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keys = list(all_doc_embeddings.keys())
    best_score = -np.inf
    best_combo = None

    for emb1, emb2 in combinations(keys, 2):
        for method in methods:
            docs = combine_embed(all_doc_embeddings[emb1], all_doc_embeddings[emb2], method).to(device)
            qrys = combine_embed(all_query_embeddings[emb1], all_query_embeddings[emb2], method).to(device)


            results = {}
            for q_emb, q_id in tqdm(zip(qrys, qry_ids), total=len(qrys), desc=f"{emb1}+{emb2}:{method}"):
                sims    = pytorch_cos_sim(q_emb.unsqueeze(0), docs)[0].cpu()
                top_idxs= torch.argsort(sims, descending=True)[:top_n].numpy()
                results[q_id] = [doc_ids[i] for i in top_idxs]

            true_lbls, pred_lbls, _ = get_true_and_predicted(gold_map, results)
            recall = mean_recall_at_k(true_lbls, pred_lbls, k=k_value)
            mAP    = mean_average_precision(true_lbls, pred_lbls, k=k_value)
            score  = recall + mAP
            print(
                f"Combination: {emb1} & {emb2}, method={method} | "
                f"Recall@{k_value}={recall:.4f}, mAP@{k_value}={mAP:.4f}, combined={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_combo = (emb1, emb2, method, recall, mAP)

    return best_combo, best_score

#################################################################
#TF-IDF + Dense Embb combinations

def fuse_results_simple(tfidf_results, dense_results, alpha=0.5, beta=0.5):
    """
    Fuse TF-IDF and Dense results by weighted sum of scores.

    For each query:
      combined_score = alpha * tfidf_score + beta * dense_score

    Then candidates are sorted by combined_score in descending order.

    Parameters:
      tfidf_results (dict): { query_id: [(candidate_id, tfidf_score), ...] }
      dense_results (dict): { query_id: [(candidate_id, dense_score), ...] }
      alpha (float): Weight for TF-IDF scores (default 0.5).
      beta (float): Weight for Dense scores (default 0.5).

    Returns:
      dict: { query_id: [candidate_id, ...] } sorted by fused score.
    """
    fused_results = {}
    for query_id in tfidf_results:
        tfidf_dict = dict(tfidf_results.get(query_id, []))
        dense_dict = dict(dense_results.get(query_id, []))
        candidates = set(tfidf_dict) | set(dense_dict)
        combined_scores = {
            cid: alpha * tfidf_dict.get(cid, 0.0) + beta * dense_dict.get(cid, 0.0)
            for cid in candidates
        }
        fused_results[query_id] = [
            cid for cid, _ in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ]
    return fused_results

def prepare_training_data_fused(ids, tfidf_results, dense_results, gold_mapping, top_k_candidates=100):
    """
    Build training features and labels from TF-IDF and Dense retrieval scores.

    Parameters:
      ids (list): List of query identifiers.
      tfidf_results (dict): { query_id: [(candidate_id, tfidf_score), ...] }
      dense_results (dict): { query_id: [(candidate_id, dense_score), ...] }
      gold_mapping (dict): { query_id: [relevant_candidate_id, ...] }
      top_k_candidates (int): Number of top candidates to consider per query.

    Returns:
      tuple:
        X (np.ndarray): Shape [n_samples, 2], columns [tfidf_score, dense_score].
        y (np.ndarray): Shape [n_samples], binary labels.
    """
    gold_sets = {qid: set(cids) for qid, cids in gold_mapping.items()}
    data = [
        (tfidf_dict.get(cid, 0.0),
         dense_dict.get(cid, 0.0),
         int(cid in gold_sets[qid]))
        for qid in ids
        if qid in gold_sets
        for tfidf_dict in [dict(tfidf_results.get(qid, [])[:top_k_candidates])]
        for dense_dict in [dict(dense_results.get(qid, [])[:top_k_candidates])]
        for cid in set(tfidf_dict) | set(dense_dict)
    ]

    if not data:
        return np.empty((0, 2)), np.empty((0,), dtype=int)

    arr = np.array(data, dtype=float)
    X = arr[:, :2]
    y = arr[:, 2].astype(int)
    return X, y    

def re_rank_candidates_fused(ids, tfidf_results, dense_results, re_rank_model, top_k_candidates=100):
    """
    For each query, union TF-IDF and Dense candidates, build [tfidf_score, dense_score] features,
    predict relevance probabilities with re_rank_model, and sort candidates by descending probability.

    Parameters:
      ids (list): List of query identifiers.
      tfidf_results (dict): { query_id: [(candidate_id, tfidf_score), ...] }
      dense_results (dict): { query_id: [(candidate_id, dense_score), ...] }
      re_rank_model: Classifier with predict_proba(X) method.
      top_k_candidates (int): Limit to top-K candidates from each list before fusing.

    Returns:
      dict: { query_id: [candidate_id, ...] } sorted by predicted relevance.
    """
    re_ranked = {}
    for qid in tqdm(ids, desc="Re-ranking candidates"):
        tfidf_dict = dict(tfidf_results.get(qid, [])[:top_k_candidates])
        dense_dict = dict(dense_results.get(qid, [])[:top_k_candidates])
        candidates = list(tfidf_dict.keys() | dense_dict.keys())
        features = np.stack([
            [tfidf_dict.get(cid, 0.0), dense_dict.get(cid, 0.0)]
            for cid in candidates
        ])
        probas = re_rank_model.predict_proba(features)[:, 1]
        order = np.argsort(probas)[::-1]
        re_ranked[qid] = [candidates[i] for i in order]
    return re_ranked    

def align_embeddings(embeddings, embedding_app_ids, reference_app_ids):
    """
    Align embeddings to a reference list of application IDs.

    Parameters:
      embeddings (np.ndarray): Array of shape [N, D] with embeddings.
      embedding_app_ids (list): List of N app IDs corresponding to rows in embeddings.
      reference_app_ids (list): List of M app IDs to align to.

    Returns:
      np.ndarray: Array of shape [M, D], where each row corresponds to the embedding
                  for the matching reference_app_id, or a zero vector if missing.
    """
    D = embeddings.shape[1]
    emb_dict = {app_id: emb for app_id, emb in zip(embedding_app_ids, embeddings)}

    # Align embeddings to the reference order, filling missing entries with zeros
    aligned = [
        emb_dict.get(app_id, np.zeros(D, dtype=np.float32))
        for app_id in reference_app_ids
    ]

    return np.vstack(aligned).astype(np.float32)

def fuse_results_rrf(tfidf_results, dense_results, k=60):
    """
    RRF fusion that accepts both [id] lists or [(id,score)] lists.
    """
    fused = {}
    qids = set(tfidf_results) & set(dense_results)  # only queries in both
    for qid in qids:
        # extract just the IDs
        raw_tf = tfidf_results[qid]
        raw_dn = dense_results[qid]
        tf_list = [cid for cid,_ in raw_tf] if raw_tf and isinstance(raw_tf[0], tuple) else raw_tf
        dn_list = [cid for cid,_ in raw_dn] if raw_dn and isinstance(raw_dn[0], tuple) else raw_dn

        rank_tf = {cid: r+1 for r, cid in enumerate(tf_list)}
        rank_dn = {cid: r+1 for r, cid in enumerate(dn_list)}
        candidates = set(rank_tf) | set(rank_dn)

        scores = {
            cid: 1.0/(k + rank_tf.get(cid, k+1)) + 1.0/(k + rank_dn.get(cid, k+1))
            for cid in candidates
        }
        fused[qid] = [
            cid for cid,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
    return fused

    
def grid_search_rrf(tfidf_results, dense_results, gold_mapping, rrf_k_values, eval_k=100):
    """
    Grid search over RRF parameter k, evaluating Recall@eval_k and mAP@eval_k.

    Parameters:
      tfidf_results (dict): { query_id: [candidate_id, ...] }
      dense_results (dict): { query_id: [candidate_id, ...] }
      gold_mapping (dict): { query_id: [true_candidate_id, ...] }
      rrf_k_values (list[int]): List of k values to test.
      eval_k (int): K for Recall@K and mAP@K.

    Returns:
      tuple:
        best_params (dict): {'rrf_k': k, 'recall': recall, 'mAP': mAP, 'combined': combined_score}
        best_score  (float): Highest combined_score = 0.5*recall + 0.5*mAP.
    """
    from task1_modules import evaluate_recommendations

    best_score = -float('inf')
    best_params = None

    for k in rrf_k_values:
        fused = fuse_results_rrf(tfidf_results, dense_results, k=k)
        recall, mAP = evaluate_recommendations(gold_mapping, fused, k=eval_k)
        combined = 0.5 * recall + 0.5 * mAP
        print(f"RRF k={k}: Recall@{eval_k}={recall:.4f}, mAP@{eval_k}={mAP:.4f}, combined={combined:.4f}")
        if combined > best_score:
            best_score = combined
            best_params = {'rrf_k': k, 'recall': recall, 'mAP': mAP, 'combined': combined}

    return best_params, best_score
    
#################################################################
#evaluations

def mean_recall_at_k(true_labels, predicted_labels, k=10):
    """
    Compute mean Recall@k over multiple recommendation lists.
    
    Parameters:
        true_labels (list of list): Actual relevant items for each query.
        predicted_labels (list of list): Recommended items for each query.
        k (int): Number of top recommendations to consider.
        
    Returns:
        float: Mean Recall@k.
    """
    recalls = []
    for true, pred in zip(true_labels, predicted_labels):
        true_set = set(true)
        top_k = pred[:k]
        relevant = sum(1 for item in top_k if item in true_set)
        recalls.append(relevant / len(true_set) if true_set else 0)
    return sum(recalls) / len(recalls) if recalls else 0

def mean_average_precision(true_labels, predicted_labels, k=10):
    """
    Compute mean Average Precision (mAP) over multiple recommendation lists.
    
    Parameters:
        true_labels (list of list): Actual relevant items for each query.
        predicted_labels (list of list): Recommended items for each query.
        k (int): Number of top recommendations to consider.
        
    Returns:
        float: Mean Average Precision.
    """
    average_precisions = []
    for true, pred in zip(true_labels, predicted_labels):
        true_set = set(true)
        precision_scores = []
        hits = 0
        for i, item in enumerate(pred[:k], start=1):
            if item in true_set:
                hits += 1
                precision_scores.append(hits / i)
        average_precisions.append(sum(precision_scores) / len(true_set) if true_set else 0)
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0

def evaluate_recommendations(gold_mapping, recommendations, k=100):
    """
    Evaluate recommendations using Recall@k and mean Average Precision.
    
    Parameters:
        gold_mapping (dict): Mapping from query_id to list of true relevant ids.
        recommendations (dict): Mapping from query_id to list of recommended ids.
        k (int): Number of top recommendations to consider.
        
    Returns:
        tuple: (recall, mean_average_precision)
    """
    true_labels = []
    predicted_labels = []
    for qid, true in gold_mapping.items():
        if qid in recommendations:
            true_labels.append(true)
            predicted_labels.append(recommendations[qid])
    recall = mean_recall_at_k(true_labels, predicted_labels, k)
    mAP = mean_average_precision(true_labels, predicted_labels, k)
    return recall, mAP

