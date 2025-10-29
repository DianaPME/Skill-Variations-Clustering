import pandas as pd
import numpy as np
from collections import Counter
import re
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
import pickle
import faiss
from text_cleaning import TextCleaning
from vgcn import SimpleVocabGCN

class SkillVariationClustering:
    def __init__(self, df, text_column, use_gcn, matrix_projection_path= None):
        """
        Initialize the SkillVariationClustering class with necessary parameters.      
        :param file_path: Path to the .xlsx file
        :param text_column: Name of the column containing text data
        """
        self.df= df
        self.text_column = text_column
        self.use_gcn= use_gcn
        self.projection_path= matrix_projection_path

        # Load tokenizer and model for generating embeddings
        self.base_model = None
        self.dapt_model= None
        self.refined_model= None

        # Initialize a TextCleaning instance to preprocess text
        print('Cleaning and Normalizing Text...\n')
        self.text_preprocessor = TextCleaning(text_column, self.df)
        self.text_preprocessor.data_cleaning()
        self.text_preprocessor.normalize()

    
    def convert_to_embeddings(self, text_column: str) -> np.ndarray:
        model_path = "dmadera/dapt-beto-skill-variations"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the base model
        self.base_model = AutoModelForMaskedLM.from_pretrained(model_path,  output_hidden_states=True, output_attentions=False)

        # Create Sentence Transformer components
        word_embedding_model = models.Transformer(
            model_path,
            max_seq_length=512  
        )

        # Create pooling model
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False
        )

        # Construct SentenceTransformer
        self.dapt_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print('Creating embeddings...\n')
        # Generate embeddings
        embeddings = self.dapt_model.encode(self.df[text_column], convert_to_tensor=True)
        return embeddings
    
    def fuse_bert_and_simple_gcn_embeddings(self, bert_embeddings, tfidf_matrix, vocab_adj_matrix, gcn_model):
        gcn_model.eval()

        with torch.no_grad():
            # Get GCN embeddings
            gcn_embeddings = gcn_model(tfidf_matrix, vocab_adj_matrix)

            # Convert to numpy if needed
            if isinstance(gcn_embeddings, torch.Tensor):
                gcn_embeddings = gcn_embeddings.detach().cpu().numpy()

            # Ensure BERT embeddings are numpy
            if isinstance(bert_embeddings, torch.Tensor):
                bert_embeddings = bert_embeddings.detach().cpu().numpy()

        # Weight for BERT embeddings
        alpha = 0.8  
        # Weight for GCN embeddings
        beta = 0.2   

        # Ensure dimensions match
        if gcn_embeddings.shape[1] != bert_embeddings.shape[1]:
            if gcn_embeddings.shape[1] != bert_embeddings.shape[1]:
            # Load the projection matrix
             with open(self.projection_path, 'rb') as f:
                projection_matrix = pickle.load(f)
            print(f"Loaded projection matrix with shape: {projection_matrix.shape}\n")
            # Apply the projection matrix to transform GCN embeddings
            gcn_embeddings = gcn_embeddings @ projection_matrix

        fused_embeddings = alpha * bert_embeddings + beta * gcn_embeddings
        return fused_embeddings
    
    def convert_to_vcgn(self):
        vectorizer = TfidfVectorizer(max_features=4000)
        tfidf_matrix = vectorizer.fit_transform(self.df['normalized'])
        # Build vocab adjacency from co-occurrence 
        adj_matrix = (tfidf_matrix.T @ tfidf_matrix)  
        # Normalize diagonal to 0
        adj_matrix.setdiag(0)
        vocab_size = tfidf_matrix.shape[1]  
        bert_dim = 512  
        simple_gcn = SimpleVocabGCN(vocab_size, bert_dim)
        return tfidf_matrix, adj_matrix, simple_gcn
    
    def generate_ngrams(self, text, ngram_range=(1, 4)):
        # Basic tokenization and n-gram generation
        tokens = re.findall(r'\b\w+\b', text.lower())
        ngrams = set()

        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngrams.add(ngram)

        return ngrams

    def normalize(self, vectors):
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-10)
    
    def assign_topics_with_ngrams_and_cosine_faiss_simplified(
        self,
        docs,
        bert_embeddings,  
        bertopic_topics,
        embedding_model,
        tfidf_matrix=None,
        vocab_adj_matrix=None,
        gcn_model=None,
        keyword_weight=0.5,
        cosine_weight=0.5,
        ngram_range=(1, 4),
        min_score_threshold=0.01,
        use_gcn=True
    ):

        # Prepare document embeddings
        if use_gcn and gcn_model is not None and tfidf_matrix is not None:
            print("Using BERT + GCN fusion\n")
            doc_embeddings = self.fuse_bert_and_simple_gcn_embeddings(
                bert_embeddings, tfidf_matrix, vocab_adj_matrix, gcn_model
            )
        else:
            print("Using BERT embeddings only\n")
            doc_embeddings = bert_embeddings
            if isinstance(doc_embeddings, torch.Tensor):
                doc_embeddings = doc_embeddings.detach().cpu().numpy()

        # Prepare topic keywords
        topic_keywords = {
            topic_id: set(kw.lower() for kw, _ in words)
            for topic_id, words in bertopic_topics.items()
            if words
        }

        # Create topic embeddings using BERT
        topic_embeddings = {}
        for topic_id, words in bertopic_topics.items():
            if words:
                topic_text = " ".join([kw for kw, _ in words])
                topic_embed = embedding_model.encode([topic_text])[0]

                # Match dimensions with document embeddings
                if topic_embed.shape[0] != doc_embeddings.shape[1]:
                    if topic_embed.shape[0] < doc_embeddings.shape[1]:
                        # Pad with zeros
                        padded_embed = np.zeros(doc_embeddings.shape[1])
                        padded_embed[:topic_embed.shape[0]] = topic_embed
                        topic_embeddings[topic_id] = padded_embed
                    else:
                        # Truncate
                        topic_embeddings[topic_id] = topic_embed[:doc_embeddings.shape[1]]
                else:
                    topic_embeddings[topic_id] = topic_embed

        # Normalize embeddings
        doc_embeddings = self.normalize(doc_embeddings)

        # Build FAISS index
        topic_ids = list(topic_embeddings.keys())
        if not topic_ids:
            return np.array([-1] * len(docs))

        topic_matrix = np.stack([topic_embeddings[tid] for tid in topic_ids])
        topic_matrix = self.normalize(topic_matrix)

        dim = topic_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity
        index.add(topic_matrix.astype('float32'))

        # Assign topics
        assigned_ids = []

        for doc, doc_embedding in zip(docs, doc_embeddings):
            doc_ngrams = self.generate_ngrams(doc, ngram_range)

            # Get cosine similarity scores from FAISS
            D, I = index.search(doc_embedding.reshape(1, -1).astype('float32'), len(topic_ids))

            best_topic = -1
            best_score = -1

            for i in range(len(topic_ids)):
                topic_id = topic_ids[I[0][i]]
                cosine_sim = D[0][i]

                keywords = topic_keywords[topic_id]
                match_count = len(doc_ngrams & keywords)
                keyword_score = match_count / len(keywords) if keywords else 0

                combined_score = (
                    keyword_weight * keyword_score +
                    cosine_weight * cosine_sim
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_topic = topic_id

            if best_score < min_score_threshold:
                best_topic = -1  

            assigned_ids.append(best_topic)

        return np.array(assigned_ids)
    
    def refine_with_doc_similarity(self, embeddings, assigned_ids, similarity_threshold=0.55):
        sim_matrix = cosine_similarity(embeddings)
        n_docs = sim_matrix.shape[0]

        # Build clusters based on similarity threshold
        clusters = []
        visited = set()

        for i in range(n_docs):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, n_docs):
                if sim_matrix[i, j] > similarity_threshold:
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)

        # Refine topic assignments using majority vote in each cluster
        refined_ids = assigned_ids.copy()
        for cluster in clusters:
            topic_votes = [assigned_ids[i] for i in cluster]
            most_common = Counter(topic_votes).most_common(1)[0][0]
            for i in cluster:
                refined_ids[i] = most_common

        return np.array(refined_ids)
    
    def get_top_words(self, topic_number, num_words=5):
        # Get the topic list from refined_model.get_topic(number)
        topic = self.refined_model.get_topic(topic_number)

        # Sort the list by probability (second element of the tuple) in descending order
        sorted_topic = sorted(topic, key=lambda x: x[1], reverse=True)

        # Get the top N words based on probability
        top_words = [word for word, prob in sorted_topic[:num_words]]

        # Join the words into a string
        return ', '.join(top_words)

    
    def cluster_texts(self):
        """
        Perform clustering on the text embeddings and return cluster IDs.
        """
        embeddings= self.convert_to_embeddings('processed_text')
        umap_model = UMAP(random_state= 42, n_components=2, metric= 'cosine',  min_dist=0.01, n_neighbors=3)

        # Create a custom vectorizer with custom stop words
        vectorizer = CountVectorizer(stop_words=self.text_preprocessor.spanish_stop_words)

        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

        # Create representation model
        mmr= MaximalMarginalRelevance(diversity=0.0)
        kb = KeyBERTInspired()
        representation_model = [kb, mmr]

        # Create BERTopic with the custom embedding model
        topic_model = BERTopic(
            umap_model=umap_model,
            embedding_model= self.dapt_model,
            top_n_words=10,
            language="spanish",
            ctfidf_model=ctfidf_model,
            calculate_probabilities=True
        )
        print('Generating topics...\n')
        # Fit the model
        topics, probs = topic_model.fit_transform(self.df['processed_text'].to_list())

        # Inspect topics
        initial_topics = topic_model.get_topics()
        print('Generating refined topics\n')
        self.refined_model= BERTopic(umap_model=UMAP(random_state= 42), 
                                representation_model=representation_model, 
                                vectorizer_model=vectorizer, 
                                calculate_probabilities=True).fit(self.df['processed_text'].to_list(), y=initial_topics)
        refined_topics, refined_probs = self.refined_model.transform(self.df['processed_text'].to_list())
        # Convert to NumPy array
        embeddings = embeddings.cpu().numpy()
        tfidf_matrix, adj_matrix, simple_gcn = self.convert_to_vcgn()
        new_labels_refined_old = self.assign_topics_with_ngrams_and_cosine_faiss_simplified(self.df['normalized'], embeddings, self.refined_model.get_topics(), self.dapt_model, tfidf_matrix, adj_matrix, simple_gcn, use_gcn=self.use_gcn)
        new_labels_refined = self.refine_with_doc_similarity(embeddings, new_labels_refined_old)
        self.df['cluster_id']= new_labels_refined
        topic_results = [self.get_top_words(x) for x in self.df['cluster_id']]
        self.df['topics'] = self.df['cluster_id'].apply(self.get_top_words)

        return new_labels_refined, topic_results
        


