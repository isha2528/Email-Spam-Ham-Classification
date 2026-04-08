import pandas as pd
import numpy as np
import sys
import math
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, auc)
from sklearn.preprocessing import LabelEncoder, normalize, LabelBinarizer
from scipy.sparse import lil_matrix, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

#required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

#########################################################
# Naïve Bayes Classifier 
#########################################################
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = []
        self.vocab = set()
        self.vocab_size = 0
        self.smoothing = 1
        self.total_words_in_training = 0

    def train(self, X, y):
        self.classes = np.unique(y)
        num_docs = len(y)
        # Calculating class probabilities
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / num_docs
        word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_in_training = 0
        for doc, label in zip(X, y):
            words = doc.split()
            self.vocab.update(words)
            self.total_words_in_training += len(words)
            for word in words:
                word_counts[label][word] += 1
        self.vocab_size = len(self.vocab)
        # Calculating feature probabilities using add-1 smoothing
        for c in self.classes:
            total_words_in_class = sum(word_counts[c].values())
            self.feature_probs[c] = {}
            for word in self.vocab:
                count = word_counts[c].get(word, 0)
                self.feature_probs[c][word] = (count + self.smoothing) / (total_words_in_class + self.smoothing * self.vocab_size)

    def predict_proba(self, doc):
        log_probs = {}
        words = doc.split()
        for c in self.classes:
            log_probs[c] = math.log(self.class_probs[c])
            for word in words:
                if word in self.vocab:
                    log_probs[c] += math.log(self.feature_probs[c][word])
                else:
                    log_probs[c] += math.log(self.smoothing / (self.total_words_in_training + self.smoothing * self.vocab_size))
        # Converting log probabilities to linear scale
        max_log = max(log_probs.values())
        probs = {c: math.exp(log_probs[c] - max_log) for c in log_probs}
        total = sum(probs.values())
        return {c: probs[c] / total for c in probs}

    def predict(self, doc):
        probs = self.predict_proba(doc)
        return max(probs.items(), key=lambda x: x[1])[0]

#########################################################
# Text Preprocessing Class
#########################################################
class TextPreprocessor:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        # Replacing non-alphabet characters with a space and lowering the text
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

#########################################################
# Data Loading, Vector Creation, and Evaluation Functions
#########################################################
def load_data(twitter_training, twitter_validation):
    colnames = ['index', 'topic', 'label', 'text']
    df_train = pd.read_csv(twitter_training, names=colnames, dtype={'text': 'string'})
    df_val = pd.read_csv(twitter_validation, names=colnames, dtype={'text': 'string'})
    df = pd.concat([df_train, df_val])
    return df

def create_sparse_vectors(docs, vocab):
    vocab_index = {word: i for i, word in enumerate(vocab)}
    X = lil_matrix((len(docs), len(vocab)), dtype=np.int32)
    for i, doc in enumerate(docs):
        for word in doc.split():
            if word in vocab_index:
                X[i, vocab_index[word]] += 1
    return X.tocsr()

def calculate_metrics(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        'true_positives': cm.diagonal(),
        'false_positives': cm.sum(axis=0) - cm.diagonal(),
        'false_negatives': cm.sum(axis=1) - cm.diagonal(),
        'true_negatives': cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal()),
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=classes),
        'confusion_matrix': cm
    }
    tp = metrics['true_positives']
    fp = metrics['false_positives']
    fn = metrics['false_negatives']
    tn = metrics['true_negatives']
    epsilon = 1e-10  # to avoid division by zero
    metrics.update({
        'sensitivity': tp / (tp + fn + epsilon),
        'specificity': tn / (tn + fp + epsilon),
        'precision': tp / (tp + fp + epsilon),
        'negative_predictive_value': tn / (tn + fn + epsilon),
        'f_score': 2 * (tp / (tp + fp + epsilon)) * (tp / (tp + fn + epsilon)) /
                   ((tp / (tp + fp + epsilon)) + (tp / (tp + fn + epsilon)) + epsilon)
    })
    return metrics

def print_metrics(metrics, classes):
    print("\nTest results / metrics:")
    print("Number of true positives:", metrics['true_positives'])
    print("Number of true negatives:", metrics['true_negatives'])
    print("Number of false positives:", metrics['false_positives'])
    print("Number of false negatives:", metrics['false_negatives'])
    print("Sensitivity (recall):", metrics['sensitivity'])
    print("Specificity:", metrics['specificity'])
    print("Precision:", metrics['precision'])
    print("Negative predictive value:", metrics['negative_predictive_value'])
    print("Accuracy:", metrics['accuracy'])
    print("F-score:", metrics['f_score'])
    print("\nClassification Report:\n", metrics['classification_report'])
    print("Confusion Matrix:\n", metrics['confusion_matrix'])

#########################################################
# Plotting Functions for ROC Curves and Confusion Matrices
#########################################################
def plot_roc_curves(y_test, nb_probs, lr_probs, classes):
    # Checking if the classification is binary or multiclass
    if len(classes) == 2:
        # For binary classification, using probabilities for the positive class
        fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probs)
        auc_nb = roc_auc_score(y_test, nb_probs)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
        auc_lr = roc_auc_score(y_test, lr_probs)
    else:
        # For multiclass, first binarize the labels
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test_bin = lb.transform(y_test)
        # Preparing Naïve Bayes probability array: each row is ordered by lb.classes_
        nb_probs_arr = []
        for doc_prob in nb_probs:
            row = [doc_prob.get(cls, 0) for cls in lb.classes_]
            nb_probs_arr.append(row)
        nb_probs_arr = np.array(nb_probs_arr)
        # Logistic Regression already produces an array
        lr_probs_arr = lr_probs
        # Computing micro-average ROC curve for Naïve Bayes
        fpr_nb, tpr_nb, _ = roc_curve(y_test_bin.ravel(), nb_probs_arr.ravel())
        auc_nb = roc_auc_score(y_test_bin, nb_probs_arr, multi_class='ovr', average='micro')
        # And for Logistic Regression
        fpr_lr, tpr_lr, _ = roc_curve(y_test_bin.ravel(), lr_probs_arr.ravel())
        auc_lr = roc_auc_score(y_test_bin, lr_probs_arr, multi_class='ovr', average='micro')
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_nb, tpr_nb, label=f'Naïve Bayes (AUC = {auc_nb:.3f})', linewidth=2)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrices(cm_nb, cm_lr, classes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes)
    axes[0].set_title('Naïve Bayes Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=classes, yticklabels=classes)
    axes[1].set_title('Logistic Regression Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

#########################################################
# Main Function: Load Data, Train Models, and Evaluate
#########################################################
def main():
    # Argument handling: expects two command line arguments: ALGO and TRAIN_SIZE
    # training both models to compare their ROC curves and confusion matrices.
    if len(sys.argv) != 3:
        print("Using default parameter: TRAIN_SIZE=80")
        train_size = 80
    else:
        train_size = max(50, min(80, int(sys.argv[2])))
    print("Gaonkar, Isha, A20585341 solution:")
    print(f"Training set size: {train_size} %")
    print("Training and evaluating both classifiers: Naïve Bayes and Logistic Regression")
    
    # Loading and preprocessing data
    print("Loading and preprocessing data...")
    df = load_data('twitter_training.csv', 'twitter_validation.csv')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data
    preprocessor = TextPreprocessor()
    df['text'] = df['text'].apply(preprocessor.preprocess)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    classes = le.classes_
    
    # Split data: training set = FIRST (TRAIN_SIZE%) samples, test set = LAST 20%
    train_end = int(train_size / 100 * len(df))
    X_train, y_train = df['text'].iloc[:train_end], y[:train_end]
    X_test, y_test = df['text'].iloc[train_end:], y[train_end:]
    
    # Building vocabulary from entire dataset 
    print("Building vocabulary from entire dataset...")
    full_text = df['text'].tolist()
    vocab = set()
    for doc in full_text:
        for word in doc.split():
            vocab.add(word)
    vocab = list(vocab)
    
    # -------------------------
    # Train Naïve Bayes Model
    # -------------------------
    print("Training Naïve Bayes classifier...")
    nb_model = NaiveBayesClassifier()
    nb_model.train(X_train, y_train)
    nb_pred = [nb_model.predict(doc) for doc in X_test]
    # Gathering probability estimates for each test document
    nb_probs = []
    for doc in X_test:
        probs_dict = nb_model.predict_proba(doc)
        # For binary classification, extracting probability for positive class;
        # for multiclass, keeping the whole dictionary.
        nb_probs.append(probs_dict)
    
    # ----------------------------
    # Train Logistic Regression Model
    # ----------------------------
    print("Creating feature vectors for Logistic Regression...")
    X_train_vec = create_sparse_vectors(X_train, vocab)
    X_test_vec = create_sparse_vectors(X_test, vocab)
    # Normalizing feature vectors 
    X_train_vec = normalize(X_train_vec, norm='l2', axis=1)
    X_test_vec = normalize(X_test_vec, norm='l2', axis=1)
    
    print("Training Logistic Regression classifier...")
    lr_model = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        n_jobs=-1
    )
    lr_model.fit(X_train_vec, y_train)
    lr_pred = lr_model.predict(X_test_vec)
    # For binary, extracting probability for positive class; for multiclass, keeping full array
    lr_probs = lr_model.predict_proba(X_test_vec)
    
    # ----------------------------
    # Evaluate and Print Metrics
    # ----------------------------
    print("\nEvaluating Naïve Bayes classifier...")
    nb_metrics = calculate_metrics(y_test, nb_pred, classes)
    print_metrics(nb_metrics, classes)
    
    print("\nEvaluating Logistic Regression classifier...")
    lr_metrics = calculate_metrics(y_test, lr_pred, classes)
    print_metrics(lr_metrics, classes)
    
    # ----------------------------
    # Plot ROC Curves
    # ----------------------------
    print("Plotting ROC curves for both classifiers...")
    # For binary classification, extract probability for positive class (assume label "1")
    if len(classes) == 2:
        nb_probs_binary = np.array([prob.get(1, 0) for prob in nb_probs])
        lr_probs_binary = lr_probs[:, 1]
        plot_roc_curves(y_test, nb_probs_binary, lr_probs_binary, classes)
    else:
        # For multiclass, passing the probability dictionaries/arrays directly.
        plot_roc_curves(y_test, nb_probs, lr_probs, classes)
    
    # ----------------------------
    # Plot Confusion Matrices
    # ----------------------------
    print("Plotting confusion matrices for both classifiers...")
    nb_cm = confusion_matrix(y_test, nb_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)
    plot_confusion_matrices(nb_cm, lr_cm, classes)
    
    # ----------------------------
    # Interactive Classification 
    # ----------------------------
    while True:
        sentence = input("\nEnter your sentence/document (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        processed = preprocessor.preprocess(sentence)
        # For Naïve Bayes
        nb_class = nb_model.predict(processed)
        nb_prob = nb_model.predict_proba(processed)
        print(f"\nNaïve Bayes: Classified as {le.inverse_transform([nb_class])[0]}.")
        for i, cl in enumerate(classes):
            print(f"P({cl} | S) = {nb_prob.get(i, 0):.4f}")
        # For Logistic Regression
        vec = create_sparse_vectors([processed], vocab)
        vec = normalize(vec, norm='l2', axis=1)
        lr_class = lr_model.predict(vec)[0]
        print(f"Logistic Regression: Classified as {le.inverse_transform([lr_class])[0]}.")
        cont = input("Do you want to enter another sentence [Y/N]? ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
