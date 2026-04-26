from datasets import load_dataset
import re
from gensim.models import Word2Vec
import numpy as np
import torch
from torch import nn


def preprocess_arabic_text(text):
    if not isinstance(text, str):
        text = ""

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", "")
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.split()


def sentence_vector(tokens, model):
    vectors = []

    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def main() -> None:
    print("RUNNING")

    dataset = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(train_data)
    print(test_data)
    print(train_data[0])

    tokenized_sentences = []

    for row in train_data:
        tokens = preprocess_arabic_text(row["tweet"])

        if len(tokens) > 0:
            tokenized_sentences.append(tokens)

    print("Example:")
    print(tokenized_sentences[:3])

    print("Training Word2Vec...")

    w2v_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=0
    )

    print("Word2Vec training finished")
    print("Vocabulary size:", len(w2v_model.wv))

    word = tokenized_sentences[0][0]
    print("Test word:", word)
    print("Vector shape:", w2v_model.wv[word].shape)

    print("Most similar words:")
    print(w2v_model.wv.most_similar(word, topn=5))

    w2v_model.save("arabic_word2vec_cbow.model")
    print("Model saved successfully")

    print("Creating sentence vectors...")

    train_vectors = []
    train_labels = []

    for row in train_data:
        tokens = preprocess_arabic_text(row["tweet"])
        vector = sentence_vector(tokens, w2v_model)

        train_vectors.append(vector)
        train_labels.append(row["label"])

    test_vectors = []
    test_labels = []

    for row in test_data:
        tokens = preprocess_arabic_text(row["tweet"])
        vector = sentence_vector(tokens, w2v_model)

        test_vectors.append(vector)
        test_labels.append(row["label"])

    train_vectors = np.array(train_vectors)
    train_labels = np.array(train_labels)
    test_vectors = np.array(test_vectors)
    test_labels = np.array(test_labels)

    print("Train vectors shape:", train_vectors.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test vectors shape:", test_vectors.shape)
    print("Test labels shape:", test_labels.shape)

    X_train = torch.tensor(train_vectors, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)

    X_test = torch.tensor(test_vectors, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    epochs = 5

    for epoch in range(epochs):
        model.train()

        y_pred = model(X_train).squeeze()
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    model.eval()

    with torch.no_grad():
        preds = model(X_test).squeeze()
        preds = (preds > 0.5).float()

        accuracy = (preds == y_test).sum().item() / len(y_test)

    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
