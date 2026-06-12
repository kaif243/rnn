import streamlit as st
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model.utils import load_data
from model.train import build_model

st.set_page_config(
    page_title="RNN Sentiment Analysis",
    layout="wide"
)

st.title(
    "🎬 Movie Review Sentiment Analysis using Simple RNN"
)

df = load_data()

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head()
)

# Use smaller subset for quick training
df = df.sample(
    5000,
    random_state=42
)

X = df["review"]

y = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

tokenizer = Tokenizer(
    num_words=5000
)

tokenizer.fit_on_texts(
    X
)

X_seq = tokenizer.texts_to_sequences(
    X
)

X_pad = pad_sequences(
    X_seq,
    maxlen=200
)

with st.spinner(
    "Training RNN..."
):

    model = build_model()

    model.fit(
        X_pad,
        y,
        epochs=2,
        batch_size=64,
        verbose=0
    )

st.success(
    "Model Trained Successfully"
)

review = st.text_area(
    "Enter Movie Review"
)

if st.button(
    "Predict Sentiment"
):

    seq = tokenizer.texts_to_sequences(
        [review]
    )

    pad = pad_sequences(
        seq,
        maxlen=200
    )

    prediction = model.predict(
        pad,
        verbose=0
    )[0][0]

    if prediction > 0.5:

        st.success(
            "Positive Review 😀"
        )

    else:

        st.error(
            "Negative Review 😞"
        )

st.subheader(
    "RNN Architecture"
)

st.code(
"""
Input Text
     ↓
Embedding
     ↓
SimpleRNN
     ↓
Dense
     ↓
Output
"""
)

st.write(
    "This project uses a genuine SimpleRNN layer for sequence processing."
)