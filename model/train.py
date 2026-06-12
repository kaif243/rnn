from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense
)

def build_model():

    model = Sequential()

    model.add(
        Embedding(
            input_dim=5000,
            output_dim=64
        )
    )

    model.add(
        SimpleRNN(
            64
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model