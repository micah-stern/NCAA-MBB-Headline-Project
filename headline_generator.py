import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Embedding,
    Concatenate,
    Attention,
)
from sklearn.model_selection import train_test_split
import pickle
import os


class HeadlineGenerator:
    def __init__(
        self,
        max_headline_length=20,
        max_stats_length=100,
        vocab_size=10000,
        embedding_dim=256,
        latent_dim=512,
    ):
        """
        Initialize the Headline Generator model

        Args:
            max_headline_length: Maximum length of output headlines
            max_stats_length: Maximum length of input statistics sequence
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            latent_dim: Dimension of LSTM hidden state
        """
        self.max_headline_length = max_headline_length
        self.max_stats_length = max_stats_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # Initialize tokenizers
        self.stats_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.headline_tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n',
        )

        # Model components will be initialized in build_model()
        self.encoder = None
        self.decoder = None
        self.model = None

    def preprocess_data(self, stats_data, headlines):
        """
        Preprocess the input data and prepare for training

        Args:
            stats_data: List of game statistics as strings
            headlines: List of corresponding headlines

        Returns:
            Tuple of (padded_stats_sequences, padded_headline_sequences, tokenizer)
        """
        # Fit tokenizers
        self.stats_tokenizer.fit_on_texts(stats_data)
        self.headline_tokenizer.fit_on_texts(headlines)

        # Add start and end tokens to headlines
        headlines = ["<start> " + h + " <end>" for h in headlines]

        # Convert text to sequences
        stats_sequences = self.stats_tokenizer.texts_to_sequences(stats_data)
        headline_sequences = self.headline_tokenizer.texts_to_sequences(headlines)

        # Pad sequences
        padded_stats = pad_sequences(
            stats_sequences, maxlen=self.max_stats_length, padding="post"
        )
        padded_headlines = pad_sequences(
            headline_sequences, maxlen=self.max_headline_length, padding="post"
        )

        # Create training data (shifted by one for teacher forcing)
        decoder_input_data = padded_headlines[:, :-1]
        decoder_target_data = padded_headlines[:, 1:]

        return padded_stats, decoder_input_data, decoder_target_data

    def build_model(self):
        """Build the seq2seq model with attention mechanism"""
        # Encoder
        encoder_inputs = Input(shape=(self.max_stats_length,), name="encoder_inputs")
        encoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(
            encoder_inputs
        )
        encoder_lstm = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            name="encoder_lstm",
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(
            decoder_inputs
        )
        decoder_lstm = LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True,
            name="decoder_lstm",
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding, initial_state=encoder_states
        )

        # Attention layer
        attention = Attention()([decoder_outputs, encoder_outputs])
        decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention])

        # Dense layer for output
        decoder_dense = Dense(
            self.vocab_size, activation="softmax", name="decoder_dense"
        )
        decoder_outputs = decoder_dense(decoder_combined_context)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Build the encoder model for inference
        self.encoder = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

        # Build the decoder model for inference
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]

        # Attention for inference
        attention = Attention()([decoder_outputs, encoder_outputs])
        decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention])

        decoder_outputs = decoder_dense(decoder_combined_context)
        self.decoder = Model(
            [decoder_inputs, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs] + decoder_states,
        )

    def train(
        self, stats_data, headlines, batch_size=64, epochs=20, validation_split=0.2
    ):
        """
        Train the model

        Args:
            stats_data: List of game statistics as strings
            headlines: List of corresponding headlines
            batch_size: Batch size for training
            epochs: Number of epochs to train for
            validation_split: Fraction of data to use for validation
        """
        # Prepare the data
        X_encoder, X_decoder, y = self.preprocess_data(stats_data, headlines)

        # Convert y to 3D array for sparse_categorical_crossentropy
        y = y.reshape(*y.shape, 1)

        # Split the data
        (
            X_encoder_train,
            X_encoder_val,
            X_decoder_train,
            X_decoder_val,
            y_train,
            y_val,
        ) = train_test_split(
            X_encoder, X_decoder, y, test_size=validation_split, random_state=42
        )

        # Train the model
        history = self.model.fit(
            [X_encoder_train, X_decoder_train],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_encoder_val, X_decoder_val], y_val),
        )

        return history

    def generate_headline(self, stats, temperature=1.0):
        """
        Generate a headline for the given statistics

        Args:
            stats: Game statistics as a string
            temperature: Sampling temperature (higher = more random, lower = more deterministic)

        Returns:
            Generated headline as a string
        """
        # Tokenize and pad the input statistics
        stats_seq = self.stats_tokenizer.texts_to_sequences([stats])
        stats_seq = pad_sequences(
            stats_seq, maxlen=self.max_stats_length, padding="post"
        )

        # Encode the input
        enc_outputs, h, c = self.encoder.predict(stats_seq)

        # Start with the start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.headline_tokenizer.word_index["<start>"]

        # Generate sequence
        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            output_tokens, h, c = self.decoder.predict([target_seq] + [h, c])

            # Sample a token with temperature
            output_tokens = output_tokens.reshape(-1)
            scaled_logits = np.log(output_tokens + 1e-10) / max(temperature, 1e-10)
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)
            sampled_token_index = np.argmax(np.random.multinomial(1, probs, 1))

            # Get the word and add to the result
            sampled_word = ""
            for word, index in self.headline_tokenizer.word_index.items():
                if index == sampled_token_index:
                    decoded_sentence.append(word)
                    sampled_word = word
                    break

            # Exit conditions: max length or end token
            if (
                sampled_word == "<end>"
                or len(decoded_sentence) >= self.max_headline_length - 1
            ):
                stop_condition = True

            # Update the target sequence
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            # h, c are already updated by the decoder

        # Remove <start> and <end> tokens and join the words
        if decoded_sentence and decoded_sentence[0] == "<start>":
            decoded_sentence = decoded_sentence[1:]
        if decoded_sentence and decoded_sentence[-1] == "<end>":
            decoded_sentence = decoded_sentence[:-1]

        return " ".join(decoded_sentence)

    def save(self, model_dir="model"):
        """Save the model and tokenizers"""
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        self.model.save(os.path.join(model_dir, "headline_generator.h5"))

        # Save tokenizers
        with open(os.path.join(model_dir, "stats_tokenizer.pkl"), "wb") as f:
            pickle.dump(self.stats_tokenizer, f)
        with open(os.path.join(model_dir, "headline_tokenizer.pkl"), "wb") as f:
            pickle.dump(self.headline_tokenizer, f)

        # Save model config
        config = {
            "max_headline_length": self.max_headline_length,
            "max_stats_length": self.max_stats_length,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.latent_dim,
        }

        with open(os.path.join(model_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

    @classmethod
    def load(cls, model_dir="model"):
        """Load a saved model"""
        # Load config
        with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        # Create model instance with saved config
        model = cls(**config)

        # Load tokenizers
        with open(os.path.join(model_dir, "stats_tokenizer.pkl"), "rb") as f:
            model.stats_tokenizer = pickle.load(f)
        with open(os.path.join(model_dir, "headline_tokenizer.pkl"), "rb") as f:
            model.headline_tokenizer = pickle.load(f)

        # Load model weights
        model.model = tf.keras.models.load_model(
            os.path.join(model_dir, "headline_generator.h5")
        )

        # Rebuild the encoder and decoder for inference
        model.build_model()

        return model


def prepare_training_data(meta_df, box_df):
    """
    Prepare training data from the dataframes

    Args:
        meta_df: DataFrame with game_id and headlines
        box_df: DataFrame with game statistics

    Returns:
        Tuple of (stats_strings, headlines)
    """
    # Group box scores by game
    game_stats = (
        box_df.groupby("game_id")
        .apply(lambda x: x.drop("game_id", axis=1).to_dict("records"))
        .to_dict()
    )

    # Prepare input-output pairs
    stats_strings = []
    headlines = []

    for _, row in meta_df.iterrows():
        game_id = row["game_id"]
        if game_id in game_stats:
            # Convert game stats to a string representation
            stats = game_stats[game_id]
            stats_str = " | ".join(
                [
                    f"{p['player']} {p.get('points', 0)}p {p.get('rebounds', 0)}r {p.get('assists', 0)}a"
                    for p in stats
                ]
            )
            stats_strings.append(stats_str)
            headlines.append(row["headline"])

    return stats_strings, headlines


def main():
    # Example usage
    # Load your data
    meta_df = pd.read_csv("cbb_headlines_11_2024.csv")
    box_df = pd.read_csv("cbb_boxscores_11_2024.csv")

    # Prepare the data
    stats_strings, headlines = prepare_training_data(meta_df, box_df)

    # Initialize and train the model
    model = HeadlineGenerator()
    model.build_model()
    history = model.train(stats_strings, headlines, epochs=20)

    # Save the model
    model.save("headline_model")

    # Example of loading and using the model
    loaded_model = HeadlineGenerator.load("headline_model")
    headline = loaded_model.generate_headline("Player1 24p 5r 3a | Player2 18p 8r 6a")
    print("Generated headline:", headline)


if __name__ == "__main__":
    main()
