import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Any


def compute_features(
    sequence: str,
    encoder: Any,
    max_length: int,
    exclude_unknown: bool = True,
    normalize_by_max_length: bool = False,
) -> tuple:
    """Compute the sequence length, one-hot representation and amino acid composition (AAC) for a given sequence using OneHotEncoder.

    Args:
        sequence (str): A string representing the protein sequence.

    Returns:
        tuple: A tuple containing the sequence length, one-hot encoded representation and a 21-length AAC vector.
    """
    # Compute sequence length
    seq_length = len(sequence)
    # Pad the sequence with 'X' to match the max length
    padded_sequence = sequence.ljust(max_length, "X")
    # Reshape the sequence into a column vector and encode it using the OneHotEncoder
    seq_array = np.array(list(padded_sequence)).reshape(-1, 1)
    one_hot_encoded = encoder.transform(seq_array)

    # Sum the one-hot encoded vectors to get the count for each amino acid, exclude 'X' if specified
    if exclude_unknown:
        aac_vector = np.sum(one_hot_encoded[:, :-1], axis=0)
    else:
        aac_vector = np.sum(one_hot_encoded, axis=0)

    # Normalize the AAC vector by the sequence length or max length
    if normalize_by_max_length:
        aac_vector = aac_vector / max_length
    else:
        aac_vector = aac_vector / seq_length

    return seq_length, one_hot_encoded, aac_vector


def extract_features_from_csv(csv_path: str, encoder: Any) -> pd.DataFrame:
    """Load the CSV containing sequence data and apply feature extraction.

    Args:
        csv_path (str): Path to the CSV file containing sequence data.

    Returns:
        pd.DataFrame: A DataFrame with sequence length, one-hot representation and AAC vector for each sequence.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    # Calculate the length of the longest sequence
    max_length = df["Sequence"].map(len).max()
    # Apply feature extraction
    df["Sequence_Length"], df["one_hot_encoded"], df["AAC_Vector"] = zip(
        *df["Sequence"].map(lambda seq: compute_features(seq, encoder, max_length))
    )

    return df


if __name__ == "__main__":
    # Run tests
    sequence = "ADHAIPNNAP"
    expected_length = 10
    # Expected one-hot encoded representation for the sequence 'ADHAIPNNAP'
    expected_one_hot_encoded = np.array(
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
)
    # Expected AAC vector for the sequence 'ADHAIPNNAP' normalized
    expected_aac = np.array(
        [0.3, 0, 0.1, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    # Initialize OneHotEncoder for 21 possible characters (20 amino acids + X for unknown)
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWYX")).reshape(-1, 1)
    encoder = OneHotEncoder(
        categories=[list("ACDEFGHIKLMNPQRSTVWYX")],
        sparse_output=False,
        handle_unknown="ignore",
    )
    encoder.fit(amino_acids)

    length, one_hot_encoded_vector, aac = compute_features(sequence, encoder, 10, exclude_unknown=False)

    # Confirm that the computed length matches the expected length
    assert (
        length == expected_length
    ), f"Expected length {expected_length}, but got {length}"

    # Confirm that the computed one-hot encoded vector is close to the expected one-hot encoded vector
    assert np.allclose(
        one_hot_encoded_vector, expected_one_hot_encoded
    ), "One-hot encoded vector does not match the expected result"

    # Confirm that the computed AAC vector is close to the expected AAC vector
    assert np.allclose(
        aac, expected_aac
    ), "AAC vector does not match the expected result"

    # Test the extract_features_from_csv function
    csv_file = "uniprot_sequences.csv"  # Update with your file path
    df = extract_features_from_csv(csv_file, encoder)
    df.to_csv("output.csv", index=False)  # Save the output to a CSV file
