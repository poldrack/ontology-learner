import secrets
import hashlib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_random_hash(prefix='', hashlength=12):
    # Generate a secure random string
    random_string = secrets.token_hex(32)  # Generates a 32-character hexadecimal string

    # Create a hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the bytes of the random string
    hash_object.update(random_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    random_hash = hash_object.hexdigest()

    return prefix + random_hash[:hashlength]


def scale_df(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, index=df.index)


