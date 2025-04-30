# This will be a helper file
#
# It contains a wide range of helper functions to simplify the actual code I work with

# PART 0: IMPORTS

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, TimeDistributed, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef

from IPython.display import HTML


# PART 1: DATA PREPROCESSING

import json

# I use JSON because you can easily open and change it with a text editor outside of the code IDE
def save_data_configs_json(data_configs, save_path):
    """Saves the data_configs dictionary to a JSON file.

    Args:
        data_configs (dict): The dictionary to be saved.
        save_path (str): The file path where the dictionary will be saved.
    """
    with open(save_path, 'w') as file:
        json.dump(data_configs, file, indent=4)  # indent for better readability


def load_data_configs_json(load_path):
    """Loads the data_configs dictionary from a JSON file.

    Args:
        load_path (str): The file path where the dictionary is saved.

    Returns:
        dict: The loaded data_configs dictionary.
    """
    with open(load_path, 'r') as file:
        loaded_data_configs = json.load(file)
    return loaded_data_configs

def data_info(burst_vec):
    """
    Prints information about the dataset represented by the burst_vec array.

    This function provides details about the shape, dimensions, and values
    contained in the burst_vec array, which represents a dataset of burst signals.

    Parameters:
        burst_vec (np.ndarray): The array containing the burst signal data.
                                 It is expected to have at least 4 dimensions.
    """
    # Define the frequency range and noise ratios used in the dataset
    mean_freqs = np.logspace(np.log10(4), np.log10(100), num=20)
    mean_freqs_round = np.round(mean_freqs).astype(int)
    snr_range = np.arange(-10, 10, 2)

    # Print general information about the dataset
    print("DATASET INFO:")
    print("Shape:", burst_vec.shape)  # Overall shape of the data array
    print("\nNo of Samples:", burst_vec.shape[0])  # Number of samples in the dataset
    print("\nNo of Frequencies:", burst_vec.shape[1])  # Number of frequency values
    print("Freqency values:", list(mean_freqs_round))  # List of frequency values used

    # Print information about noise ratios
    print("\nNo of noise ratios:", burst_vec.shape[2])  # Number of noise ratios
    print("Signal to Noise ratios (in db)", list(snr_range))  # List of signal-to-noise ratios

    # Print information about data points
    print("\nNo of Datapoints:", burst_vec.shape[3])  # Number of data points per sample

    # Print information about features per data point (if available)
    if len(burst_vec.shape) > 4 and burst_vec.shape[4]:
        print("\nNo of Features per Datapoint:", burst_vec.shape[4], "(signal, hilbert amp, 20 wavelets for each freq)")
    else:
        print("No of Features per Datapoint:", 1, "\n")  # If no features are specified, assume 1

def load_data(data_path, label_path):
    """
    Loads the signal and label data from the specified file paths.

    Parameters:
        data_path (str): Path to the NumPy file containing the signal data.
        label_path (str): Path to the NumPy file containing the label data.

    Returns:
        tuple: A tuple containing the signal data and label data as NumPy arrays.
              (burst_vec, label_vec)
    """
    burst_vec = np.load(data_path)
    label_vec = np.load(label_path)
    return burst_vec, label_vec

def select_data(data_config:dict, data_vec=None, label_vec=None):
    """
    Selects a subset of the data based on configuration parameters.

    This function extracts specific portions of the signal and label data
    based on the number of samples, frequency values, and noise ratios
    specified in the data_config dictionary.

    Parameters:
        data_config (dict): Configuration dictionary containing data selection parameters.
        data_vec (np.ndarray): The signal data array. (Can also be Null)
        label_vec (np.ndarray): The label data array. (Can also be Null)

    Returns:
        tuple: A tuple containing the selected label and signal data as NumPy arrays.
              (label_vec, data_vec)
    """
    data_path = data_config['data_path']
    label_path = data_config['label_path']

    # This allows values to be None and therefore enable loading the whole
    freq_vals = data_config.get('freq_vals')
    noise_ratios = data_config.get('noise_ratios')
    n_samples = data_config.get('n_samples')
    feature_vals = data_config.get('feature_vals')

    if freq_vals is None:
        freq_vals = [0, data_vec.shape[1]]
    if noise_ratios is None:
        noise_ratios = [0, data_vec.shape[2]]
    if n_samples is None:
        n_samples = data_vec.shape[0]
    if feature_vals is None:
        feature_vals = [0,data_vec.shape[4]]

    # This serves in case a specific preloaded data vec wasnt loaded already
    if data_vec is None and label_vec is None:
        data_vec, label_vec = load_data(data_path, label_path)

    data_vec = data_vec[:n_samples,freq_vals[0]:freq_vals[1],noise_ratios[0]:noise_ratios[1],:,feature_vals[0]:feature_vals[1]]
    label_vec = label_vec[:n_samples,freq_vals[0]:freq_vals[1],noise_ratios[0]:noise_ratios[1],:]
    return label_vec, data_vec

import numpy as np

def preprocess_data(data_vector, label_vector):
    """
    Preprocesses data and label vectors to the desired shape (S, timesteps, I).
    Handles various input shapes, including timesteps from 1000 and up:
    - S: N_of_samples
    - F: N_of_frequencies
    - N: N_of_noise_burst_ratios
    - I: N_of_Input_Features
    - 2D: (S, timesteps)
    - 3D: (S, F, timesteps) or (S, timesteps, I)
    - 4D: (S, F, N, timesteps) or (S, F, timesteps, I)
    - 5D: (S, F, N, timesteps, I)

    Args:
        data_vector (np.ndarray): The input data vector.
        label_vector (np.ndarray): The input label vector.

    Returns:
        tuple: A tuple containing the preprocessed data and label vectors.
    """

    def reshape_vector(vector):
        """Helper function to reshape a single vector."""
        if vector.ndim == 2:  # (S, timesteps)
            # Reshape 2D to 3D by adding a feature dimension with I = 1
            vector = vector.reshape(vector.shape[0], vector.shape[1], 1)
        elif vector.ndim == 3:
            if vector.shape[1] < 1000 and vector.shape[2] >= 1000:  # (S, F, timesteps)
                # Assume timesteps is the last dimension, move it to the second position
                N = vector.shape[0] * vector.shape[1]
                vector = vector.reshape(N, vector.shape[-1], 1)
            # else: (S, timesteps, I) - already in desired format
        elif vector.ndim == 4:
            if vector.shape[2] < 1000 and vector.shape[3] >= 1000:  # (S, F, N, timesteps)
                # Reshape by flattening the first three dimensions
                N = vector.shape[0] * vector.shape[1] * vector.shape[2]
                vector = vector.reshape(N, vector.shape[3], 1)
            # Assume its (S, F, timesteps, I), flatten first 2 to get (N, timesteps, I)
            elif vector.shape[2] >= 1000 and vector.shape[3] < 1000:
                N = vector.shape[0] * vector.shape[1]
                vector = vector.reshape(N, vector.shape[2], vector.shape[3])
            else:
                raise ValueError("4D input shape not recognized. Please check the dimensions.")
        elif vector.ndim == 5:  # (S, F, N, timesteps, I)
            # Reshape by flattening the first three dimensions
            N = vector.shape[0] * vector.shape[1] * vector.shape[2]
            vector = vector.reshape(N, vector.shape[3], vector.shape[4])
        else:
            raise ValueError("Input vector must be 2D, 3D, 4D, or 5D.")
        return vector

    # Reshape data and label vectors independently
    data_vector = reshape_vector(data_vector)
    label_vector = reshape_vector(label_vector)

    return data_vector, label_vector
    
    
def create_data_dict(data_configs, config_names=None, universal_data=None, universal_label=None):
    """
    Creates a data dictionary containing preprocessed data for specified configurations.

    Args:
        data_configs (dict): A dictionary containing data configurations.
        config_names (list, optional): A list of configuration names to process.
                                        If None, all configurations are processed.
                                        Defaults to None.
        universal_data (array-like, optional): Data vector to use for all configurations.
                                               If provided, data loading is skipped.
                                               Defaults to None.
        universal_label (array-like, optional): Label vector to use for all configurations.
                                                If provided, label loading is skipped.
                                                Defaults to None.

    Returns:
        dict: A dictionary containing preprocessed data for each specified configuration.
    """
    data_dict = {}
    config_names = config_names or list(data_configs.keys())  # Use all configs if not specified

    for config_name in config_names:
        config = data_configs[config_name]
        
        # Load data using configuration if universal_data and universal_label are not provided
        if universal_data is None or universal_label is None:
            data, label = load_data(config["data_path"], config["label_path"])
        else:
            data, label = universal_data, universal_label
        
        # Select data using configuration
        label_data, signal_data = select_data(data_config=config, data_vec=data, label_vec=label)
        
        # Preprocess data
        signal_data, label_data = preprocess_data(data_vector=signal_data, label_vector=label_data)
        
        data_dict[config_name] = (signal_data, label_data)

    return data_dict

# PART 2: MACHINE LEARNING PROCESSING

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = None):
    """
    Splits data into training and testing sets.

    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The target data.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.25.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split.
                                     Pass an int for reproducible output across multiple function calls. Defaults to None.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Set the random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])  # Create indices for n samples
    np.random.shuffle(indices)  # Shuffle the indices

    # Use the shuffled indices to reorder the data
    signal_data_shuffled = X[indices]
    label_data_shuffled = y[indices]

    # Split into training and testing datasets
    split_index = int((1 - test_size) * len(signal_data_shuffled))
    X_train = signal_data_shuffled[:split_index]
    X_test = signal_data_shuffled[split_index:]
    y_train = label_data_shuffled[:split_index]
    y_test = label_data_shuffled[split_index:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
    
def create_train_test_splits(data_dict):
    """
    Creates train-test splits for multiple datasets and stores them in a dictionary.

    Args:
        data_dict (dict): A dictionary where keys represent dataset names (e.g., 'low_noise', 'high_noise')
                          and values are tuples containing the signal data (X) and label data (y)
                          for each dataset.

    Returns:
        dict: A dictionary containing the train-test splits for each dataset.
              Keys are dataset names, and values are dictionaries with keys 'X_train', 'X_test',
              'y_train', 'y_test'.
    """
    splits_dict = {}
    for dataset_name, (X, y) in data_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # Add random_state for reproducibility
        splits_dict[dataset_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    return splits_dict
    
def save_splits_dict(splits_dict, save_path):
    """
    Saves the entire splits_dict to a file using pickle.

    Args:
        splits_dict (dict): The dictionary containing the train-test splits.
        save_path (str): The path to save the dictionary to.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(splits_dict, f)

    print(f"Splits dictionary saved to: {save_path}")
    

# PART 3: DATA VISUALIZATION HELPERS 

# For me this is for visualization but can be necessary for preprocessing as well, depending on the set-up
def one_hot_encode_single_sample(label_sample, categories):
    """
    One-hot encodes string categories for single samples.

    Parameters:
        label_sample (list or np.ndarray): A 1D array-like structure containing the labels to encode.
        categories (list or np.ndarray): A 1D array-like structure containing the unique categories 
                                           to use for encoding.

    Returns:
        np.ndarray: A 2D numpy array of one-hot encoded labels.
    """
    one_hot_encoded = np.array([[1 if cat == label else 0 for cat in categories] for label in label_sample])
    return one_hot_encoded
    
def visualize_training_data(label_sample, signal_sample, sampling_frequency=250, duration=60, time_window=(0, 10), category_labels=['noise', 'beta'], title="Ground Truth of When Bursts Occur"):
    """
    Visualizes training data with ground truth labels and signal plot.

    Args:
        label_sample (np.ndarray): 1D array of ground truth labels.
        signal_sample (np.ndarray): 1D array of signal data.
        sampling_frequency (int, optional): Sampling frequency of the data. Defaults to 250.
        duration (int, optional): Duration of the data in seconds. Defaults to 60.
        time_window (tuple, optional): Time window to display in seconds (start, end). Defaults to (0, 10).
        category_labels (list, optional): Labels for the categories in the stackplot. Defaults to ['noise', 'beta'].
    """

    time_vec = np.linspace(0, duration, duration * sampling_frequency, endpoint=False)

    fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    # Create a one-hot encoded matrix for stackplot
    label_sample = one_hot_encode_single_sample(label_sample, categories=np.unique(label_sample))  # Use unique labels for categories

    # Visualizing burst labels as colors
    ax[0].stackplot(time_vec, label_sample.T, labels=category_labels[:label_sample.shape[1]], alpha=0.7)  # Adjust labels based on categories
    ax[0].set_xlabel("Time (s)")
    ax[0].set_yticks([])  # y-axis does not mean anything
    ax[0].set_title("Ground Truth of When Bursts Occur")
    ax[0].legend()

    # the period of blue: bursts
    ax[1].plot(time_vec, signal_sample, color="k")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Signal Plot")

    plt.xlim(time_window[0], time_window[1])
    plt.tight_layout()
    plt.show()
    
def visualize_feature_data(label_sample, signal_sample, feature_sample, sampling_frequency=250, duration=60, time_window=(0, 10), category_labels=None):
    """
    Visualizes feature data (Hilbert Amplitude & Wavelets) with ground truth labels and signal plot.

    Args:
        label_sample (np.ndarray): 1D array of ground truth labels.
        signal_sample (np.ndarray): 1D array of signal data.
        feature_sample (np.ndarray): 2D array of feature data. (sequence_len, num_features)
        sampling_frequency (int, optional): Sampling frequency of the data. Defaults to 250.
        duration (int, optional): Duration of the data in seconds. Defaults to 60.
        time_window (tuple, optional): Time window to display in seconds (start, end). Defaults to (0, 10).
        category_labels (list, optional): Labels for the categories in the stackplot. Defaults to ['noise', 'beta'].
    """

    time_vec = np.linspace(0, duration, duration * sampling_frequency, endpoint=False)
    
    num_features = feature_sample.shape[1]

    fig, ax = plt.subplots(3, 1, figsize=(20, 16), sharex=True)

    if category_labels is not None:
        unique_label_list = list(range(len(category_labels)))
        # Create a one-hot encoded matrix for stackplot
        one_hot_label_sample = one_hot_encode_single_sample(label_sample, categories=unique_label_list)  # Use unique labels for categories
    else:
        one_hot_label_sample = one_hot_encode_single_sample(label_sample, categories=np.unique(category_labels))  # Use unique labels for categories

    # Visualizing burst labels as colors
    ax[0].stackplot(time_vec, one_hot_label_sample.T, labels=category_labels, alpha=0.7)  # Adjust labels based on categories
    ax[0].set_xlabel("Time (s)")
    ax[0].set_yticks([])  # y-axis does not mean anything
    ax[0].set_title("Ground Truth of When Bursts Occur")
    ax[0].legend()

    # the period of blue: bursts
    ax[1].plot(time_vec, signal_sample, color="k")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Signal Plot")

    # Plot LFP signal
    for feature_idx in range(num_features):
      if num_features == 1:
          ax[2].plot(time_vec, feature_sample[:, feature_idx], alpha=0.7, label='Hilbert Amplitdue')
          ax[2].set_title(f"Hilbert Amplitude")
      else:
          # These will determine for what freq band the wavelet is adjusted
          mean_freqs = np.logspace(np.log10(4), np.log10(100), num=20)
          mean_freqs_round = np.round(mean_freqs).astype(int)
          ax[2].plot(time_vec, feature_sample[:, feature_idx], alpha=0.7, label=f'Wavelets at Freq: {mean_freqs_round[feature_idx]}')
          ax[2].set_title(f"All Wavelets")
    # Overall maximum
    max_value = np.max(feature_sample)
    ax[2].stackplot(time_vec, label_sample*max_value, labels=["Ground Truth"], alpha=0.7, colors=['lightblue'])
    ax[2].set_xlabel("Time (s)")
    ax[2].legend()
    ax[2].set_xlim(time_window)
    ax[2].set_ylim(0, max_value)

    plt.xlim(time_window[0], time_window[1])
    plt.tight_layout()
    plt.show()
    
# PART 4: MACHINE LEARNING MODEL SET-UP

class SaveModelsCallback(Callback):
    """
    Callback to save copies of the model at specified intervals during training.

    This callback clones and saves the model's weights after every `save_frequency`
    epochs. This allows you to access and analyze the model's state at different
    stages of training.

    Attributes:
        save_frequency (int): The frequency (in epochs) at which to save the model. 
                              Defaults to 1, meaning the model is saved after every epoch.
    """
    def __init__(self, save_frequency=1):
        super().__init__()
        self.saved_models = []  # List to store models after each epoch
        self.save_frequency = save_frequency  # Frequency of saving models (every n epochs)

    def on_epoch_end(self, epoch, logs=None):
        # Clone and save the model only if the epoch is a multiple of save_frequency
        if (epoch + 1) % self.save_frequency == 0 or epoch == 0:
            model_copy = tf.keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())
            self.saved_models.append((epoch, model_copy))  # Save as (epoch, model) pair
            print(f"Model saved at epoch {epoch + 1}")

def create_simple_lstm_attention_model(input_shape):
    """Creates an LSTM model with attention for burst detection.

    Args:
        input_shape (tuple): The shape of the input data 
                             (timesteps, features).

    Returns:
        tf.keras.Model: The compiled LSTM model.
                             
    Model Architecture:
    
    The model consists of the following layers:
    1. Input Layer: Takes input of shape `input_shape`.
    2. LSTM Layer: An LSTM layer with 32 units. It returns 
       sequences and the hidden/cell states. Output shape:
       (batch_size, timesteps, 32).
    3. Dense Layer: A dense layer with sigmoid activation 
       for binary classification. Output shape: (batch_size, timesteps, 1).

    """
    inputs = tf.keras.Input(input_shape)

    # LSTM layer with return_sequences=True and return_state=True
    lstm_layer = LSTM(32,
                      return_sequences=True,
                      return_state=True,
                      name='lstm_layer')
    lstm_out, state_h, state_c = lstm_layer(inputs)

    outputs = Dense(1, activation="sigmoid")(lstm_out)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(X_train, y_train, X_test, y_test, input_shape=None, epochs=40, batch_size=16, custom_model=None, save_frequency=1, save_intermediate_models=True):
    """
    Trains an LSTM model with the given training and testing data.

    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        y_test (np.ndarray): Testing labels.
        input_shape (tuple, optional): Input shape for the LSTM model.
                                      If None, it is inferred from X_train.shape.
                                      Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 40.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        custom_model (tf.keras.Model, optional): A custom Keras model to use for training. 
                                                 If provided, this model will be used instead of 
                                                 creating a new one. Defaults to None.
        save_frequency (int, optional): Frequency of saving models (every n epochs). Defaults to 1.
        save_intermediate_models (bool, optional): Whether to save intermediate models during training. 
                                                   Defaults to True.

    Returns:
        tuple: A tuple containing the trained model, training history, and the SaveModelsCallback instance (if used).
              (model, history, save_models_callback)
    """
    # Infer input shape if not provided
    if input_shape is None:
        input_shape = (X_train.shape[1], X_train.shape[2])

    # Use custom model if provided, otherwise create the default model
    if custom_model is not None:
        model = custom_model
    else:
        model = create_simple_lstm_attention_model(input_shape)  # Or your default model creation function
    
    # The model shall come pre-compiled. That way we can allow for more dynamic compile settings.
    # model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()

    # Create SaveModelsCallback if save_intermediate_models is True
    if save_intermediate_models:
        save_models_callback = SaveModelsCallback(save_frequency=save_frequency)
        callbacks = [save_models_callback]
    else:
        save_models_callback = None  # No callback for saving intermediate models
        callbacks = []
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=callbacks,
    )
    return model, history, save_models_callback

def train_gate_model(X_train, y_train, X_test, y_test, input_shape=None, epochs=40, batch_size=16, custom_model=None, save_frequency=1, save_intermediate_models=True):
    """
    Trains an LSTM model with the given training and testing data.

    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        y_test (np.ndarray): Testing labels.
        input_shape (tuple, optional): Input shape for the LSTM model.
                                      If None, it is inferred from X_train.shape.
                                      Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 40.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        custom_model (tf.keras.Model, optional): A custom Keras model to use for training. 
                                                 If provided, this model will be used instead of 
                                                 creating a new one. Defaults to None.
        save_frequency (int, optional): Frequency of saving models (every n epochs). Defaults to 1.
        save_intermediate_models (bool, optional): Whether to save intermediate models during training. 
                                                   Defaults to True.

    Returns:
        tuple: A tuple containing the trained model, training history, and the SaveModelsCallback instance (if used).
              (model, history, save_models_callback)
    """
    # Infer input shape if not provided
    if input_shape is None:
        input_shape = (X_train.shape[1], X_train.shape[2])

    # Use custom model if provided, otherwise create the default model
    if custom_model is not None:
        model = custom_model
    else:
        model = create_simple_lstm_attention_model(input_shape)  # Or your default model creation function
    
    # The model shall come pre-compiled. That way we can allow for more dynamic compile settings.
    # model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()

    # Create SaveModelsCallback if save_intermediate_models is True
    if save_intermediate_models:
        save_models_callback = SaveModelsCallback(save_frequency=save_frequency)
        callbacks = [save_models_callback]
    else:
        save_models_callback = None  # No callback for saving intermediate models
        callbacks = []
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=callbacks,
    )
    return model, history, save_models_callback

def train_models_from_splits_dict(splits_dict, epochs=40, batch_size=16, model_architecture=None, **kwargs):
    """
    Trains multiple LSTM models based on a dictionary of data splits.

    Parameters:
        splits_dict (dict): A dictionary containing data splits, where keys
                            represent split names (e.g., 'low_noise', 'high_noise')
                            and values are dictionaries containing 'X_train', 'y_train',
                            'X_test', 'y_test', and optionally 'input_shape'.
        epochs (int, optional): Number of training epochs. Defaults to 40.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        **kwargs: Any additional keyword arguments to be passed to the 
                  `train_model` function. These can include 
                  `input_shape`, `custom_model`, `save_frequency`, 
                  `save_intermediate_models`, and any other arguments
                  accepted by `train_model`.

    Returns:
        dict: A dictionary containing trained models, histories, and 
              save_models_callbacks for each split, with keys 
              corresponding to the split names.
    """
    trained_models = {}
    for split_name, split_data in splits_dict.items():
        print(f"Training model for split: {split_name}")
        
        X_train = split_data['X_train'] # Needed for input shape determination

        # Create a new model instance for each split using the provided architecture
        if model_architecture:
            model = model_architecture(X_train.shape[1:])  
        else:
            model = create_lstm_attention_model(X_train.shape[1:])
        
        # Pass the keyword arguments to train_model
        model, history, save_models_callback = train_model(
            split_data['X_train'],
            split_data['y_train'],
            split_data['X_test'],
            split_data['y_test'],
            epochs=epochs,
            batch_size=batch_size,
            custom_model=model,
            **kwargs  # Pass all other keyword arguments
        )

        trained_models[split_name] = {
            'model': model,
            'history': history,
            'save_models_callback': save_models_callback
        }
    return trained_models

def trained_models_with_performance(trained_models_dict, splits_dict):
    """
    Exports a dictionary of trained models with subkeys for the model and performance metrics.

    Args:
        trained_models_dict (dict): A dictionary containing trained models and their histories.
        splits_dict (dict): A dictionary containing the data splits for each model.

    Returns:
        dict: A dictionary with model names as keys and subkeys for 'model' and performance metrics.
    """

    exported_models_dict = {}
    for model_name, model_data in trained_models_dict.items():
        model = model_data['model']

        # Get X_test and y_test from splits_dict for the current model
        X_test = splits_dict[model_name]['X_test']
        y_test = splits_dict[model_name]['y_test']

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions if necessary
        y_pred = y_pred.reshape(y_test.shape)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Store the model and performance metrics in the dictionary
        exported_models_dict[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'mcc': mcc,
        }

    return exported_models_dict
    
from IPython.display import HTML

from IPython.display import HTML
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef

def print_model_performances(trained_models_dict, splits_dict):
    """
    Culminates and prints the final performances of trained LSTM models in an HTML table,
    including F1 score, accuracy, recall, and Matthews Correlation Coefficient.
    Uses the splits_dict to access the appropriate train/test splits for each model.

    Args:
        trained_models_dict (dict): A dictionary containing trained models and their histories.
        splits_dict (dict): A dictionary containing the data splits for each model.
    Returns:
        sorted_performance_data: A dictionary with model names as keys and their performance metrics.
    """

    performance_data = {}
    for model_name, model_data in trained_models_dict.items():
        model = model_data['model']

        # Get X_test and y_test from splits_dict for the current model
        X_test = splits_dict[model_name]['X_test']
        y_test = splits_dict[model_name]['y_test']

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions if necessary
        
        # Reshape y_pred to match y_test's shape
        y_pred = y_pred.reshape(-1, y_test.shape[1]) # Reshape to (num_samples, timesteps)
        
        # Flatten both y_test and y_pred for metric calculation
        y_test = y_test.flatten() 
        y_pred = y_pred.flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        performance_data[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'mcc': mcc,
        }
    
    # Sort performance_data by accuracy in descending order
    sorted_performance_data = dict(sorted(performance_data.items(), key=lambda item: item[1]['accuracy'], reverse=True))

    # Create the HTML table string with padding
    html_table = "<table style='border-collapse: collapse; width: 100%;'><tr><th>Model Name</th><th>Accuracy</th><th>F1 Score</th><th>Recall</th><th>MCC</th></tr>"
    for model_name, metrics in sorted_performance_data.items():
        html_table += f"<tr><td style='padding: 10px;'>{model_name}</td><td style='padding: 10px;'>{metrics['accuracy']:.4f}</td><td style='padding: 10px;'>{metrics['f1']:.4f}</td><td style='padding:                         10px;'>{metrics['recall']:.4f}</td><td style='padding: 10px;'>{metrics['mcc']:.4f}</td></tr>"
    html_table += "</table>"

    # Display the HTML table
    display(HTML(html_table))

    return sorted_performance_data
    
from sklearn.metrics import classification_report

def multi_class_trained_models_with_performance(trained_models_dict, splits_dict):
    """
    Exports a dictionary of trained models with subkeys for the model and performance metrics.
    This serves as a precursor for exporting this type of dictionary to file

    Args:
        trained_models_dict (dict): A dictionary containing trained models and their histories.
        splits_dict (dict): A dictionary containing the data splits for each model.

    Returns:
        dict: A dictionary with model names as keys and subkeys for 'model' and performance metrics.
    """
    # Define class names
    class_names = ['noise', 'theta', 'alpha', 'beta', 'gamma'] 

    exported_models_dict = {}
    for model_name, model_data in trained_models_dict.items():
        model = model_data['model']

        # Get X_test and y_test from splits_dict for the current model
        X_test = splits_dict[model_name]['X_test']
        y_test = splits_dict[model_name]['y_test']

        y_pred = model.predict(X_test)

        n_samples, n_timesteps, n_cats  = y_pred.shape  # (8, 15000, 5)
        y_pred_reshaped = y_pred.reshape(-1, n_cats, 1)
        y_test_classes = y_test.flatten()
        y_pred_classes = np.argmax(y_pred_reshaped, axis=1).flatten()

        # Generate and print classification report
        report = classification_report(
            y_test_classes,
            y_pred_classes,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        print(report)

        # Calculate Matthew Coefficient as extra metric
        mcc = matthews_corrcoef(y_test_classes, y_pred_classes)

        # Store the model and performance metrics in the dictionary
        exported_models_dict[model_name] = {
            'model': model,
            'mcc': mcc,
        }
        exported_models_dict[model_name].update(report)

    return exported_models_dict
    
# PART 5: MACHINE LEARNING ANALYSIS ATTRIBUTES
    
def get_model_sizes(trained_models_dict):
    """
    Calculates the size of temporarily stored models in megabytes (MB).

    Args:
        trained_models_dict (dict): A dictionary containing trained models,
                                    histories, and save_models_callbacks for each split.
                                    (output of train_models_from_splits)

    Returns:
        dict: A dictionary with model names as keys and their sizes in MB as values,
              including a 'total_size_mb' key for the total size of all models.
    """
    model_sizes = {}
    total_size_mb = 0  # Initialize total size

    for split_name, split_data in trained_models_dict.items():
        save_models_callback = split_data['save_models_callback']
        total_size_bytes = 0
        for epoch, model in save_models_callback.saved_models:
          # Temporarily save the model to a file
          model.save('/content/temp_model.h5')

          # Get the file size
          size = os.path.getsize('/content/temp_model.h5')

          # Remove the temporary file
          os.remove('/content/temp_model.h5')
          total_size_bytes += size
        size_in_mb = total_size_bytes / (1024 * 1024)  # Convert to MB
        model_sizes[split_name] = size_in_mb  # Store size in MB
        total_size_mb += size_in_mb  # Accumulate total size

    model_sizes['total_size_mb'] = total_size_mb  # Add total size to the dictionary

    return model_sizes

def prep_hidden_cell_development(testing_data, save_models_callback):
    """
    Extracts the hidden and cell states at each timestep and epoch for the testing data.

    Args:
        save_models_callback: Trained LSTM model list with 40 models depeneding on epoch amount
        test_data: Numpy array of shape (samples, timesteps, features), representing the testing data.

    Returns:
        A dictionary containing:
            - 'hidden_states_activation': Activation of hidden states over epochs and samples.
            - 'epoch_model_predictions': Predictions of the model for each epoch.
            - 'hidden_states_output': Hidden state outputs per epoch.
            - 'cell_states_output': Cell state outputs per epoch.
            - 'dense_weights': Dense layer weights at each epoch.
    """
    hidden_s = []
    cell_s = []
    hidden_states_activation = []
    epoch_model_pred = []
    dense_weights = []
    dense_bias = []

    for epoch, saved_model in save_models_callback.saved_models:
        print(f"Epoch {epoch + 1}:")

        lstm_layer = saved_model.layers[-2]
        lstm_out, state_h, state_c = lstm_layer(saved_model.input)

        #This is a samller model, that opens up the hidden layer, being the LSTM
        # 32 cells
        hidden_state_model = Model(inputs=saved_model.input, outputs=[lstm_out, state_h, state_c])

        # Generate hidden states for a sample input
        hidden_states, hidden_state_out, cell_state_out = hidden_state_model.predict(testing_data)  # Shape: (1, time_steps, hidden_units)

        model_pred = saved_model.predict(testing_data)

        d_weights, d_bias = saved_model.layers[-1].get_weights()

        # Print shapes ones at the beginning to get an understanding
        if epoch == 0:
          print(f"Hidden States Shape: {hidden_states.shape}")
          print(f"Hidden State Output Shape: {hidden_state_out.shape}")
          print(f"Cell State Output Shape: {cell_state_out.shape}")
          print(f"Dense Weights Output Shape: {d_weights.shape}")
          print(f"Dense Bias Output Shape: {d_bias.shape}")

        hidden_s.append(hidden_state_out)
        cell_s.append(cell_state_out)

        hidden_states_activation.append(hidden_states)
        epoch_model_pred.append(model_pred)

        dense_weights.append(d_weights)
        dense_bias.append(d_bias)

    # Reformating the data to numpy arrays for future interpretation
    hs_activation_on_pred = np.array(hidden_states_activation)
    hs_activation_on_pred = np.transpose(hs_activation_on_pred, (0, 1, 3, 2))

    cell_s = np.array(cell_s)
    hidden_s = np.array(hidden_s)
    epoch_model_pred = np.array(epoch_model_pred)
    dense_weights = np.array(dense_weights)
    dense_bias = np.array(dense_bias)

    final_dict = {
        'hidden_states_activation': hs_activation_on_pred,
        'epoch_model_predictions': epoch_model_pred,
        'final_hidden_states': hidden_s,
        'final_cell_states': cell_s,
        'dense_weights': dense_weights,
        'dense_bias': dense_bias
        }

    return final_dict
    
def extract_hidden_data_from_trained_models(trained_models_dict, splits_dict):
    """
    Applies the prep_hidden_cell_development function to all trained models
    and returns a dictionary with extracted data.

    Parameters:
        trained_models_dict (dict): A dictionary containing trained models,
                                    histories, and save_models_callbacks for each split.
                                    (output of train_models_from_splits)

    Returns:
        dict: A dictionary containing the extracted data for each split,
              with keys corresponding to the split names.
    """
    extracted_data_dict = {}
    for split_name, split_data in trained_models_dict.items():
        print(f"Extracting data for split: {split_name}")
        model = split_data['model']
        save_models_callback = split_data['save_models_callback']
        # Assuming you want to use the X_test data from the split for extraction
        testing_data = splits_dict.get(split_name, {}).get('X_test', None)  # Handle potential KeyError gracefully
        if testing_data is not None:
            extracted_data = prep_hidden_cell_development(testing_data, save_models_callback)
            extracted_data_dict[split_name] = extracted_data
        else:
            print(f"Warning: Skipping extraction for {split_name} as 'X_test' data is missing.")
    return extracted_data_dict
    
# PART 6: SAVING/LOADING ML-MODELS or DATA

def save_extracted_data(extracted_data_dict, save_path):
    """
    Saves the extracted_data_dict to a specified path using pickle.

    Parameters:
        extracted_data_dict (dict): The dictionary containing the extracted data.
        save_path (str): The path where the dictionary should be saved.
                            Includes the filename and extension (e.g., 'data/extracted_data.pkl').
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(extracted_data_dict, f)

    print(f"Extracted data saved to: {save_path}")
    

def save_trained_models_and_performance(exported_models_dict, save_directory):
    """
    Saves trained models and their performance metrics to separate files.

    Args:
        exported_models_dict (dict): A dictionary containing trained models and performance metrics.
        save_directory (str): The directory where the models and metrics will be saved.
    """
    os.makedirs(save_directory, exist_ok=True)  # Create the save directory if it doesn't exist

    performance_data = {}  # To store performance metrics

    for model_name, model_data in exported_models_dict.items():
        model = model_data['model']
        performance_metrics = {k: v for k, v in model_data.items() if k != 'model'}  # Extract performance metrics

        # Save the model
        model_save_path = os.path.join(save_directory, f"{model_name}.h5")
        model.save(model_save_path)

        # Store performance metrics in the dictionary
        performance_data[model_name] = performance_metrics

    # Save performance metrics to a separate file
    performance_save_path = os.path.join(save_directory, "performance_metrics.pkl")
    with open(performance_save_path, 'wb') as f:
        pickle.dump(performance_data, f)


def save_trained_models_and_performance(exported_models_dict, save_directory, performance_file_name, ):
    """
    Saves trained models and their performance metrics to separate files.

    Args:
        exported_models_dict (dict): A dictionary containing trained models and performance metrics.
        save_directory (str): The directory where the models and metrics will be saved.
    """
    os.makedirs(save_directory, exist_ok=True)  # Create the save directory if it doesn't exist

    performance_data = {}  # To store performance metrics

    for model_name, model_data in exported_models_dict.items():
        model = model_data['model']
        performance_metrics = {k: v for k, v in model_data.items() if k != 'model'}  # Extract performance metrics

        # Save the model
        model_save_path = os.path.join(save_directory, f"{model_name}.h5")
        model.save(model_save_path)

        # Store performance metrics in the dictionary
        performance_data[model_name] = performance_metrics

    # Save performance metrics to a separate file
    performance_save_path = os.path.join(save_directory, f"{performance_file_name}.pkl")
    with open(performance_save_path, 'wb') as f:
        pickle.dump(performance_data, f)

def expand_data_configs(data_configs, num_samples):
    """
    Expands the data_configs dictionary to include noise levels in keys,
    adds noise_ratios, and sets the number of samples.

    Args:
        data_configs (dict): Original data configuration dictionary.
        num_samples (int): Number of samples to use.

    Returns:
        dict: Expanded data configuration dictionary.
    """
    expanded_configs = {}
    noise_list = [
            ("low", (3,4)),  # Example noise ratios, adjust as needed
            ("mid", (2,3)),
            ("high", (1,2)),
            ("veryHigh", (0,1)), # Important to keep them as one word for string decocatenation later
        ]
    for model_name, config in data_configs.items():
        for noise_level, noise_ratio in noise_list:
            new_key = f"{model_name}_{noise_level}"  # Create new key with noise level
            expanded_configs[new_key] = config.copy()  # Copy original config
            expanded_configs[new_key]["noise_ratios"] = noise_ratio  # Add noise_ratios key
            expanded_configs[new_key]["n_samples"] = num_samples  # Set number of samples
    return expanded_configs
    
# TEST-DATA GENERATION

def load_models_from_directory(directory_path, intermediate=False):
    """
    Loads all models (`.h5` files) from a directory into a dictionary.

    Args:
        directory_path (str): The path to the directory containing the models.
        intermediate (bool): Whether to load intermediate models as well.

    Returns:
        dict: A dictionary where keys are model names (filenames without extension)
              and values are dictionaries with 'model' (final model) and
              'epoch_models' (intermediate models, if loaded).
    """
    loaded_models = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".h5"):  # Check if it's a Keras model file
            model_name = os.path.splitext(filename)[0]  # Get filename without extension
            model_path = os.path.join(directory_path, filename)
            model = tf.keras.models.load_model(model_path)
            model.name = model_name  # Set the model's name to its filename
            loaded_models[model_name] = {'model': model}  # Store final model

            if intermediate:  # Load intermediate models if requested
                intermediate_dir = os.path.join(directory_path, f"{model_name}_intermediate")
                if os.path.exists(intermediate_dir):
                    epoch_models = []
                    for epoch_filename in os.listdir(intermediate_dir):
                        if epoch_filename.endswith(".h5"):
                            epoch_model_path = os.path.join(intermediate_dir, epoch_filename)
                            epoch_model = tf.keras.models.load_model(epoch_model_path)
                            epoch_model.name = epoch_filename  # Set the model's name to its filename
                            epoch_models.append(epoch_model)
                    loaded_models[model_name]['epoch_models'] = epoch_models
                else:
                    print(f"Warning: Intermediate directory not found for {model_name}")

    return loaded_models
    
def add_model_to_data_dict(restructured_data_dict, loaded_models):
  """
  Adds a 'model' subkey to the restructured_data_dict,
  matching keys with loaded_models and removing noise level from key.

  Args:
      restructured_data_dict (dict): The dictionary to modify.
      loaded_models (dict): Dictionary of loaded models.

  Returns:
      dict: The modified restructured_data_dict.
  """

  for data_key in restructured_data_dict.keys():
    # Remove noise level from data_key
    model_key_base = "_".join(data_key.split("_")[:-1])
    print(model_key_base)

    # Find matching model in loaded_models
    if model_key_base in loaded_models:
        print(data_key, model_key_base)
        restructured_data_dict[data_key]["model"] = loaded_models[model_key_base]["model"]

        # Add epoch_models if they exist
        if 'epoch_models' in loaded_models[model_key_base]:
            restructured_data_dict[data_key]["epoch_models"] = loaded_models[model_key_base]['epoch_models']

  return restructured_data_dict
  
  
def split_sequences_keep_data(X, y, new_seq_len):
    """
    Splits input sequences and their per-timestep labels into smaller sequences of length `new_seq_len`.

    Args:
        X (numpy array): Input data of shape (samples, seq_len, features).
        y (numpy array): Labels of shape (samples, seq_len) or (samples, seq_len, ...).
        new_seq_len (int): Desired sequence length.

    Returns:
        X_new: New input data with shape (new_samples, new_seq_len, features).
        y_new: New labels with shape (new_samples, new_seq_len, ...).
    """
    num_samples, original_seq_len, num_features = X.shape
    if original_seq_len % new_seq_len != 0:
        print(f"Warning: Original seq_len ({original_seq_len}) is not a multiple of new_seq_len ({new_seq_len}).")

    # Number of new samples per original sequence
    num_segments = original_seq_len // new_seq_len

    # Create new datasets by reshaping
    X_new = np.reshape(X[:, :num_segments * new_seq_len, :],
                       (-1, new_seq_len, num_features))  # Shape: (samples * num_segments, new_seq_len, features)
    y_new = np.reshape(y[:, :num_segments * new_seq_len, ...],
                       (-1, new_seq_len, *y.shape[2:]))  # Adjust labels to match input shape

    return X_new, y_new
