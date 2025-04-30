# This will be a helper file
#
# It contains a wide range of helper functions to simplify the actual code I work with

# PART 0: IMPORTS

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, TimeDistributed, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback



from IPython.display import display, HTML


# PART 1: LOAD in DATA

def load_extracted_data(file_path):
    """
    Loads the extracted_data_dict from a specified path using pickle.

    Parameters:
        file_path (str): The path where the dictionary should be loaded from.
                            Includes the filename and extension (e.g., 'data/extracted_data.pkl').

    Returns:
        dict: The extracted_data_dict loaded from the file.
    """
    with open(file_path, 'rb') as f:
        extracted_data_dict = pickle.load(f)

    print(f"Extracted data loaded from: {file_path}")
    return extracted_data_dict

def load_splits_dict(file_path):
    """
    Loads the splits_dict from a file using pickle.

    Args:
        file_path (str): The path to the file containing the splits_dict.

    Returns:
        dict: The loaded splits_dict.
    """
    with open(file_path, 'rb') as f:
        splits_dict = pickle.load(f)

    print(f"Splits dictionary loaded from: {file_path}")
    return splits_dict
    
def explain_dict_structure(data_dict, indent=0, max_depth=3):
    """
    Explains the structure of a nested dictionary by printing keys and shapes of values.

    Parameters:
        data_dict (dict): The dictionary to explain.
        indent (int, optional): Indentation level for printing. Defaults to 1.
        max_depth (int, optional): Maximum depth to explore nested dictionaries. Defaults to 3.
    """
    for key, value in data_dict.items():
        print(' ' * indent + f'{key}:')
        if isinstance(value, dict) and max_depth > 0:
            explain_dict_structure(value, indent + 3, max_depth - 1)
        else:
            try:
                shape_str = f"Shape: {value.shape}"
            except AttributeError:
                shape_str = f"Type: {type(value)}"
            print(' ' * (indent + 3) + shape_str)
    print("")    


# PART 1: DENSE WEIGHTS

def rank_by_absolute_value(array):
    """
    Ranks the elements of an array by their absolute values, preserving their original values.
    Also calculates and returns the min, max, median, and mean of the array.

    Args:
        array (np.ndarray): Input 1D NumPy array.

    Returns:
        tuple: (ranked_list, min_val, max_val, median_val, mean_val)
    """
    sorted_indices = np.argsort(np.abs(array))[::-1]
    sorted_by_abs = array[sorted_indices]
    ranked_list = [(rank + 1, index, value) for rank, (index, value) in enumerate(zip(sorted_indices, sorted_by_abs))]

    min_val = np.min(array)
    max_val = np.max(array)
    median_val = np.median(array)
    mean_val = np.mean(array)

    return ranked_list, min_val, max_val, median_val, mean_val

import pandas as pd

def rank_dense_weights_for_all_models_binary(extracted_data_dict, epoch=-1, output=True):
    """Ranks and displays the dense weights for all models in the extracted data dictionary.

    Args:
        extracted_data_dict (dict): Dictionary containing extracted data from the models.
        epoch (int, optional): The epoch to analyze. Defaults to -1 (last epoch).
    """
    ranked_weights_dict = {}
    for model_name, model_data in extracted_data_dict.items():
        dense_weights = model_data['dense_weights']

        # Reshape if necessary
        if len(dense_weights.shape) > 2:
            dense_weights = dense_weights.reshape(dense_weights.shape[0], dense_weights.shape[1])

        # Select weights for the specified epoch
        weights_for_epoch = dense_weights[epoch]

        # Rank weights
        ranked_weights, min_val, max_val, median_val, mean_val = rank_by_absolute_value(weights_for_epoch)

        ranked_weights_dict[model_name] = ranked_weights
        # Create DataFrame (without styling)
        df = pd.DataFrame(ranked_weights, columns=['Rank', 'Cell Index', 'Dense Weight'])
        df = df.set_index('Rank')  # Set 'Rank' as the index

        # Display DataFrame with basic formatting
        print(f"\nModel: {model_name} (Epoch {epoch})")
        print(f"Min: {min_val:.4f}, Max: {max_val:.4f}, Median: {median_val:.4f}, Mean: {mean_val:.4f}")
        print(df.to_string())  # Use to_string() for basic formatting
    if output:
        return ranked_weights_dict

    
def calculate_weighted_activations(lstm_outputs, dense_weights, dense_bias):
  """Calculates the weighted contributions of LSTM cells to the final output.

  Args:
    lstm_outputs (np.ndarray): The output of the LSTM layer (shape: [epochs, sample_size, timesteps, lstm_units]).
    dense_weights (np.ndarray): The weights of the dense layer (shape: [epochs, lstm_units]).
    dense_bias (np.ndarray): The bias of the dense layer (shape: [epochs]).

  Returns:
    np.ndarray: The weighted activations of the LSTM cells (shape: [epochs, batch_size, timesteps]).
  """

  weighted_contributions_by_epoch = []
  for epoch in range(lstm_outputs.shape[0]):
      epoch_outputs = lstm_outputs[epoch]
      cell_weights = dense_weights[epoch]
      bias_value = dense_bias[epoch]

      # Apply weights and bias to LSTM outputs
      weighted_activations = (epoch_outputs * cell_weights[:, np.newaxis]) + bias_value

      weighted_contributions_by_epoch.append(weighted_activations)

  return np.array(weighted_contributions_by_epoch)
    
# VISUALIZE HIDDEN ACTIVATION

import matplotlib.cm as cm
import matplotlib.colors as mcolors


def visualize_hidden_activation(extracted_data_dict, tt_splits_dict, directory_name='low_noise_beta_binary', 
                                sample_number=4, top_n=4, epoch_to_visualize=-1, time_limit=(0,5),
                                weighted_cells=False, sequence_duration=None, 
                                show_worst_cells=True):
  """Visualizes the hidden activation of an LSTM model for a specific sample.

  Args:
        extracted_data_dict (dict): Dictionary containing extracted data from the model, including hidden states and predictions.
        tt_splits_dict (dict): Dictionary containing train-test splits, including X_test and y_test.
        directory_name (str, optional): The noise level of the data to visualize ('low_noise_beta_binary' or 'high_noise_beta_binary'). Defaults to 'low_noise_beta_binary'.
        sample_number (int, optional): The index of the sample to visualize. Defaults to 4.
        top_n (int, optional): The number of top and bottom LSTM cells to highlight based on their weights. Defaults to 4.
        epoch_to_visualize (int, optional): The epoch to visualize the hidden states from. Defaults to 7.
        time_limit (tuple, optional): The time range to display on the x-axis (in seconds). Defaults to (55, 60).
        weighted_cells (bool, optional): Whether to display the weighted cell activations. Defaults to False.
        sequence_duration (int, optional): The duration of the input sequence in seconds. If None, it will be inferred from the signal length. Defaults to None.
        show_worst_cells (bool, optional): Whether to display the worst performing LSTM cells. Defaults to True.


  Returns:
    None: Displays the visualization using matplotlib.
  """
  # Create custom color lists with variations of green and red
  top_colors = [mcolors.to_rgba('green', alpha=1 - (i / (top_n + 1))) for i in range(top_n)] # More intense green, less alpha variation
  bottom_colors = [mcolors.to_rgba('red', alpha=1 - (i / (top_n + 1))) for i in range(top_n)] # More intense red, less alpha variation

  # Extract data for the specified noise level
  X_test = tt_splits_dict[directory_name]['X_test'][:,:,0] # Gets only the signal (works for featue Xs as well)
  y_test = tt_splits_dict[directory_name]['y_test']
  hidden_states_activation = extracted_data_dict[directory_name]['hidden_states_activation']
  dense_weights = extracted_data_dict[directory_name]['dense_weights']
  dense_bias = extracted_data_dict[directory_name]['dense_bias']
  epoch_model_predictions = extracted_data_dict[directory_name]['epoch_model_predictions']

  dense_weights = dense_weights.reshape(dense_weights.shape[0], dense_weights.shape[1])

  # Define time vector based on sequence_duration
  sampling_frequency = 250  # Hz
  if sequence_duration is None:  # If sequence_duration is not provided, infer it from the signal length
      sequence_duration = len(X_test[sample_number].flatten()) / sampling_frequency
  time_vec = np.linspace(0, sequence_duration, int(sequence_duration * sampling_frequency), endpoint=False)  # Ensure time_vec length matches signal length

  # Create visualization with potential extra subplot for weighted cells
  num_subplots = 3 + int(weighted_cells)  # Add a subplot if weighted_cells is True
  fig, ax = plt.subplots(num_subplots, 1, figsize=(20, 14 + 6 * int(weighted_cells)), sharex=True) 

  # Extract signal and ground truth for the sample
  signal_sample = X_test[sample_number].flatten()
  ground_truth = y_test[sample_number].flatten()

  # Get top and bottom LSTM cell IDs based on weights
  final_weights, _, _, _, _ = au.rank_by_absolute_value(dense_weights[-1]) # Assuming au.rank_by_absolute_value is defined elsewhere
  cell_ids_good = [item[1] for item in final_weights[:top_n]]
  cell_ids_bad = [item[1] for item in final_weights[-top_n:]]

  print(f"Model: {directory_name} \nSample: {sample_number}")
  print(f"Top {top_n} LSTM Cells: {cell_ids_good}")
  print(f"Bottom {top_n} LSTM Cells: {cell_ids_bad}")

  # Extract hidden states for the specified epoch
  epoch_lstm_act = hidden_states_activation[epoch_to_visualize]
  model_pred = epoch_model_predictions[epoch_to_visualize]
  model_pred_binary = np.rint(model_pred).astype(int)

  # Plot LFP signal
  ax[0].plot(time_vec, signal_sample, label='Signal', alpha=1, color="#00138a")
  ax[0].set_xlabel("Time (s)")
  ax[0].set_title(f"True LFP Signal (Sample: {sample_number})")
  ax[0].legend()
  ax[0].set_xlim(time_limit)

  # Plot model predictions and ground truth (moved to ax[1])
  predictions = model_pred[sample_number].flatten()
  burst_indices = predictions > 0.5
  ax[1].plot(time_vec, predictions, label='Confidence', alpha=0.9)
  ax[1].stackplot(time_vec, ground_truth, labels=["Ground Truth"], alpha=0.7, colors=['lightblue'])
  ax[1].axhline(y=0.5, color='red', linestyle='-', linewidth=1, label='Threshold (0.5)')
  ax[1].set_xlabel("Time (s)")
  ax[1].set_title("Predicted Signal Type")
  ax[1].legend()
  ax[1].set_xlim(time_limit)
  ax[1].set_ylim(0, 1)

  # Plot hidden states for top and bottom cells (moved to ax[2])
  for i, cell_id in enumerate(cell_ids_good):
      ax[2].plot(time_vec, epoch_lstm_act[sample_number][cell_id], label=f'Good LSTM Output Cell {cell_id + 1}') 
  
  if show_worst_cells:  # Only plot worst cells if show_worst_cells is True
      for i, cell_id in enumerate(cell_ids_bad):
          ax[2].plot(time_vec, epoch_lstm_act[sample_number][cell_id], label=f'Bad LSTM Output Cell {cell_id + 1}', color=bottom_colors[-i])
  
  ax[2].set_title(f'Epoch: {epoch_to_visualize} LSTM Hidden States Testing Data NOT weight adjusted')
  ax[2].set_xlabel('Time Steps')
  ax[2].set_ylabel('Activation')
  ax[2].set_xlim(time_limit)
  ax[2].legend()


  # Plot weighted cell states if weighted_cells is True (added as a separate subplot below hidden states)
  if weighted_cells:
      weighted_activations = au.calculate_weighted_activations(hidden_states_activation, dense_weights, dense_bias)
      epoch_weighted_act = weighted_activations[epoch_to_visualize]
      
      for i, cell_id in enumerate(cell_ids_good):
          ax[3].plot(time_vec, epoch_weighted_act[sample_number][cell_id], label=f'Weighted LSTM Output {cell_id + 1}') 
      if show_worst_cells:
          for i, cell_id in enumerate(cell_ids_bad):
              ax[3].plot(time_vec, epoch_weighted_act[sample_number][cell_id], label=f'Weighted LSTM Output {cell_id + 1}', color="orange") 
      
      ax[3].set_title(f'Epoch: {epoch_to_visualize} LSTM Weighted Hidden States Testing Data')
      ax[3].set_xlabel('Time Steps')
      ax[3].set_ylabel('Weighted Activation')
      ax[3].set_xlim(time_limit)
      ax[3].legend()

  plt.show()


import matplotlib.cm as cm
import matplotlib.colors as mcolors


def visualize_hidden_activation_no_signal(extracted_data_dict, tt_splits_dict, sample_signal, directory_name='low_noise_beta_binary',
                                sample_number=4, top_n=4, epoch_to_visualize=-1, time_limit=(0,5),
                                weighted_cells=False, sequence_duration=None,
                                show_worst_cells=True):
  """Visualizes the hidden activation of an LSTM model for a specific sample.
     Requirements!  Needs at least one

  Args:
        extracted_data_dict (dict): Dictionary containing extracted data from the model, including hidden states and predictions.
        tt_splits_dict (dict): Dictionary containing train-test splits, including X_test and y_test.
        sample_signal (np.ndarray): The input signal for the sample. For visualiztion. Needs same shape as X_test.
        directory_name (str, optional): The noise level of the data to visualize ('low_noise_beta_binary' or 'high_noise_beta_binary'). Defaults to 'low_noise_beta_binary'.
        sample_number (int, optional): The index of the sample to visualize. Defaults to 4.
        top_n (int, optional): The number of top and bottom LSTM cells to highlight based on their weights. Defaults to 4.
        epoch_to_visualize (int, optional): The epoch to visualize the hidden states from. Defaults to 7.
        time_limit (tuple, optional): The time range to display on the x-axis (in seconds). Defaults to (55, 60).
        weighted_cells (bool, optional): Whether to display the weighted cell activations. Defaults to False.
        sequence_duration (int, optional): The duration of the input sequence in seconds. If None, it will be inferred from the signal length. Defaults to None.
        show_worst_cells (bool, optional): Whether to display the worst performing LSTM cells. Defaults to True.


  Returns:
    None: Displays the visualization using matplotlib.
  """

  # Create custom color lists with variations of green and red
  top_colors = [mcolors.to_rgba('green', alpha=1 - (i / (top_n + 1))) for i in range(top_n)] # More intense green, less alpha variation
  bottom_colors = [mcolors.to_rgba('red', alpha=1 - (i / (top_n + 1))) for i in range(top_n)] # More intense red, less alpha variation

  # Extract data for the specified noise level
  X_test = tt_splits_dict[directory_name]['X_test'] # Gets only the signal (works for featue Xs as well)
  y_test = tt_splits_dict[directory_name]['y_test']
  hidden_states_activation = extracted_data_dict[directory_name]['hidden_states_activation']
  dense_weights = extracted_data_dict[directory_name]['dense_weights']
  dense_bias = extracted_data_dict[directory_name]['dense_bias']
  epoch_model_predictions = extracted_data_dict[directory_name]['epoch_model_predictions']

  dense_weights = dense_weights.reshape(dense_weights.shape[0], dense_weights.shape[1])

  # Extract signal and ground truth for the sample
  # Get number of samples, sequence length, and features
  num_samples, seq_len, num_features = X_test.shape

  signal_sample = sample_signal[sample_number,:,0].flatten()
  ground_truth = y_test[sample_number].flatten()

  if len(signal_sample) != len(ground_truth):
    raise ValueError(f"Sample signal and ground truth must have the same length. You might need a different reference signal array. \nSignal Sample: {len(signal_sample)} \nGround Truth: {len(ground_truth)}")

  # Define time vector based on sequence_duration
  sampling_frequency = 250  # Hz
  if sequence_duration is None:  # If sequence_duration is not provided, infer it from the signal length
      sequence_duration = len(y_test[sample_number].flatten()) / sampling_frequency
  time_vec = np.linspace(0, sequence_duration, int(sequence_duration * sampling_frequency), endpoint=False)  # Ensure time_vec length matches signal length

  # Create visualization with potential extra subplot for weighted cells
  num_subplots = 4 + int(weighted_cells)  # Add a subplot if weighted_cells is True
  fig, ax = plt.subplots(num_subplots, 1, figsize=(20, 14 + 6 * int(weighted_cells)), sharex=True)

  # Get top and bottom LSTM cell IDs based on weights
  final_weights, _, _, _, _ = au.rank_by_absolute_value(dense_weights[-1]) # Assuming au.rank_by_absolute_value is defined elsewhere
  cell_ids_good = [item[1] for item in final_weights[:top_n]]
  cell_ids_bad = [item[1] for item in final_weights[-top_n:]]

  print(f"Model: {directory_name} \nSample: {sample_number}")
  print(f'Number of Features: {num_features}')
  print(f"Top {top_n} LSTM Cells: {cell_ids_good}")
  print(f"Bottom {top_n} LSTM Cells: {cell_ids_bad}")

  # Extract hidden states for the specified epoch
  epoch_lstm_act = hidden_states_activation[epoch_to_visualize]
  model_pred = epoch_model_predictions[epoch_to_visualize]
  model_pred_binary = np.rint(model_pred).astype(int)

  # Plot LFP signal (now on ax[0])
  ax[0].plot(time_vec, signal_sample, label='Signal', alpha=1, color="#00138a")
  ax[0].set_xlabel("Time (s)")
  ax[0].set_title(f"True LFP Signal (Sample: {sample_number})")
  ax[0].legend()
  ax[0].set_xlim(time_limit)

  # Plot LFP signal
  for feature_idx in range(num_features):
    if num_features == 1:
        ax[1].plot(time_vec, X_test[sample_number, :, feature_idx], alpha=0.7, label='Hilbert Amplitdue')
        ax[1].set_title(f"Hilbert Amplitude")
    else:
        # These will determine for what freq band the wavelet is adjusted
        mean_freqs = np.logspace(np.log10(4), np.log10(100), num=20)
        mean_freqs_round = np.round(mean_freqs).astype(int)
        ax[1].plot(time_vec, X_test[sample_number, :, feature_idx], alpha=0.7, label=f'Wavelets at Freq: {mean_freqs_round[feature_idx]}')
        ax[1].set_title(f"All Wavelets")

  ax[1].set_xlabel("Time (s)")
  ax[1].legend()
  ax[1].set_xlim(time_limit)

  # Plot model predictions and ground truth (moved to ax[1])
  predictions = model_pred[sample_number].flatten()
  ax[2].plot(time_vec, predictions, label='Confidence', alpha=0.9)
  ax[2].stackplot(time_vec, ground_truth, labels=["Ground Truth"], alpha=0.7, colors=['lightblue'])
  ax[2].axhline(y=0.5, color='red', linestyle='-', linewidth=1, label='Threshold (0.5)')
  ax[2].set_xlabel("Time (s)")
  ax[2].set_title("Predicted Signal Type")
  ax[2].legend()
  ax[2].set_xlim(time_limit)
  ax[2].set_ylim(0, 1)

  # Plot hidden states for top and bottom cells (moved to ax[2])
  for i, cell_id in enumerate(cell_ids_good):
      ax[3].plot(time_vec, epoch_lstm_act[sample_number][cell_id], label=f'Good LSTM Output Cell {cell_id + 1}')

  if show_worst_cells:  # Only plot worst cells if show_worst_cells is True
      for i, cell_id in enumerate(cell_ids_bad):
          ax[3].plot(time_vec, epoch_lstm_act[sample_number][cell_id], label=f'Bad LSTM Output Cell {cell_id + 1}', color=bottom_colors[-i])

  ax[3].set_title(f'Epoch: {epoch_to_visualize} LSTM Hidden States Testing Data NOT weight adjusted')
  ax[3].set_xlabel('Time Steps')
  ax[3].set_ylabel('Activation')
  ax[3].set_xlim(time_limit)
  ax[3].legend()


  # Plot weighted cell states if weighted_cells is True (added as a separate subplot below hidden states)
  if weighted_cells:
      weighted_activations = au.calculate_weighted_activations(hidden_states_activation, dense_weights, dense_bias)
      epoch_weighted_act = weighted_activations[epoch_to_visualize]

      for i, cell_id in enumerate(cell_ids_good):
          ax[4].plot(time_vec, epoch_weighted_act[sample_number][cell_id], label=f'Weighted LSTM Output {cell_id + 1}')
      if show_worst_cells:
          for i, cell_id in enumerate(cell_ids_bad):
              ax[4].plot(time_vec, epoch_weighted_act[sample_number][cell_id], label=f'Weighted LSTM Output {cell_id + 1}', color="orange")

      ax[4].set_title(f'Epoch: {epoch_to_visualize} LSTM Weighted Hidden States Testing Data')
      ax[4].set_xlabel('Time Steps')
      ax[4].set_ylabel('Weighted Activation')
      ax[4].set_xlim(time_limit)
      ax[4].legend()

  plt.show()