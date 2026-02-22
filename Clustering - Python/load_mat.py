# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:03:57 2025

@author: RBU-Kilosort2
"""

import h5py
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os

def load_mat(filepath=None):
    """
    Load all top-level variables from a MATLAB v7.3 .mat file into a dictionary.

    Parameters:
        filepath (str): Optional path to the .mat file.

    Returns:
        data_dict (dict): Dictionary where keys are variable names and values are NumPy arrays.
    """

    # Use file dialog if no path provided
    if filepath is None:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            "Select a MATLAB v7.3 file",
            "",
            "MAT-files (*.mat);;All files (*)"
        )
        if not filepath:
            raise ValueError("No file selected.")

    # Validate file path
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load variables
    data_dict = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            try:
                data = f[key][:]
                data_dict[key] = np.array(data)
            except Exception as e:
                print(f"Could not read variable '{key}': {e}")

    return data_dict

