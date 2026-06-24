"""
Resolves the data directory used by all scripts in this folder (G_F_/v4).

Set the G_PROJECT_DATA environment variable to your local data path.
If it is unset, the code falls back to a ./data folder next to this file.

Example (zsh/bash):
    export G_PROJECT_DATA=/path/to/your/G_project_data

This is the same convention already used in G_F_/v2/config.py.
"""
import os

DATA_DIR = os.environ.get(
    "G_PROJECT_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
