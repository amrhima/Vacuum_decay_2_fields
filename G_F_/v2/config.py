"""
Resolves the data directory used by all scripts in G_F_/v2/.

Set the G_PROJECT_DATA environment variable to your local data path.
If unset, falls back to ./data relative to this file.

Example (zsh/bash):
    export G_PROJECT_DATA=/path/to/your/G_project_data
"""
import os

DATA_DIR = os.environ.get(
    "G_PROJECT_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
