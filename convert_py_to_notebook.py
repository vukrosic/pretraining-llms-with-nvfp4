#!/usr/bin/env python3
"""
Script to convert Python files with markdown cells to Jupyter notebooks.

This script parses Python files that contain markdown cells marked with:
    # %% [markdown]
    
And code cells marked with:
    # %%
    
And converts them to proper .ipynb format.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any


def parse_python_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a Python file and extract markdown and code cells.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of cell dictionaries for the notebook
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cells = []
    current_cell_type = None
    current_content = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for cell markers
        if re.match(r'^#\s*%%\s*(\[markdown\])?\s*$', line.strip()):
            # Save previous cell if it exists
            if current_content:
                source_lines = [line.rstrip('\n\r') for line in current_content]
                if current_cell_type == "markdown":
                    # Remove leading # and single space from markdown lines
                    cleaned_lines = []
                    for source_line in source_lines:
                        if source_line.startswith('# '):
                            cleaned_lines.append(source_line[2:])
                        elif source_line.startswith('#'):
                            cleaned_lines.append(source_line[1:])
                        else:
                            cleaned_lines.append(source_line)
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": cleaned_lines
                    })
                else:
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": source_lines
                    })
            
            # Determine new cell type
            if '[markdown]' in line:
                current_cell_type = "markdown"
            else:
                current_cell_type = "code"
            
            current_content = []
        else:
            current_content.append(line)
        
        i += 1
    
    # Handle the last cell
    if current_content:
        source_lines = [line.rstrip('\n\r') for line in current_content]
        if current_cell_type == "markdown":
            # Remove leading # and single space from markdown lines
            cleaned_lines = []
            for source_line in source_lines:
                if source_line.startswith('# '):
                    cleaned_lines.append(source_line[2:])
                elif source_line.startswith('#'):
                    cleaned_lines.append(source_line[1:])
                else:
                    cleaned_lines.append(source_line)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cleaned_lines
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            })
    
    return cells


def create_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a notebook structure from cells.
    
    Args:
        cells: List of cell dictionaries
        
    Returns:
        Notebook dictionary
    """
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def convert_file(input_path: Path, output_path: Path = None) -> None:
    """
    Convert a Python file to a Jupyter notebook.
    
    Args:
        input_path: Path to the input Python file
        output_path: Path to the output notebook file (optional)
    """
    if output_path is None:
        output_path = input_path.with_suffix('.ipynb')
    
    print(f"Converting {input_path} to {output_path}")
    
    # Parse the Python file
    cells = parse_python_file(input_path)
    
    # Create the notebook
    notebook = create_notebook(cells)
    
    # Write the notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_path} with {len(cells)} cells")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Python files with markdown cells to Jupyter notebooks"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input Python file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Path to the output notebook file (default: input_file.ipynb)"
    )
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    try:
        convert_file(args.input_file, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
