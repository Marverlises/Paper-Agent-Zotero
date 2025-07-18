# -*- coding: utf-8 -*-
# @Time       : 2025/7/18 13:29
# @Author     : Marverlises
# @File       : utils.py
# @Description: PyCharm
from pathvalidate import sanitize_filename
from typing import Optional


def normalize_filename(filename: str, max_length: int = 255, sanitize: bool = True) -> str:
    """
    Normalize a filename by replacing all dots except the last one with underscores
    and optionally sanitizing invalid characters.

    Args:
        filename (str): The input filename (e.g., "1234.5678.My.Paper.Title.pdf")
        max_length (int): Maximum length of the filename (default: 255)
        sanitize (bool): Whether to sanitize invalid characters (default: True)

    Returns:
        str: The normalized filename (e.g., "1234_5678_My_Paper_Title.pdf")

    Raises:
        ValueError: If the filename is empty
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Replace all dots except the last one with underscores
    try:
        last_dot_index = filename.rindex('.')
        normalized = filename[:last_dot_index].replace('.', '_') + filename[last_dot_index:]
    except ValueError:
        # No dot found in the filename
        normalized = filename

    # Sanitize invalid characters if requested
    if sanitize:
        normalized = sanitize_filename(normalized)

    # Truncate to max_length, preserving the extension
    if len(normalized) > max_length:
        try:
            last_dot_index = normalized.rindex('.')
            extension = normalized[last_dot_index:]
            base = normalized[:last_dot_index]
            base = base[:max_length - len(extension)]
            normalized = base + extension
        except ValueError:
            # No extension, truncate the whole filename
            normalized = normalized[:max_length]

    return normalized

if __name__ == '__main__':
    file_name = r'2507.11997v1.Can_LLMs_Find_Fraudsters_Multi-level_LLM_Enhanced_Graph_Fraud_Detection.pdf'
    normalized_name = normalize_filename(file_name, max_length=255, sanitize=True)
    print(f"Original filename: {normalized_name}")
