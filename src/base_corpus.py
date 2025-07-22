# -*- coding: utf-8 -*-
# @Time       : 2025/7/22 16:39
# @Author     : Marverlises
# @File       : base_corpus.py
# @Description: PyCharm
import glob
import pytz
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
from pyzotero import zotero
from tqdm import tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from src.paper import ArxivPaper, LocalPaper


def get_zotero_corpus(id: str, key: str, recency_months: int = -1) -> list[dict]:
    """Fetches and filters the Zotero corpus based on recency."""
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']: c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']

    # Filter by recency if enabled
    if recency_months > 0:
        now = datetime.now(pytz.utc)
        months_ago = now - relativedelta(months=recency_months)

        recent_corpus = []
        for item in corpus:
            try:
                date_added = isoparse(item['data']['dateAdded'])
                if date_added >= months_ago:
                    recent_corpus.append(item)
            except (ValueError, TypeError):
                # Handle cases where dateAdded is missing or malformed
                continue

        logger.info(
            f"Filtered to {len(recent_corpus)} papers from the last {recency_months} months for recommendations.")
        corpus = recent_corpus

    def get_collection_path(col_key: str) -> str:
        """Recursively builds the full path for a Zotero collection."""
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']

    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus


def get_local_corpus(path: str) -> list[dict]:
    """Fetches and processes local PDFs to build a corpus."""
    if not os.path.isdir(path):
        logger.error(f"Local PDF path does not exist or is not a directory: {path}")
        return []

    pdf_files = glob.glob(os.path.join(path, '**', '*.pdf'), recursive=True)

    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {path}. Processing...")

    corpus = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_pdf = {executor.submit(LocalPaper, pdf_file): pdf_file for pdf_file in pdf_files}

        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing Local PDFs"):
            paper = future.result()
            if paper.title != "Unknown Title":
                corpus.append({
                    'data': {
                        'title': paper.title,
                        'abstractNote': paper.abstract
                    }
                })
                logger.success(f"Successfully processed: {os.path.basename(paper.pdf_path)}")
            else:
                logger.warning(f"Failed to process: {os.path.basename(paper.pdf_path)}")

    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    _, filename = mkstemp()
    with open(filename, 'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir='../')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    try:
        os.remove(filename)
    except OSError as e:
        logger.error(f"Error removing temporary file {filename}: {e}")
    return new_corpus


def get_corpus(pref_source: str = 'zotero', z_config: dict = None, local_config: dict = None) -> list[dict]:
    if pref_source == 'zotero':
        logger.info("Retrieving Zotero corpus as the preference source...")
        corpus = get_zotero_corpus(z_config['id'], z_config['key'], z_config.get('recency_months', -1))
        logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
        if z_config.get('ignore'):
            ignore_pattern = "\n".join(z_config['ignore'])
            logger.info(f"Ignoring papers in collections:\n{ignore_pattern}")
            corpus = filter_corpus(corpus, ignore_pattern)
            logger.info(f"Finished filtering. {len(corpus)} papers remaining.")
    elif pref_source == 'local':
        logger.info("Using local PDF directory as the preference source...")
        corpus = get_local_corpus(local_config['path'])
        logger.info(f"Built a corpus from {len(corpus)} local papers.")
    else:
        logger.error(f"Invalid preference_source: '{pref_source}'. Please use 'zotero' or 'local'.")
        sys.exit(1)
    return corpus
