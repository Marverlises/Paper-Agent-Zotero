# -*- coding: utf-8 -*-
# @Time       : 2025/7/16
# @Author     : Marverlises
# @File       : main.py
# @Description: Main script to generate daily paper reports.

import arxiv
import os
import sys
import yaml
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import feedparser
import pytz
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
from pyzotero import zotero
from src.recommender import rerank_paper
from tqdm import tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from src.paper import ArxivPaper, LocalPaper
from src.llm import set_global_llm
from src.pdf_layout_analyzer import set_global_pdf_layout_analyzer


def load_config(path="src/my_config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(
            f"Configuration file not found at {path}. Please copy 'config.yaml.template' to 'config.yaml' and fill in your details.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)


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


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    """Fetches new papers from arXiv."""
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")

    all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.arxiv_announce_type == 'new']
    if not all_paper_ids:
        return []

    papers = []
    bar = tqdm(total=len(all_paper_ids), desc="Fetching arXiv paper details")
    search_batches = [all_paper_ids[i:i + 50] for i in range(0, len(all_paper_ids), 50)]

    for batch_ids in search_batches:
        search = arxiv.Search(id_list=batch_ids)
        try:
            batch_results = [ArxivPaper(p) for p in client.results(search)]
            papers.extend(batch_results)
            bar.update(len(batch_results))
        except Exception as e:
            logger.error(f"Error fetching a batch of papers: {e}")
    bar.close()

    return papers


def process_paper(paper, dir_name):
    """
    Processes a single paper: downloads PDF, generates TLDR.
    Returns a dictionary with paper info for reporting.
    """
    try:
        # Download the PDF
        paper.download_pdf(dirpath=dir_name, title=paper.title)

        # Generate TLDR
        tldr = paper.tldr
        logger.success(f"Generated deep-dive summary for: '{paper.title}'")

        return {
            "title": paper.title,
            "authors": ', '.join([author.name for author in paper.authors]),
            "tldr": tldr,
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "arxiv_id": paper.arxiv_id,
            "score": paper.score
        }
    except Exception as e:
        logger.error(f"Failed to process paper '{paper.title}': {e}")
        return None


if __name__ == '__main__':
    config = load_config()

    z_config = config['zotero']
    a_config = config['arxiv']
    app_config = config['app']
    llm_config = config['llm']
    pdf_analyzer_config = config['pdf_layout_analyzer']
    pref_source = config.get('preference_source', 'zotero')  # Default to zotero

    # Setup logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Setup LLM first, as it might be needed for local PDF processing
    if llm_config['use_llm_api']:
        logger.info("Using API-based LLM.")
        set_global_llm(
            api_key=llm_config['openai_api_key'],
            base_url=llm_config['openai_api_base'],
            model=llm_config['model_name'],
            lang=llm_config['language']
        )
    else:
        logger.info("Using Local LLM.")
        set_global_llm(lang=llm_config['language'])

    # Initialize PDF Layout Analyzer
    set_global_pdf_layout_analyzer(pdf_analyzer_config['model_dir_path'], device=pdf_analyzer_config['device'],
                                   strict=pdf_analyzer_config['strict'])

    corpus = []
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
        local_config = config['local']
        corpus = get_local_corpus(local_config['path'])
        logger.info(f"Built a corpus from {len(corpus)} local papers.")
    else:
        logger.error(f"Invalid preference_source: '{pref_source}'. Please use 'zotero' or 'local'.")
        sys.exit(1)

    logger.info("Fetching new papers from arXiv...")
    papers = get_arxiv_paper(a_config['query'])
    if not papers:
        logger.info("No new papers found today.")
        exit(0)

    logger.info("Reranking papers based on your Zotero library...")
    papers = rerank_paper(papers, corpus, pref_source)
    if app_config['max_paper_num'] > 0:
        papers = papers[:app_config['max_paper_num']]

    # Create output directory
    now = datetime.now()
    dir_name = now.strftime("%Y-%m-%d-%H")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # Change dir to absolute path
    dir_name = os.path.abspath(dir_name)
    processed_papers = []
    with ThreadPoolExecutor(max_workers=app_config['max_workers']) as executor:
        future_to_paper = {executor.submit(process_paper, paper, dir_name): paper for paper in papers}

        for future in tqdm(as_completed(future_to_paper), total=len(papers),
                           desc="Generating Report and Downloading PDFs"):
            result = future.result()
            if result:
                processed_papers.append(result)

    # Sort papers by their original reranked score
    processed_papers.sort(key=lambda p: p['score'], reverse=True)

    # Write the final report
    report_path = os.path.join(dir_name, "Daily_Report.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"# Daily Paper Report - {now.strftime('%Y-%m-%d %H:%M')}\n\n")
        report_file.write("## Today's Recommended Papers\n\n")

        for i, paper_data in enumerate(processed_papers):
            report_file.write(f"### {i + 1}. {paper_data['title']}\n\n")
            report_file.write(f"**Authors:** {paper_data['authors']}\n\n")
            report_file.write(f"**Deep-Dive Summary:**\n{paper_data['tldr']}\n\n")
            report_file.write(f"**Original Abstract:** {paper_data['summary']}\n\n")
            report_file.write(f"**PDF Link:** [{paper_data['arxiv_id']}]({paper_data['pdf_url']})\n\n")
            report_file.write("---\n\n")

    logger.success(f"All tasks completed. Report and PDFs are saved in '{dir_name}'.")
