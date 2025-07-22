# -*- coding: utf-8 -*-
# @Time       : 2025/7/16
# @Author     : Marverlises
# @File       : main.py
# @Description: Main script to generate daily paper reports.
import feedparser
import arxiv
import os
import yaml
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.recommender import rerank_paper
from typing import Optional, List

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.paper import ArxivPaper, LocalPaper
from src.llm import set_global_llm
from src.pdf_layout_analyzer import set_global_pdf_layout_analyzer
from src.base_corpus import *


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


def get_arxiv_papers(arxiv_config: dict) -> List[ArxivPaper]:
    """Fetches papers from arXiv with flexible filtering and sorting options, using Beijing time.

    Args:
        arxiv_config (dict): Configuration dictionary containing:
            - query (str): Search query for arXiv.
            - start_date (Optional[str]): Start date in 'YYYY-MM-DD' format.
            - end_date (Optional[str]): End date in 'YYYY-MM-DD' format.
            - max_results (Optional[int]): Maximum number of results to return.
            - sort_by (str): Field to sort by ('submittedDate', 'relevance', 'lastUpdatedDate').
            - sort_order (str): Sort order ('ascending', 'descending').
            - debug (bool): If True, shows progress bar.

    Returns:
        List[ArxivPaper]: List of ArxivPaper objects matching the query and filters.

    Raises:
        ValueError: If invalid parameters are provided (e.g., invalid date format or range).
    """
    """
     query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[int] = None,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        debug: bool = False
    """
    query = arxiv_config['query']
    start_date = arxiv_config.get('start_time', None)
    end_date = arxiv_config.get('end_time', None)
    max_results = arxiv_config.get('max_results', 20)
    sort_by = arxiv_config.get('sort_by', 'submittedDate')
    sort_order = arxiv_config.get('sort_order', 'descending')

    beijing_tz = ZoneInfo("Asia/Shanghai")

    def parse_date(date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(tzinfo=beijing_tz)
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use 'YYYY-MM-DD'.")

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    if start_dt and end_dt and start_dt > end_dt:
        raise ValueError("start_date must be earlier than or equal to end_date")

    current_time = datetime.now(beijing_tz)
    if end_dt and end_dt > current_time:
        logger.warning("end_date is in the future; setting to current Beijing time")
        end_dt = current_time

    sort_by_map = {
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate
    }
    sort_order_map = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending
    }

    if sort_by not in sort_by_map:
        raise ValueError(f"Invalid sort_by value. Choose from {list(sort_by_map.keys())}")
    if sort_order not in sort_order_map:
        raise ValueError(f"Invalid sort_order value. Choose from {list(sort_order_map.keys())}")

    client = arxiv.Client(num_retries=10, delay_seconds=10)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by_map[sort_by],
        sort_order=sort_order_map[sort_order]
    )

    papers = []
    bar = tqdm(desc="Fetching arXiv paper details")

    try:
        for result in client.results(search):
            published_beijing = result.published.astimezone(beijing_tz)
            if start_dt and published_beijing.date() < start_dt.date():
                continue
            if end_dt and published_beijing.date() > end_dt.date():
                continue
            papers.append(ArxivPaper(result))
            bar.update(1)
            if max_results and len(papers) >= max_results:
                break
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
    finally:
        bar.close()

    return papers


def get_arxiv_paper_daily(query: str, debug: bool = False) -> list[ArxivPaper]:
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


def process_paper(paper, dir_name, paper_download_retry=3):
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
    local_config = config['local']
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
    # Get the base paper seed
    corpus = get_corpus(pref_source=pref_source, z_config=z_config, local_config=local_config)

    logger.info("Fetching new papers from arXiv...")
    papers = get_arxiv_paper_daily(a_config['query'])
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

    # config process_paper
    partial(process_paper, paper_download_retry=app_config.get('paper_download_retry', 3))
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
