# -*- coding: utf-8 -*-
# @Time       : 2025/7/16
# @Author     : Marverlises
# @File       : paper.py
# @Description: Defines the ArxivPaper class for handling paper data and operations.

from typing import Optional, LiteralString
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from pathlib import Path
from src.llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack
import os
import fitz  # PyMuPDF
from .llm import get_llm, set_global_llm
from .pdf_layout_analyzer import get_pdf_layout_analyzer


class ArxivPaper:
    def __init__(self, paper: arxiv.Result):
        self.paper = paper
        self.score = None
        self._is_downloaded = False
        self.pdf_path = None  # To store path of downloaded PDF

    @property
    def arxiv_id(self):
        return self.paper.get_short_id()

    @property
    def title(self):
        return self.paper.title

    @property
    def summary(self):
        return self.paper.summary

    @property
    def authors(self):
        return self.paper.authors

    @property
    def pdf_url(self):
        return self.paper.pdf_url

    @cached_property
    def pdf_text(self) -> str:
        """Extracts text from the downloaded PDF. Caches the result."""
        if not self.pdf_path.exists():
            logger.error(f"PDF for '{self.title}' not available at '{self.pdf_path}'. Cannot extract text.")
            return ""

        logger.info(f"Extracting text from {self.pdf_path}...")
        try:
            text = get_pdf_layout_analyzer().extract_as_md(pdf=self.pdf_path).strip()
            logger.success(f"Successfully extracted text for '{self.title}'.")
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF text from {self.pdf_path}: {e}")
            return ""

    @cached_property
    def images_in_order(self) -> list[LiteralString | str | bytes] | None:
        """Extracts images url from pdf_text."""
        pattern = r'!\[.*?\]\((assets_dir\\.*?\.png)\)'
        matches = re.findall(pattern, self.pdf_text)
        if not matches:
            logger.debug(f"No images found in {self.arxiv_id}.")
            return None
        # Convert to absolute paths
        assets_dir = self.pdf_path.parent
        image_urls = [os.path.join(assets_dir, match) for match in matches]
        if not image_urls:
            logger.debug(f"No valid image URLs found in {self.arxiv_id}.")
            return None
        logger.info(f"Found {len(image_urls)} images in {self.arxiv_id}.")
        return image_urls

    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count', 0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count', 0) == 0:
            return None
        return repo_list['results'][0]['url']

    @cached_property
    def tex(self) -> dict[str, str]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            file = self.paper.download_source(dirpath=tmpdirname)
            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                return None

            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None

            bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
            match len(bbl_file):
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl', '')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                        main_tex = None
                case _:
                    logger.debug(
                        f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                    main_tex = None
            if main_tex is None:
                logger.debug(
                    f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
            # read all tex files
            file_contents = {}
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode('utf-8', errors='ignore')
                # remove comments
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                # remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                content = re.sub(r'\\\\', '', content)
                # remove consecutive spaces
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content

            if main_tex is not None:
                main_source: str = file_contents[main_tex]
                # find and replace all included sub-files
                include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}',
                                                                                          main_source)
                for f in include_files:
                    if not f.endswith('.tex'):
                        file_name = f + '.tex'
                    else:
                        file_name = f
                    main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
            else:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                file_contents["all"] = None
        return file_contents

    @cached_property
    def tldr(self) -> str:
        """Generates a deep-dive summary of the paper using the full text."""
        # The PDF must be downloaded first for full text access.
        if not self._is_downloaded:
            logger.error(f"Cannot generate TLDR for '{self.title}' because PDF is not downloaded.")
            return "Error: PDF not downloaded. Cannot generate detailed summary."

        full_text = self.pdf_text
        if not full_text:
            logger.warning(
                f"Could not generate TLDR for '{self.title}' because full text extraction failed. Using abstract instead.")
            # Fallback to abstract if full text is empty
            return get_llm().get_tldr(self.title, self.summary, self.summary)

        return get_llm().get_tldr(self.title, self.summary, full_text)

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            # search for affiliations
            possible_regions = [r'\\author.*?\\maketitle', r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
            match = next((m for m in matches if m), None)
            if match:
                information_region = match.group(0)
            else:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None
            prompt = f"Given the author information of a paper in latex format, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]'. Following is the author information:\n{information_region}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
            prompt = enc.decode(prompt_tokens)
            llm = get_llm()
            affiliations = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            try:
                affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
                affiliations = eval(affiliations)
                affiliations = list(set(affiliations))
                affiliations = [str(a) for a in affiliations]
            except Exception as e:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affiliations

    def download_pdf(self, dirpath: str = './', title=None):
        """Downloads the PDF and stores its path."""
        if self._is_downloaded:
            return
        title = re.sub(r'[^\w\s-]', '', self.title).replace(' ', '_')  # Clean title for filename
        filename = f"{self.arxiv_id}.{title}.pdf"
        # replace dots in filename to avoid issues with file systems
        filename = re.sub(r'\.+', '.', filename)
        try:
            # The arxiv library handles the download and saving.
            self.paper.download_pdf(dirpath=dirpath, filename=filename)
            self._is_downloaded = True
            logger.info(f"Successfully downloaded '{filename}'.")
            self.pdf_path = Path(os.path.join(dirpath, filename))
        except Exception as e:
            logger.error(f"Failed to download PDF for '{self.arxiv_id}': {e}")
            self.pdf_path = None  # Reset path on failure
            self._is_downloaded = False


class LocalPaper:
    """Represents a paper from a local PDF file."""

    def __init__(self, pdf_path: str, strict: bool = False):
        self.pdf_path = Path(pdf_path)
        self._text = None
        self._title = None
        self._abstract = None
        self._get_metadata(strict)

    def _get_text(self, strict: bool = False) -> str:
        """Extracts text from the first few pages of the PDF."""
        if self._text is None:
            try:
                if strict:
                    self._text = get_pdf_layout_analyzer().extract_as_md(pdf=self.pdf_path)
                else:
                    doc = fitz.open(self.pdf_path)
                    text = ""
                    # Extract text from the first 2 pages (usually enough for metadata)
                    for page_num in range(min(2, doc.page_count)):
                        text += doc[page_num].get_text()
                    self._text = text.strip()
                    doc.close()
            except Exception as e:
                logger.error(f"Failed to extract text from {self.pdf_path}: {e}")
                self._text = ""
        return self._text

    def _get_metadata(self, strict: bool = False):
        """
        Extracts metadata (title and abstract) from the PDF using LLM.
        :param strict:  if True, uses strict mode for more accurate extraction.
        :return: None
        """

        text = self._get_text(strict)
        if not text:
            self._title = "Unknown Title"
            self._abstract = "Could not extract abstract."
            return

        try:
            metadata = get_llm().get_pdf_metadata(text)
            self._title = metadata.get('title', 'Unknown Title')
            self._abstract = metadata.get('abstract', 'Could not extract abstract.')
        except Exception as e:
            logger.error(f"Failed to extract metadata from {self.pdf_path} using LLM: {e}")
            self._title = os.path.basename(self.pdf_path)
            self._abstract = text[:500]  # Fallback to first 500 chars

    @property
    def title(self):
        return self._title

    @property
    def abstract(self):
        return self._abstract
