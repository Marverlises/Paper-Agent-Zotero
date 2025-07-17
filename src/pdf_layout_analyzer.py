# -*- coding: utf-8 -*-
# @Time       : 2025/7/17
# @Author     : Marverlises
# @File       : recommender.py
# @Description: Reranks papers based on Zotero corpus similarity.


import os
from pdf_craft import create_pdf_page_extractor, PDFPageExtractor, MarkDownWriter


class PDFLayoutAnalyzer:
    def __init__(self, model_dir_path: str, device: str = "cpu"):
        self.extractor: PDFPageExtractor = create_pdf_page_extractor(
            device=device,
            model_dir_path=model_dir_path
        )

    def extract(self, pdf: str, markdown_save_path: str = None, asset_dir: str = "images") -> iter:
        """
        Extracts layout information from a PDF file and optionally saves it in Markdown format.
        :param pdf: Path to the PDF file to be analyzed.
        :param markdown_save_path: Optional path to save the extracted information in Markdown format.
        :param asset_dir: Directory to save images and other assets.
        :return: An iterator yielding blocks of extracted information.
        """
        if markdown_save_path:
            md_writer = MarkDownWriter(markdown_save_path, asset_dir, "utf-8")
            with md_writer as md:
                for block in self.extractor.extract(pdf=pdf):
                    md.write(block)
        else:
            return self.extractor.extract(pdf=pdf)
