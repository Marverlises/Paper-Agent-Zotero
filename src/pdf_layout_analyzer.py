# -*- coding: utf-8 -*-
# @Time       : 2025/7/17
# @Author     : Marverlises
# @File       : recommender.py
# @Description: Reranks papers based on Zotero corpus similarity.


import os
from pathlib import Path

from loguru import logger

from pdf_craft import create_pdf_page_extractor, PDFPageExtractor, MarkDownWriter, OCRLevel


class PDFLayoutAnalyzer:
    def __init__(self, model_dir_path: str, device: str = "cuda", strict: bool = False):
        if strict:
            logger.warning("Strict mode is enabled. This may lead to slower processing and higher memory usage.")
            self.extractor: PDFPageExtractor = create_pdf_page_extractor(
                device=device,
                model_dir_path=model_dir_path
            )
        else:
            self.extractor: PDFPageExtractor = create_pdf_page_extractor(
                device=device,
                model_dir_path=model_dir_path,
                ocr_level=OCRLevel.OncePerLayout,
            )

    def extract_as_md(self, pdf: Path, markdown_save_path: str = None, asset_dir: str = "assets_dir") -> str:
        """
        Extracts layout information from a PDF file and optionally saves it in Markdown format or returns it as a string.
        :param pdf: Path to the PDF file to be analyzed.
        :param markdown_save_path: Optional path to save the extracted information in Markdown format.
        :param asset_dir: Directory to save images and other assets.
        :return: An iterator yielding blocks of extracted information.
        """
        logger.info(f"Starting PDF layout analysis for: {pdf}")
        # Validate input parameters
        if not isinstance(pdf, Path):
            pdf = Path(pdf)
        if not pdf.exists():
            raise FileNotFoundError(f"The specified PDF file does not exist: {pdf}")

        try:
            # Extract layout information and save to Markdown if a path is provided
            if markdown_save_path:
                logger.info(f"Saving extracted layout information to Markdown file: {markdown_save_path}")
                base_dir = os.path.dirname(markdown_save_path)
                if markdown_save_path and not os.path.exists(base_dir):
                    os.makedirs(base_dir, exist_ok=True)

                md_writer = MarkDownWriter(markdown_save_path, asset_dir, "utf-8")
                with md_writer as md:
                    for block in self.extractor.extract(pdf=str(pdf)):
                        md.write(block)
                return markdown_save_path
            else:
                logger.info(f"No markdown_save_path provided, returning extracted text as string.")
                temp_md_file = os.path.join(os.path.dirname(pdf), asset_dir, os.path.basename(pdf),
                                            os.path.splitext(os.path.basename(pdf))[0] + ".md")
                if not os.path.exists(os.path.dirname(temp_md_file)):
                    os.makedirs(os.path.dirname(temp_md_file), exist_ok=True)
                md_writer = MarkDownWriter(temp_md_file, asset_dir, "utf-8")
                with md_writer as md:
                    for block in self.extractor.extract(pdf=str(pdf).replace('.pdf', '')):
                        md.write(block)
                with open(temp_md_file, 'r', encoding='utf-8') as f:
                    result_text = f.read()
                # Clean up the temporary file
                return result_text
        except Exception as e:
            logger.error(f"An error occurred during PDF layout analysis: {e}")
            raise e

    def extract_images_in_order(self, pdf_path: str, output_dir: str):
        """ Extracts images from a PDF file in the order they appear and saves them to the specified output directory."""
        os.makedirs(output_dir, exist_ok=True)
        image_counter = 0

        for page_index, blocks, page_image in self.extractor.extract_enumerated_blocks_and_image(pdf_path):
            page_image.save(f"{output_dir}/page_{page_index + 1}.png")

            for block in blocks:
                # Check if the block has an image attribute
                if hasattr(block, 'image'):
                    image_counter += 1
                    block.image.save(f"{output_dir}/image_{image_counter:03d}.png")


# Global instance of PDFLayoutAnalyzer
_pdf_layout_analyzer: PDFLayoutAnalyzer = None


def set_global_pdf_layout_analyzer(model_dir_path: str, device: str = "cuda", strict: bool = False):
    """
    Initializes the global PDFLayoutAnalyzer instance.
    :param model_dir_path: Path to the directory containing the PDF layout analysis model.
    :param device: Device to run the model on (e.g., 'cuda' or 'cpu').
    :param strict: If True, enables strict mode for more accurate but slower processing.
    """
    global _pdf_layout_analyzer
    _pdf_layout_analyzer = PDFLayoutAnalyzer(model_dir_path=model_dir_path, device=device, strict=strict)


def get_pdf_layout_analyzer() -> PDFLayoutAnalyzer:
    """
    Retrieves the global PDFLayoutAnalyzer instance.
    :raises Exception: If the PDFLayoutAnalyzer has not been initialized.
    :return: The global PDFLayoutAnalyzer instance.
    """
    global _pdf_layout_analyzer
    if _pdf_layout_analyzer is None:
        raise Exception("PDFLayoutAnalyzer has not been initialized. Call set_global_pdf_layout_analyzer() first.")
    return _pdf_layout_analyzer
