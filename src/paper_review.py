# -*- coding: utf-8 -*-
# @Time       : 2025/7/22 16:09
# @Author     : Marverlises
# @File       : paper_review.py
# @Description: Get a literature review from the papers listed in the folder
import glob
import os


class PaperReview:
    """A class to review papers in a specified folder."""

    def __init__(self, source_folder: str, file_type: str = "md"):
        """ file type: md or pdf """
        self.folder_path = source_folder
        self.file_type = file_type.lower()
        self.papers = self._load_papers()

    def _load_papers(self):
        """
        Load papers from the specified folder.
        :return: List of paper file paths.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")
        if self.file_type not in ["md", "pdf"]:
            raise ValueError("Unsupported file type. Only 'md' and 'pdf' are supported.")
        return glob.glob(os.path.join(self.folder_path, f"*.{self.file_type}"))

    def review_papers(self):
        """
        Review the loaded papers and return a summary.
        :return: Summary of the papers.
        """
        summaries = []
        for paper in self.papers:
            summary = self._review_paper(paper)
            summaries.append(summary)
        return summaries

    def _review_paper(self, paper):
        """
        Review a single paper and return its summary.
        :param paper: Path to the paper file.
        :return: Summary of the paper.
        """
        if self.file_type.lower() == "pdf":
            # Todo
            pass



if __name__ == '__main__':
    folder_path = r'D:\Workspace\ProgramWorkspace\Python\GithubProject\Paper-Agent-Zotero-github\2025-07-18-13\assets_dir'  # Replace with your folder path
    paper_review = PaperReview(folder_path)
    summaries = paper_review.review_papers()

    for summary in summaries:
        print(summary)
