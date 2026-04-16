"""Document loading utilities supporting PDF and plain-text files."""

import logging
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from rag_system.exceptions import DocumentLoadError

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentLoader:
    """Loads PDF and TXT files into LangChain ``Document`` objects.

    Usage::

        loader = DocumentLoader()
        docs = loader.load("path/to/file.pdf")
        all_docs = loader.load_directory("path/to/docs/")
    """

    def load(self, path: Union[str, Path]) -> List[Document]:
        """Load a single file.

        Args:
            path: Absolute or relative path to a ``.pdf`` or ``.txt`` file.

        Returns:
            A list of ``Document`` objects (one per page for PDFs, one for TXTs).

        Raises:
            DocumentLoadError: If the file is missing, unsupported, or cannot be
                parsed.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise DocumentLoadError(str(path), "file does not exist")

        if not file_path.is_file():
            raise DocumentLoadError(str(path), "path is not a file")

        extension = file_path.suffix.lower()
        if extension not in _SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                str(path),
                f"unsupported file type '{extension}'; supported: {sorted(_SUPPORTED_EXTENSIONS)}",
            )

        try:
            if extension == ".pdf":
                return self._load_pdf(file_path)
            return self._load_txt(file_path)
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(str(path), str(exc)) from exc

    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
    ) -> List[Document]:
        """Load all supported files from a directory.

        Args:
            directory: Path to the directory containing documents.
            recursive: When *True*, walk subdirectories as well.

        Returns:
            A flat list of ``Document`` objects loaded from every supported
            file found in *directory*.

        Raises:
            DocumentLoadError: If *directory* does not exist or is not a
                directory.
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise DocumentLoadError(str(directory), "directory does not exist")
        if not dir_path.is_dir():
            raise DocumentLoadError(str(directory), "path is not a directory")

        pattern = "**/*" if recursive else "*"
        documents: List[Document] = []
        errors: List[str] = []

        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                try:
                    documents.extend(self.load(file_path))
                except DocumentLoadError as exc:
                    logger.warning("Skipping '%s': %s", file_path, exc)
                    errors.append(str(exc))

        if not documents and errors:
            raise DocumentLoadError(
                str(directory),
                f"failed to load any documents; errors: {'; '.join(errors)}",
            )

        logger.info(
            "Loaded %d document(s) from '%s'%s",
            len(documents),
            directory,
            f" ({len(errors)} error(s) skipped)" if errors else "",
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pdf(path: Path) -> List[Document]:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        if not pages:
            raise DocumentLoadError(str(path), "PDF contains no extractable text")
        return pages

    @staticmethod
    def _load_txt(path: Path) -> List[Document]:
        try:
            loader = TextLoader(str(path), encoding="utf-8")
            return loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(str(path), encoding="latin-1")
            return loader.load()
