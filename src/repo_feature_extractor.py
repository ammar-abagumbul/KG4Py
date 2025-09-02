import os
import logging
import argparse
import pathlib
import json
import libcst
from tqdm import tqdm
from typing import List

from compact_json import CompactJSONEncoder

from extractors import extract_module_info

from config import BLACKLISTED_DIRECTORIES, WHITELISTED_EXTENSIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

VERBOSE = False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the Knowledge Graph construction tool.
    """
    parser = argparse.ArgumentParser(
        description="Construct a Knowledge Graph from a repository for RAG systems."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path or URL to the repository source (local directory or git repo URL).",
    )
    parser.add_argument(
        "output_json",
        type=str,
        default="repo_features.json",
        help="Output JSON file to store extracted module features.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for debugging."
    )
    parser.add_argument(
        "--clear-database",
        action="store_true",
        help="Clear all data in the Neo4j database before processing.",
    )
    parser.add_argument(
        "--embeddings", action="store_true", help="Generate embeddings database."
    )
    args = parser.parse_args()
    logger.info("Parsed arguments: %s", args)
    return args


def extract_repository_src_files(source: str) -> List[str]:
    """
    Identify source files in the repository/directory recursively.
    """
    logger.info("Identifying source files in directory: %s", source)

    source_files = []
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in BLACKLISTED_DIRECTORIES]

        for file in files:
            if any(file.endswith(ext) for ext in WHITELISTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                source_files.append(file_path)
                if VERBOSE:
                    logger.info(f"Found source file: {file_path}")
    return source_files


def extract_from_file(file_path: str, embeddings: bool = False) -> dict:
    """
    Extract module, class, and function information from a Python source file using extractors.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        tree = libcst.parse_module(source_code)

        module_name = os.path.basename(file_path).replace(".py", "")
        extracted_info = extract_module_info(tree, include_embeddings=embeddings)

        return {
            "name": module_name,
            "docstring": extracted_info["docstring"],
            "imports": extracted_info["imports"],
            "embedding": extracted_info["embedding"],
            "classes": extracted_info["classes"],
            "functions": extracted_info["functions"],
        }
    except Exception as e:
        logger.error("Error parsing file %s: %s", file_path, e)
        return {}


def construct_json(
    source_files: List[str],
    output_json: str = "repo_features.json",
    embeddings: bool = False,
) -> bool:
    """
    Construct a JSON file containing a list of extracted module features and embeddings.
    """
    logger.info("Starting JSON construction with module features and embeddings.")

    try:
        logger.info(f"Output JSON file: {output_json}")
        json_path = pathlib.Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.touch(exist_ok=True)

        extracted_data = []

        for source_file in tqdm(
            source_files, desc="Processing files", unit="file", leave=True
        ):
            extracted_info = extract_from_file(source_file, embeddings=embeddings)
            extracted_info["file_path"] = source_file
            if extracted_info:
                extracted_data.append(extracted_info)

        if extracted_data:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=4, cls=CompactJSONEncoder)
            logger.info("JSON file successfully created at %s.", output_json)
        return True
    except Exception as e:
        logger.error("Error constructing JSON file: %s", e)
        return False


def main():
    args = parse_arguments()
    if args.verbose:
        global VERBOSE
        VERBOSE = True
    repo = args.source
    output_json = args.output_json
    extract_embeddings = args.embeddings

    source_files = extract_repository_src_files(repo)
    if construct_json(
        source_files, output_json=output_json, embeddings=extract_embeddings
    ):
        logger.info("Json extraction completed successfully.")
    else:
        logger.error("Failed to construct the Knowledge Graph JSON file.")


main()
