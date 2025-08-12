#!/usr/bin/env python3
"""
Command-line tool to construct a Knowledge Graph from a repository for use with a RAG system.
This tool extracts content (documentation, examples, tutorials) and populates a Neo4j database.
"""

import argparse
import sys
from typing import List, Optional
import logging
import os
import json
import time
from pathlib import Path

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    WHITELISTED_EXTENSIONS,
    BLACKLISTED_DIRECTORIES,
)

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, DriverError

# Setup logging
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
        "json-file",
        type=str,
        required=True,
        help="Path or URL to the json file containing repo structure.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for debugging."
    )
    parser.add_argument(
        "--clear-database",
        action="store_true",
        help="Clear all data in the Neo4j database before processing.",
    )
    args = parser.parse_args()
    logger.info("Parsed arguments: %s", args)
    return args


def connect_to_neo4j(
    max_retries: int = 3, retry_delay: float = 2.0
) -> Optional[Driver]:
    """
    Establish connection to Neo4j database with retry mechanism.
    """
    uri = NEO4J_URI
    user = NEO4J_USER
    password = NEO4J_PASSWORD

    logger.info("Attempting to connect to Neo4j at %s with user %s", uri, user)

    for attempt in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Successfully connected to Neo4j.")
            return driver
        except (DriverError, Neo4jError) as e:
            if VERBOSE:
                logger.debug(
                    f"Neo4j connection error details (attempt {attempt + 1}/{max_retries}): {e}"
                )
            if attempt < max_retries - 1:
                logger.info("Retrying connection in %s seconds (attempt %d/%d)...", retry_delay, attempt + 1, max_retries)
                time.sleep(retry_delay)
    logger.error("Failed to connect to Neo4j after all retries.")
    return None


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


def escape_cypher_string(value: str) -> str:
    """Properly escape a string for use in Cypher queries."""
    if value is None:
        return "null"

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")

    return f'"{escaped}"'


def build_cypher_queries(module_info: dict) -> List[str]:
    """Build Cypher queries from the extracted module information.

    Returns list of properly escaped string queries ready for execution.
    """
    queries = []

    # Create Module node
    module_name = escape_cypher_string(module_info["name"])
    module_doc = escape_cypher_string(module_info["docstring"])
    queries.append(
        f"MERGE (m:Module {{name: {module_name}}}) " f"SET m.docstring = {module_doc}"
    )

    # Create Import relationships
    for imp in module_info["imports"]:
        if imp:
            import_name = escape_cypher_string(imp)
            module_name_escaped = escape_cypher_string(module_info["name"])
            queries.append(
                f"MERGE (i:Module {{name: {import_name}}}) "
                f"MERGE (m:Module {{name: {module_name_escaped}}}) "
                f"MERGE (m)-[:IMPORTS]->(i)"
            )

    # Create Class nodes and relationships
    for cls in module_info["classes"]:
        cls_name = escape_cypher_string(cls["name"])
        cls_doc = escape_cypher_string(cls["docstring"])
        module_name_escaped = escape_cypher_string(module_info["name"])
        queries.append(
            f"MERGE (c:Class {{name: {cls_name}}}) "
            f"SET c.docstring = {cls_doc} "
            f"MERGE (m:Module {{name: {module_name_escaped}}}) "
            f"MERGE (m)-[:CONTAINS]->(c)"
        )

        # Create Method nodes and relationships
        for method in cls["methods"]:
            method_name = escape_cypher_string(method["name"])
            method_doc = escape_cypher_string(method["docstring"])
            cls_name_escaped = escape_cypher_string(cls["name"])
            queries.append(
                f"MERGE (meth:Function {{name: {method_name}}}) "
                f"SET meth.docstring = {method_doc} "
                f"MERGE (c:Class {{name: {cls_name_escaped}}}) "
                f"MERGE (c)-[:HAS_METHOD]->(meth)"
            )

    # Create Function nodes and relationships
    for func in module_info["functions"]:
        func_name = escape_cypher_string(func["name"])
        func_doc = escape_cypher_string(func["docstring"])
        module_name_escaped = escape_cypher_string(module_info["name"])
        queries.append(
            f"MERGE (f:Function {{name: {func_name}}}) "
            f"SET f.docstring = {func_doc} "
            f"MERGE (m:Module {{name: {module_name_escaped}}}) "
            f"MERGE (m)-[:CONTAINS]->(f)"
        )

    return queries


def save_to_json_fallback(module_info: dict, file_path: str) -> None:
    """Save extracted module information to a JSON file as a fallback."""
    fallback_dir = Path("failed_operations")
    fallback_dir.mkdir(exist_ok=True)
    module_name = module_info["name"]
    json_path = fallback_dir / f"{module_name}_failed.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(module_info, f, indent=2)
        logger.info(f"Saved module data to fallback file: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save fallback JSON for {module_name}: {e}")


def execute_cypher(
    driver: Driver, query: str, max_retries: int = 3, retry_delay: float = 2.0
) -> str:
    """Execute a single Cypher query with retry mechanism. Returns True if successful, False otherwise."""
    error_message = ""
    logger.debug(f"Executing Cypher query: {query}")
    for attempt in range(max_retries):
        try:
            with driver.session(
                database=NEO4J_DATABASE if NEO4J_DATABASE else None
            ) as session:
                session.run(query)
                return ""
        except (Neo4jError, DriverError) as e:
            if VERBOSE:
                logger.debug(
                    f"Cypher query execution error details (attempt {attempt + 1}/{max_retries}): {e}"
                )
            if attempt < max_retries - 1:
                logger.info(f"Retrying query execution in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to execute Cypher query after {max_retries} attempts: {query}"
                )
                logger.error(f"Error details: {e}")
            error_message = str(e)
    logger.error("Failed to execute Cypher query after all retries.")
    return error_message


def populate_graph(driver: object, json_path: str) -> None:
    """
    Populate the Neo4j database with extracted content.
    This function orchestrates the parsing of Python source files and population of the Neo4j graph.
    Implements graceful degradation by continuing processing even if database operations fail.
    """
    logger.info("Starting to populate Neo4j graph with extracted content.")
    failed_files = []
    repo_info = json.load(json_path)

    for module_info in repo_info:
        if module_info:
            file_path = module_info.get("file_path", "unknown_file")
            queries = build_cypher_queries(module_info)
            success = True
            for query in queries:
                error_message = execute_cypher(driver, query)
                if error_message:
                    success = False
            if not success:
                logger.error(f"Failed to fully process {file_path} into database.")
                module_info["error"] = error_message
                save_to_json_fallback(module_info, file_path)
                failed_files.append(file_path)

    if failed_files:
        logger.warning(
            f"Completed with failures for {len(failed_files)} files. Data saved to fallback JSON files."
        )
        logger.warning("Failed files: %s", ", ".join(failed_files))
    else:
        logger.info("Successfully completed populating Neo4j graph with no failures.")


def main() -> int:
    """
    Main function to orchestrate Knowledge Graph construction.
    """
    global VERBOSE

    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
        VERBOSE = True

    # Check if config.py exists
    config_path = "./config.py"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found. Please create it.")
        return 1

    logger.info("Starting Knowledge Graph construction process.")

    # Connect to Neo4j
    driver = connect_to_neo4j()
    if driver is None:
        logger.error("Failed to establish Neo4j connection. Exiting.")
        return 1

    # Clear database if requested
    if args.clear_database:
        logger.warning("Clearing all data in Neo4j database as requested.")
        clear_query = "MATCH (n) DETACH DELETE n"
        error_message = execute_cypher(driver, clear_query)
        if error_message:
            logger.error("Failed to clear database. Proceeding anyway.")
        else:
            logger.info("Database cleared successfully.")

    # json path for module features
    json_file = args.json_path
    if not json_file:
        logger.error("JSON path must be provided for module features.")
        return 1

    # Populate the graph
    try:
        populate_graph(driver, json_file)
        logger.info("Knowledge Graph construction completed.")
    except Exception as e:
        logger.error(f"Unexpected error during graph population: {e}")
        return 1
    finally:
        if driver:
            driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
