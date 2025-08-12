"""
Configuration settings for the Knowledge Graph construction tool.
This file contains settings for connecting to Neo4j and defining content extraction parameters.
"""

# Neo4j connection settings (placeholders, to be customized by user)
import logging
from logging_config import setup_logging

logger = logging.getLogger(__name__)

try:
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "manim_neo4j"
    NEO4J_DATABASE = "neo4j"
    logger.info("Neo4j connection settings loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Neo4j connection settings: {e}")

# Content types supported for extraction
try:
    supported_content_types = ["docs", "examples", "tutorials"]
    logger.info(
        "Supported content types loaded successfully: %s", supported_content_types
    )
except Exception as e:
    logger.error(f"Error loading supported content types: {e}")

# Mapping of content types to file patterns or directories (placeholders)
# These are repository-agnostic placeholders; actual paths depend on the target repository structure
CONTENT_TYPE_PATTERNS = {
    "docs": ["*.md", "*.json", "docs/*"],  # Placeholder for documentation files
    "examples": ["examples/*", "samples/*"],  # Placeholder for example code
    "tutorials": ["tutorials/*", "guides/*"],  # Placeholder for tutorial content
}

# Graph schema definition
try:
    GRAPH_SCHEMA = "./graph_schema.json"
    logger.info("Graph schema path loaded successfully: %s", GRAPH_SCHEMA)
except Exception as e:
    logger.error(f"Error loading graph schema path: {e}")

# Logging configuration

LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Initialize logging
setup_logging(level=LOGGING_LEVEL, log_format=LOGGING_FORMAT)

WHITELISTED_EXTENSIONS = [".py"]
BLACKLISTED_DIRECTORIES = ["shaders", "templates", "testing", ".git", "__pycache__"]
