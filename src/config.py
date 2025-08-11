"""
Configuration settings for the Knowledge Graph construction tool.
This file contains settings for connecting to Neo4j and defining content extraction parameters.
"""

# Neo4j connection settings (placeholders, to be customized by user)
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "manim_neo4j"
NEO4J_DATABASE = "neo4j"

# Content types supported for extraction
supported_content_types = ["docs", "examples", "tutorials"]

# Mapping of content types to file patterns or directories (placeholders)
# These are repository-agnostic placeholders; actual paths depend on the target repository structure
CONTENT_TYPE_PATTERNS = {
    "docs": ["*.md", "*.json", "docs/*"],  # Placeholder for documentation files
    "examples": ["examples/*", "samples/*"],  # Placeholder for example code
    "tutorials": ["tutorials/*", "guides/*"]  # Placeholder for tutorial content
}

# Graph schema definition 
GRAPH_SCHEMA = "./graph_schema.json"

# Logging configuration
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

WHITELISTED_EXTENSIONS = [
  ".py"
]
BLACKLISTED_DIRECTORIES = [
  "shaders",
  "templates",
  "testing",
  ".git",
  "__pycache__"
]
