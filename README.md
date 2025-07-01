# KG4Py: Python Knowledge Graph Generator

A Python toolkit for generating knowledge graphs from source code, enabling semantic search and enhanced code understanding capabilities.

## Overview

This project implements the methodology described in the research paper ["KG4Py: A toolkit for generating Python knowledge graph and code semantic search"](https://www.tandfonline.com/doi/epdf/10.1080/09540091.2022.2072471), providing a foundation for building intelligent code analysis systems.

### Knowledge Graphs in Software Development

Knowledge graphs represent structured information as interconnected entities and relationships, making them particularly valuable for code analysis and understanding. In the context of software development, knowledge graphs can capture complex relationships between modules, classes, functions, and their dependencies, enabling:

- **Enhanced Code Navigation**: Quickly understand how different components relate to each other
- **Semantic Search**: Find code based on conceptual similarity rather than just text matching
- **Dependency Analysis**: Visualize and analyze complex code dependencies
- **Code Recommendation**: Suggest relevant code snippets based on context and usage patterns

### Applications in RAG (Retrieval-Augmented Generation)

Knowledge graphs are particularly powerful when integrated with RAG applications for code-related tasks:

1. **Contextual Code Retrieval**: Instead of simple keyword matching, knowledge graphs enable retrieval based on semantic relationships between code components
2. **Enhanced Code Generation**: LLMs can leverage structured knowledge about existing codebases to generate more contextually appropriate code
3. **Intelligent Documentation**: Automatically generate documentation that reflects actual code relationships and usage patterns
4. **Code Quality Analysis**: Identify potential issues by analyzing patterns in the knowledge graph structure

## Methodology

The implementation follows a systematic approach to transform Python source code into structured knowledge graphs:

### 1. Concrete Syntax Tree Parsing
- **Tool**: LibCST (Concrete Syntax Tree library)
- **Purpose**: Parse Python source code while preserving formatting and comments
- **Output**: Structured representation of code that maintains original syntax

### 2. Information Extraction
The system extracts key structural elements from the parsed code:

- **Module Information**: Module names, Module level docstrings, imports, classes, functions, and hierarchical relationships
- **Import Statements**: Dependencies between modules, including external libraries
- **Function Definitions**: Signatures, parameters, return types, and docstrings
- **Class Definitions**: Class hierarchies, methods, attributes, and inheritance relationships

*Note: The complete schema is defined in `graph_schema.json`*

### 3. Knowledge Graph Construction
- **Database**: Neo4j graph database for efficient storage and querying
- **Nodes**: Represent code entities (modules, classes, functions)
- **Edges**: Capture relationships (imports, calls, inheritance, contains)
- **Properties**: Store metadata (docstrings, parameters, types)

### 4. Future Enhancements
The current implementation provides the foundation for more advanced features described in the original paper but does not yet include:
- Semantic similarity calculations
- Advanced query capabilities
- Code recommendation systems
- Integration with development tools

## Installation and Setup

### Prerequisites
- Python 3.8+
- Neo4j Database
- pip package manager

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Neo4j Database

**macOS (using Homebrew):**
```bash
brew install neo4j
```

**Windows/Linux:**
Follow the [official Neo4j installation guide](https://neo4j.com/docs/operations-manual/current/installation/)

### Step 3: Configure Neo4j
1. Start the Neo4j server:
   ```bash
   neo4j start
   ```

2. Access the Neo4j browser interface at `http://localhost:7474`

3. Set up authentication credentials and update `config.py` with your database connection details:
   ```python
   NEO4J_URI = "bolt://localhost:7687"
   NEO4J_USERNAME = "your_username"
   NEO4J_PASSWORD = "your_password"
   ```

## Usage

### Basic Usage
Generate a knowledge graph from a Python repository:

```bash
python build_graph.py --source path/to/your/repo/
```

### Advanced Options
```bash
python build_graph.py --source path/to/your/repo/ --verbose --clear-database
```

**Command Line Arguments:**
- `--source`: Path to the Python repository to analyze
- `--verbose`: Enable detailed logging output
- `--clear-database`: Clear existing database before creating new graph

### Example Output
The tool will process your Python codebase and create a knowledge graph containing:
- Module nodes with metadata
- Function and class nodes with signatures and documentation
- Relationship edges showing imports, calls, and containment


## Schema Overview

The knowledge graph follows a structured schema designed to capture the essential relationships in Python codebases. Key node types include:

- **Module**: Represents Python files and packages
- **Function**: Represents function definitions with parameters and metadata
- **Class**: Represents class definitions with methods and attributes
- **Import**: Represents import statements and dependencies

Relationships capture semantic connections such as:
- `CONTAINS`: Module contains classes/functions
- `IMPORTS`: Module imports other modules
- `INHERITS`: Class inheritance relationships
- `CALLS`: Function call relationships

## Current Status and Roadmap

### ✅ Completed Features
- [x] Python code parsing with LibCST
- [x] Basic entity extraction (modules, functions, classes, imports)
- [x] Neo4j graph database integration
- [x] Command-line interface for graph generation

### 🚧 In Progress
- [ ] Enhanced relationship detection
- [ ] Semantic similarity calculations
- [ ] Query optimization

### 📋 Future Enhancements
- [ ] RAG integration for code retrieval
- [ ] Integrate examples and documentations into the knowledge graph


## Citation

If you use this toolkit in your research, please cite the original paper:

*Note: This is an ongoing implementation of the KG4Py methodology. The current version provides core functionality with additional features planned for future releases.*
