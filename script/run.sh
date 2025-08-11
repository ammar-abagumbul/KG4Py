#!/bin/bash

# Usage:
# ./run.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}" || exit

SOURCE="${PROJECT_ROOT}/demo_repo"
OUTPUT_JSON="${PROJECT_ROOT}/data/manim_features.json"
DB_DIR="${PROJECT_ROOT}/data/chromadb"
COLLECTION_NAME="manim_embeddings"  
CLEAR_DB="--clear-database"
VERBOSE="--verbose"
EMBEDDINGS="--embeddings"

echo "Running feature extraction $SOURCE"

# python src/repo_feature_extractor.py "$SOURCE" "$OUTPUT_JSON" $VERBOSE $CLEAR_DB $EMBEDDINGS
# python src/build_graph.py "$OUTPUT_JSON" $VERBOSE $CLEAR_DB
python src/build_chromadb.py "$OUTPUT_JSON" --db_dir "$DB_DIR" --collection_name "$COLLECTION_NAME"
