#!/bin/bash

# Usage:
# ./run_all.sh [--verbose] [--clear-database] [--output_json FILE] [--db_dir DIR] [--collection_name NAME] source

# Default values
OUTPUT_JSON="repo_features.json"
DB_DIR="chromadb"
COLLECTION_NAME="repo_embeddings"
CLEAR_DB=""
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --clear-database)
            CLEAR_DB="--clear-database"
            shift
            ;;
        --output_json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --db_dir)
            DB_DIR="$2"
            shift 2
            ;;
        --collection_name)
            COLLECTION_NAME="$2"
            shift 2
            ;;
        -* )
            echo "Unknown option: $1"
            exit 1
            ;;
        * )
            # First positional argument is the repo source
            if [ -z "$SOURCE" ]; then
                SOURCE="$1"
            else
                echo "Unexpected positional argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$SOURCE" ]; then
    echo "Error: source repository path or URL is required."
    echo "Usage: ./run_all.sh [--verbose] [--clear-database] [--output_json FILE] [--db_dir DIR] [--collection_name NAME] source"
    exit 1
fi

echo "Step 1: Extracting repo features..."
python3 src/repo_feature_extractor.py "$SOURCE" "$OUTPUT_JSON" $VERBOSE $CLEAR_DB
if [ $? -ne 0 ]; then
    echo "repo_feature_extractor.py failed."
    exit 1
fi

echo "Step 2: Building graph from features..."
python3 src/build_graph.py "$OUTPUT_JSON" $VERBOSE $CLEAR_DB
if [ $? -ne 0 ]; then
    echo "build_graph.py failed."
    exit 1
fi

echo "Step 3: Building chromadb database..."
python3 src/build_chromo_db.py "$OUTPUT_JSON" --db_dir "$DB_DIR" --collection_name "$COLLECTION_NAME"
if [ $? -ne 0 ]; then
    echo "build_chromo_db.py failed."
    exit 1
fi

echo "All steps completed successfully."
