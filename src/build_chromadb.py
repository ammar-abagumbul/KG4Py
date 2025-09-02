import argparse
import json
import chromadb
import tqdm
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a chromadb database from a JSON file of module features and embeddings."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file containing module features and embeddings.",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="chromo_db",
        help="Directory to store the chromadb database.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="repo_embeddings",
        help="Name of the chromadb collection.",
    )
    args = parser.parse_args()
    logger.info("Parsed arguments: %s", args)
    return args


def main():
    args = parse_args()
    logger.info("Starting database build process.")
    with open(args.json_file, "r", encoding="utf-8") as f:
        modules = json.load(f)

    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.create_collection(name=args.collection_name)

    items_to_add = []
    # Extract features and embedddings from modules
    logger.info("Processing %d modules.", len(modules))
    for module in tqdm.tqdm(modules, desc="Processing modules"):
        # Classes
        try:
            for cls in module.get("classes", []):
                if "embedding" in cls and "name" in cls:
                    items_to_add.append(
                        {
                            "embeddings": cls["embedding"],
                            "name": cls["name"],
                            "document": cls["docstring"],
                            "type": "class",
                        }
                    )
                # add methods inside classes
                for method in cls.get("methods", []):
                    if "embedding" in method and "name" in method:
                        items_to_add.append(
                            {
                                "embeddings": method["embedding"],
                                "name": method["name"],
                                "document": method["code"],
                                "type": "method",
                            }
                        )
            # Functions
            for func in module.get("functions", []):
                if "embedding" in func and "name" in func:
                    items_to_add.append(
                        {
                            "embeddings": func["embedding"],
                            "name": func["name"],
                            "document": func["code"],
                            "type": "function",
                        }
                    )
        except Exception as e:
            logger.error("Error processing module %s: %s", module.get('name', 'unknown'), e)
            continue

    # Add to chromadb
    logger.info("Adding %d items to the database.", len(items_to_add))
    for idx, item in enumerate(
        tqdm.tqdm(items_to_add, desc="Adding items to chromadb")
    ):
        try:
            collection.add(
                embeddings=item["embeddings"],
                ids=[str(idx)],
                metadatas=[{"name": item["name"], "type": item["type"]}],
                documents=[item["document"]]
            )
        except Exception as e:
            logger.error(
                f'Error adding item {idx}, module name {item.get("name", "unknown")}: {e}'
            )

    logger.info("Chromadb database created successfully at %s", args.db_dir)


if __name__ == "__main__":
    main()
