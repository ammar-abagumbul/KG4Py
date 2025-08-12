import torch
import libcst as cst
import numpy as np
import logging
from typing import Dict, Any, List, Callable

# Initialize logger
logger = logging.getLogger(__name__)


def preprocess_code(code: str) -> str:
    """Preprocess code by removing line breaks, indentations, and normalizing whitespace."""
    code = code.replace("\n", " ").replace("\r", " ")
    code = code.replace("\t", " ").replace("    ", " ")
    while "  " in code:
        code = code.replace("  ", " ")
    result = code.strip()
    return result


class DocStringExtractor(cst.CSTVisitor):
    """Extractor for module-level docstrings."""

    def __init__(self):
        self.docstring = ""
        self.docstring_found = False

    def visit_SimpleString(self, node: cst.SimpleString) -> None:
        if not self.docstring_found:
            self.docstring = node.evaluated_value or ""
            self.docstring_found = True


class ImportExtractor(cst.CSTVisitor):
    """Extractor for import statements and imported module names."""

    def __init__(self):
        self.imports = []

    def visit_Import(self, node: cst.Import) -> None:
        """Handle 'import module' statements."""
        for name in node.names:
            if isinstance(name, cst.ImportAlias):
                module_name = self._get_full_name(name.name)
                self.imports.append(module_name)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """Handle 'from module import ...' statements."""
        if node.module:
            # Get the base module name
            base_module = self._get_full_name(node.module)

            # Handle the imported names
            if isinstance(node.names, cst.ImportStar):
                # from module import *
                self.imports.append(f"{base_module}.*")
            else:
                # from module import name1, name2, ...
                for name in node.names:
                    if isinstance(name, cst.ImportAlias):
                        imported_name = self._get_full_name(name.name)
                        full_import = f"{base_module}.{imported_name}"
                        self.imports.append(full_import)

    def _get_full_name(self, node) -> str:
        """Extract the full dotted name from a Name or Attribute node."""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            # Handle dotted imports like 'os.path'
            return f"{self._get_full_name(node.value)}.{node.attr.value}"
        else:
            return str(node)


class ClassExtractor(cst.CSTVisitor):
    """Extractor for class definitions, including docstrings and methods.
    This applies to outer classes, not nested classes."""

    def __init__(self, embeddings: bool = False, model=None, device=None):
        self.classes = []
        self.depth = 0
        self.current_class = None
        self.embeddings = embeddings
        self.model = model
        self.device = device

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if self.depth > 0:
            return

        self.depth += 1
        try:
            raw_code = preprocess_code(cst.Module(body=[node]).code)
            class_def = raw_code.split(":")[0] + ":"
            docstring = preprocess_code(node.get_docstring() or "")
            embedding = None
            if self.embeddings and self.model and self.device:
                tokens_id = self.model.tokenize(
                    [f"{class_def} {docstring}"],
                    max_length=512,
                    mode="<encoder-only>",
                )
                source_id = torch.tensor(tokens_id, device=self.device)
                _, class_embedding = self.model(source_id)
                embedding = class_embedding.tolist()
            class_info = {
                "name": node.name.value,
                "docstring": node.get_docstring() or "",
                "embedding": embedding,
                "methods": [],
            }
            self.current_class = class_info
            self.classes.append(class_info)
        except Exception as e:
            logger.error("Error processing class definition %s: %s", node.name.value, e)
            self.current_class = {
                "name": node.name.value,
                "docstring": node.get_docstring() or "",
                "embedding": None,
                "methods": [],
            }
            self.classes.append(self.current_class)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self.depth == 1:
            self.current_class = None
        self.depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self.current_class is None:
            # If we're not inside a class, skip method definitions
            return
        try:
            raw_code = preprocess_code(cst.Module(body=[node]).code)
            func_def = raw_code.split(":")[0] + ":"
            docstring = preprocess_code(node.get_docstring() or "")
            embedding = None
            if self.embeddings and self.model and self.device:
                tokens_id = self.model.tokenize(
                    [f"{func_def} {docstring}"],
                    max_length=512,
                    mode="<encoder-only>",
                )
                source_id = torch.tensor(tokens_id, device=self.device)
                _, method_embedding = self.model(source_id)
                embedding = method_embedding.tolist()
            if self.current_class:
                method_info = {
                    "name": node.name.value,
                    "docstring": node.get_docstring() or "",
                    "code": cst.Module(body=[node]).code,
                    "embedding": embedding,
                }
                self.current_class["methods"].append(method_info)
        except Exception as e:
            logger.error(
                "Error processing method definition %s: %s", node.name.value, e
            )
            if self.current_class:
                method_info = {
                    "name": node.name.value,
                    "docstring": node.get_docstring() or "",
                    "code": cst.Module(body=[node]).code,
                    "embedding": None,
                }
                self.current_class["methods"].append(method_info)


class FunctionExtractor(cst.CSTVisitor):
    """Extractor for top-level function definitions and their docstrings."""

    def __init__(self, embeddings: bool = False, model=None, device=None):
        self.functions = []
        self.embeddings = embeddings
        self.model = model
        self.device = device
        self.depth = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Only extract top-level functions (not methods within classes)
        if self.depth != 0:
            return

        try:
            raw_code = preprocess_code(cst.Module(body=[node]).code)
            func_def = raw_code.split(":")[0] + ":"
            docstring = preprocess_code(node.get_docstring() or "")
            embedding = None
            if self.embeddings and self.model and self.device:
                tokens_id = self.model.tokenize(
                    [f"{func_def} {docstring}"],
                    max_length=512,
                    mode="<encoder-only>",
                )
                source_id = torch.tensor(tokens_id, device=self.device)
                _, func_embedding = self.model(source_id)
                embedding = func_embedding.tolist()
            func_info = {
                "name": node.name.value,
                "docstring": node.get_docstring() or "",
                "code": cst.Module(body=[node]).code,
                "embedding": embedding,
            }
            self.functions.append(func_info)

        except Exception as e:
            logger.error(
                "Error processing function definition %s: %s", node.name.value, e
            )
            func_info = {
                "name": node.name.value,
                "docstring": node.get_docstring() or "",
                "code": cst.Module(body=[node]).code,
                "embedding": None,
            }
            self.functions.append(func_info)


def chunk_module_content(content: str, max_tokens: int = 1024) -> List[str]:
    """Divide module content into chunks that fit within the token limit."""

    # NOTE: explore other methods to fit more context into UniXcoder's 512 token limit
    words = content.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + 1 > max_tokens:  # +1 for space
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += word_length + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def aggregate_embeddings(
    embeddings: List[List[float]], strategy: str = "mean"
) -> List[float]:
    """Aggregate multiple embeddings into a single embedding using the specified strategy."""

    if strategy == "mean":
        return np.mean(embeddings, axis=0).tolist()
    elif strategy == "max":
        return np.max(embeddings, axis=0).tolist()
    elif strategy == "weighted":
        # TODO: i might consider weighting different module components differently
        # weights = np.ones(len(embeddings)) / len(embeddings)
        # return np.average(embeddings, axis=0, weights=weights).tolist()
        logger.warning("Weighted aggregation is not implemented yet.")
        raise NotImplementedError("Weighted aggregation is not implemented yet.")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_module_embeddings(module_info: Dict[str, Any], strategy: str = "mean"):
    """Aggregate embeddings from module information into a single embedding."""

    all_embeddings = []

    # Collect embeddings from classes
    for cls in module_info.get("classes", []):
        if cls.get("embedding") is not None:
            all_embeddings.append(cls["embedding"])
        for method in cls.get("methods", []):
            if method.get("embedding") is not None:
                all_embeddings.append(method["embedding"])

    # Collect embeddings from functions
    for func in module_info.get("functions", []):
        if func.get("embedding") is not None:
            all_embeddings.append(func["embedding"])

    # Aggregate embeddings
    if all_embeddings:
        return aggregate_embeddings(all_embeddings, strategy)
    else:
        return None  # No embeddings to aggregate


def extract_module_info(
    tree: cst.Module, include_embeddings: bool = False
) -> Dict[str, Any]:
    """Extract all relevant information from a module's CST using the defined extractors."""

    model = None
    device = None
    if include_embeddings:
        import torch
        from unixcoder import UniXcoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("microsoft/unixcoder-base")
        model.to(device)

    docstring_extractor = DocStringExtractor()
    import_extractor = ImportExtractor()
    class_extractor = ClassExtractor(
        embeddings=include_embeddings, model=model, device=device
    )
    function_extractor = FunctionExtractor(
        embeddings=include_embeddings, model=model, device=device
    )

    tree.visit(docstring_extractor)
    tree.visit(import_extractor)
    tree.visit(class_extractor)
    tree.visit(function_extractor)

    module_info = {
        "docstring": docstring_extractor.docstring,
        "imports": import_extractor.imports,
        "classes": class_extractor.classes,
        "functions": function_extractor.functions,
        "embedding": None,
    }
    if include_embeddings:
        module_info["embedding"] = aggregate_module_embeddings(
            module_info, strategy="mean"
        )

    return module_info
