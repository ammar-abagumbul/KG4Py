import libcst as cst
from typing import Dict, Any

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
                # Get the module name
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
    """Extractor for class definitions, including docstrings and methods."""
    def __init__(self):
        self.classes = []
        self.current_class = None

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        class_info = {
            'name': node.name.value,
            'docstring': node.get_docstring() or '',
            'methods': []
        }
        self.current_class = class_info
        self.classes.append(class_info)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.current_class = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self.current_class:
            method_info = {
                'name': node.name.value,
                'docstring': node.get_docstring() or '',
                'code': cst.Module(body=[node]).code 
            }
            self.current_class['methods'].append(method_info)

class FunctionExtractor(cst.CSTVisitor):
    """Extractor for top-level function definitions and their docstrings."""
    def __init__(self):
        self.functions = []
        self.depth = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Only extract top-level functions (not methods within classes)
        if self.depth == 0:
            func_info = {
                'name': node.name.value,
                'docstring': node.get_docstring() or '',
                'code': cst.Module(body=[node]).code 
            }
            self.functions.append(func_info)

def extract_module_info(tree: cst.Module) -> Dict[str, Any]:
    """Extract all relevant information from a module's CST using the defined extractors."""
    docstring_extractor = DocStringExtractor()
    import_extractor = ImportExtractor()
    class_extractor = ClassExtractor()
    function_extractor = FunctionExtractor()

    tree.visit(docstring_extractor)
    tree.visit(import_extractor)
    tree.visit(class_extractor)
    tree.visit(function_extractor)

    return {
        'docstring': docstring_extractor.docstring,
        'imports': import_extractor.imports,
        'classes': class_extractor.classes,
        'functions': function_extractor.functions
    }
