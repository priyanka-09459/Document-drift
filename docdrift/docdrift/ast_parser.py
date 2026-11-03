"""
ast_parser.py
-------------
Extracts a structured index of code symbols (functions, classes, methods)
from a Python source tree using the built-in `ast` module.

Each symbol record captures enough information to (a) map it to relevant
documentation later, and (b) compare it across two commits to detect
behavior-relevant changes (signature, docstring, body).
"""

import ast
import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Symbol:
    qualified_name: str        # e.g. "mymodule.MyClass.my_method"
    kind: str                  # "function" | "method" | "class"
    file_path: str
    lineno: int
    end_lineno: Optional[int]
    signature: str             # e.g. "def foo(a, b=1, *args, **kwargs)"
    docstring: str              # "" if none
    body_hash: str              # hash of the normalized function body (ignoring docstring)
    decorators: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _format_arg_node(arg: ast.arg, default=None) -> str:
    name = arg.arg
    ann = ""
    if arg.annotation is not None:
        try:
            ann = f": {ast.unparse(arg.annotation)}"
        except Exception:
            ann = ": ..."
    default_str = ""
    if default is not None:
        try:
            default_str = f"={ast.unparse(default)}"
        except Exception:
            default_str = "=..."
    return f"{name}{ann}{default_str}"


def _format_args(args: ast.arguments) -> str:
    parts = []
    posonly = getattr(args, "posonlyargs", [])
    all_pos = posonly + args.args
    defaults = [None] * (len(all_pos) - len(args.defaults)) + list(args.defaults)

    # Positional-only parameters
    posonly_count = len(posonly)
    for arg, default in zip(all_pos[:posonly_count], defaults[:posonly_count]):
        parts.append(_format_arg_node(arg, default))
    if posonly_count > 0:
        parts.append("/")

    # Standard positional / keyword parameters
    for arg, default in zip(all_pos[posonly_count:], defaults[posonly_count:]):
        parts.append(_format_arg_node(arg, default))

    # *vararg or bare * if kwonlyargs present without vararg
    if args.vararg:
        vararg_ann = ""
        if getattr(args.vararg, "annotation", None) is not None:
            try:
                vararg_ann = f": {ast.unparse(args.vararg.annotation)}"
            except Exception:
                pass
        parts.append(f"*{args.vararg.arg}{vararg_ann}")
    elif getattr(args, "kwonlyargs", None):
        parts.append("*")

    # Keyword-only parameters
    kwonly = getattr(args, "kwonlyargs", [])
    kw_defaults = getattr(args, "kw_defaults", [None] * len(kwonly))
    for arg, default in zip(kwonly, kw_defaults):
        parts.append(_format_arg_node(arg, default))

    # **kwargs
    if args.kwarg:
        kwarg_ann = ""
        if getattr(args.kwarg, "annotation", None) is not None:
            try:
                kwarg_ann = f": {ast.unparse(args.kwarg.annotation)}"
            except Exception:
                pass
        parts.append(f"**{args.kwarg.arg}{kwarg_ann}")

    return ", ".join(parts)


def _body_hash(node) -> str:
    """
    Hash the function/class body excluding the docstring, so that
    docstring-only edits don't register as a "code change", and real
    logic changes always do.
    """
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(
        getattr(body[0], "value", None), (ast.Constant,)
    ) and isinstance(body[0].value.value, str):
        body = body[1:]  # drop docstring node
    try:
        src = "\n".join(ast.unparse(n) for n in body)
    except Exception:
        src = str(body)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


class SymbolVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: list[Symbol] = []
        self._scope_stack: list[str] = []

    def _qualify(self, name: str) -> str:
        if self._scope_stack:
            return ".".join(self._scope_stack) + "." + name
        return name

    def visit_ClassDef(self, node: ast.ClassDef):
        qname = self._qualify(node.name)
        docstring = ast.get_docstring(node) or ""
        self.symbols.append(
            Symbol(
                qualified_name=qname,
                kind="class",
                file_path=self.file_path,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                signature=f"class {node.name}",
                docstring=docstring,
                body_hash=_body_hash(node),
                decorators=[ast.unparse(d) for d in node.decorator_list],
            )
        )
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _visit_func(self, node):
        qname = self._qualify(node.name)
        kind = "method" if self._scope_stack else "function"
        docstring = ast.get_docstring(node) or ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        
        returns_str = ""
        if getattr(node, "returns", None) is not None:
            try:
                returns_str = f" -> {ast.unparse(node.returns)}"
            except Exception:
                returns_str = " -> ..."

        sig = f"{prefix} {node.name}({_format_args(node.args)}){returns_str}"
        self.symbols.append(
            Symbol(
                qualified_name=qname,
                kind=kind,
                file_path=self.file_path,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                signature=sig,
                docstring=docstring,
                body_hash=_body_hash(node),
                decorators=[ast.unparse(d) for d in node.decorator_list],
            )
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def visit_Assign(self, node: ast.Assign):
        # Index top-level module constants/configurations
        if not self._scope_stack:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    try:
                        val_src = ast.unparse(node.value)
                    except Exception:
                        val_src = "..."
                    sig = f"{name} = {val_src}"
                    self.symbols.append(
                        Symbol(
                            qualified_name=name,
                            kind="constant",
                            file_path=self.file_path,
                            lineno=node.lineno,
                            end_lineno=getattr(node, "end_lineno", None),
                            signature=sig,
                            docstring="",
                            body_hash=hashlib.sha256(val_src.encode("utf-8")).hexdigest()[:16],
                            decorators=[],
                        )
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Index top-level annotated constants/configurations
        if not self._scope_stack:
            if isinstance(node.target, ast.Name):
                name = node.target.id
                try:
                    ann_src = ast.unparse(node.annotation)
                except Exception:
                    ann_src = "..."
                val_src = ""
                if node.value is not None:
                    try:
                        val_src = f" = {ast.unparse(node.value)}"
                    except Exception:
                        val_src = " = ..."
                sig = f"{name}: {ann_src}{val_src}"
                self.symbols.append(
                    Symbol(
                        qualified_name=name,
                        kind="constant",
                        file_path=self.file_path,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", None),
                        signature=sig,
                        docstring="",
                        body_hash=hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16],
                        decorators=[],
                    )
                )
        self.generic_visit(node)


def parse_file(file_path: str) -> list[Symbol]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        print(f"[docdrift] SyntaxError parsing {file_path}: {e}")
        return []
    visitor = SymbolVisitor(file_path)
    visitor.visit(tree)
    return visitor.symbols


def parse_repo(repo_root: str, exclude_dirs=(".git", "venv", ".venv", "__pycache__", "node_modules")) -> list[Symbol]:
    """Walk a repo root and parse every .py file into Symbol records."""
    all_symbols: list[Symbol] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                all_symbols.extend(parse_file(fpath))
    return all_symbols


if __name__ == "__main__":
    import sys
    import json

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    syms = parse_repo(root)
    print(json.dumps([s.to_dict() for s in syms], indent=2))
