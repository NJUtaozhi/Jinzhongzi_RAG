"""Calculator tool implementation."""

from typing import Any
import ast
import math

from langchain.tools import tool


def _safe_eval(node: ast.AST, names: dict) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval(node.operand, names)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, names)
        right = _safe_eval(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname in names:
            args = [_safe_eval(arg, names) for arg in node.args]
            return names[fname](*args)
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def compute(expression: str) -> Any:
    """Safely evaluate simple math expressions."""
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": pow,
        "abs": abs,
    }
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree, allowed_names)
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression: {e}") from e


@tool
def calculator(expression: str) -> str:
    """Calculate expression and return result as string."""
    try:
        return str(compute(expression))
    except ValueError as e:
        return f"Error: {e}"


__all__ = ["calculator", "compute"]
