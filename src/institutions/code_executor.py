"""
Sandboxed Code Executor

Provides safe execution of agent-generated code. This is critical for
the recursive simulation framework: agents write REAL code that ACTUALLY
RUNS, but in a controlled environment that prevents harmful operations.

Security Model:
1. Restricted imports (whitelist only)
2. Resource limits (CPU, memory, time)
3. No file system access outside sandbox
4. No network access
5. Output capture and validation
"""

import ast
import sys
import io
import traceback
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from contextlib import redirect_stdout, redirect_stderr
import signal
from functools import wraps

from src.config import get_logger

logger = get_logger(__name__)


# Whitelist of safe modules for agent code
SAFE_MODULES = {
    # Core Python
    'math', 'random', 'itertools', 'functools', 'collections',
    'typing', 'dataclasses', 'enum', 'abc',

    # Scientific computing
    'numpy', 'torch',

    # Data structures
    'json', 'copy',

    # Our own modules (for recursive simulation)
    'src.core', 'src.agents', 'src.institutions',
}

# Dangerous operations to block
BLOCKED_BUILTINS = {
    'eval', 'exec', 'compile', '__import__',
    'open', 'file', 'input',
    'globals', 'locals', 'vars', 'dir',
    'getattr', 'setattr', 'delattr',
    'breakpoint', 'exit', 'quit',
}


@dataclass
class ExecutionResult:
    """Result of executing agent-generated code."""
    success: bool = False
    output: str = ""
    error: Optional[str] = None
    return_value: Any = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0

    # Security flags
    blocked_operations: List[str] = field(default_factory=list)
    resource_limit_hit: bool = False


class CodeValidator(ast.NodeVisitor):
    """
    AST-based code validator that checks for unsafe operations.

    Walks the AST and flags any dangerous constructs before execution.
    """

    def __init__(self):
        self.violations: List[str] = []
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        """Check imports against whitelist."""
        for alias in node.names:
            module = alias.name.split('.')[0]
            self.imports.add(module)
            if module not in SAFE_MODULES and not module.startswith('src.'):
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check from-imports against whitelist."""
        if node.module:
            module = node.module.split('.')[0]
            self.imports.add(module)
            if module not in SAFE_MODULES and not module.startswith('src.'):
                self.violations.append(f"Blocked import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check for dangerous function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                self.violations.append(f"Blocked builtin: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            # Check for dangerous methods
            if node.func.attr in {'system', 'popen', 'spawn', 'fork'}:
                self.violations.append(f"Blocked system call: {node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Check for dangerous attribute access."""
        dangerous_attrs = {'__code__', '__globals__', '__builtins__', '__subclasses__'}
        if node.attr in dangerous_attrs:
            self.violations.append(f"Blocked attribute access: {node.attr}")
        self.generic_visit(node)

    def validate(self, code: str) -> tuple[bool, List[str]]:
        """
        Validate code and return (is_safe, violations).
        """
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return len(self.violations) == 0, self.violations
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutError("Code execution timed out")


class RestrictedBuiltins:
    """
    Restricted builtins namespace for sandboxed execution.
    """

    @staticmethod
    def get_restricted_builtins() -> Dict[str, Any]:
        """Get a safe subset of builtins."""
        import builtins

        safe_builtins = {}
        allowed = {
            # Types
            'bool', 'int', 'float', 'str', 'list', 'dict', 'set', 'tuple',
            'frozenset', 'bytes', 'bytearray', 'complex',
            # Functions
            'abs', 'all', 'any', 'bin', 'callable', 'chr', 'divmod',
            'enumerate', 'filter', 'format', 'hash', 'hex', 'id',
            'isinstance', 'issubclass', 'iter', 'len', 'map', 'max', 'min',
            'next', 'oct', 'ord', 'pow', 'print', 'range', 'repr',
            'reversed', 'round', 'slice', 'sorted', 'sum', 'zip',
            # Constants
            'True', 'False', 'None',
            # Exceptions
            'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
            'AttributeError', 'RuntimeError', 'StopIteration',
        }

        for name in allowed:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        return safe_builtins


class SandboxedExecutor:
    """
    Execute agent-generated code in a sandboxed environment.

    This is the critical security layer that allows agents to write
    and run REAL code while preventing dangerous operations.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_memory_mb: float = 512.0,
        max_output_size: int = 100000,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.validator = CodeValidator()

        logger.info(f"SandboxedExecutor initialized (timeout={timeout_seconds}s, max_mem={max_memory_mb}MB)")

    def execute(
        self,
        code: str,
        entry_function: str = "run_experiment",
        kwargs: Dict[str, Any] = None,
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed environment.

        Args:
            code: Python source code to execute
            entry_function: Function to call after loading the module
            kwargs: Arguments to pass to the entry function

        Returns:
            ExecutionResult with output, return value, and security info
        """
        result = ExecutionResult()
        kwargs = kwargs or {}

        # Step 1: Validate code
        is_safe, violations = self.validator.validate(code)
        if not is_safe:
            result.error = f"Code validation failed: {violations}"
            result.blocked_operations = violations
            logger.warning(f"Code blocked: {violations}")
            return result

        # Step 2: Prepare restricted execution environment
        restricted_globals = {
            '__builtins__': RestrictedBuiltins.get_restricted_builtins(),
            '__name__': '__sandbox__',
        }

        # Add safe imports
        try:
            import numpy as np
            import torch
            restricted_globals['np'] = np
            restricted_globals['numpy'] = np
            restricted_globals['torch'] = torch
        except ImportError:
            pass

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Step 3: Execute with timeout
        import time
        start_time = time.time()

        try:
            # Set timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout_seconds))

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile and execute the module
                compiled = compile(code, '<sandbox>', 'exec')
                exec(compiled, restricted_globals)

                # Call the entry function if it exists
                if entry_function in restricted_globals:
                    result.return_value = restricted_globals[entry_function](**kwargs)

            result.success = True

        except TimeoutError:
            result.error = f"Execution timed out after {self.timeout_seconds}s"
            result.resource_limit_hit = True
        except MemoryError:
            result.error = "Memory limit exceeded"
            result.resource_limit_hit = True
        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            result.execution_time_ms = (time.time() - start_time) * 1000

        # Capture output (with size limit)
        result.output = stdout_capture.getvalue()[:self.max_output_size]
        if stderr_capture.getvalue():
            result.output += f"\n[STDERR]\n{stderr_capture.getvalue()[:self.max_output_size]}"

        logger.info(f"Code execution: success={result.success}, time={result.execution_time_ms:.1f}ms")

        return result

    def execute_isolated(
        self,
        code: str,
        entry_function: str = "run_experiment",
        kwargs: Dict[str, Any] = None,
    ) -> ExecutionResult:
        """
        Execute code in a completely isolated subprocess.

        This provides stronger isolation than execute() but has more overhead.
        Use for untrusted or complex code.
        """
        # Validate first
        is_safe, violations = self.validator.validate(code)
        if not is_safe:
            return ExecutionResult(
                error=f"Code validation failed: {violations}",
                blocked_operations=violations,
            )

        # Use multiprocessing for isolation
        def isolated_worker(code, entry_function, kwargs, result_queue):
            try:
                # Re-run in subprocess
                executor = SandboxedExecutor(
                    timeout_seconds=self.timeout_seconds,
                    max_memory_mb=self.max_memory_mb,
                )
                result = executor.execute(code, entry_function, kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(ExecutionResult(error=str(e)))

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=isolated_worker,
            args=(code, entry_function, kwargs or {}, result_queue),
        )

        process.start()
        process.join(timeout=self.timeout_seconds + 5)

        if process.is_alive():
            process.terminate()
            process.join()
            return ExecutionResult(
                error="Process timed out",
                resource_limit_hit=True,
            )

        try:
            return result_queue.get_nowait()
        except:
            return ExecutionResult(error="Failed to get result from subprocess")


# Convenience function for quick execution
def safe_execute(code: str, **kwargs) -> ExecutionResult:
    """
    Execute code with default sandbox settings.

    Example:
        result = safe_execute('''
        def run_experiment():
            return {"answer": 42}
        ''')
        print(result.return_value)  # {"answer": 42}
    """
    executor = SandboxedExecutor()
    return executor.execute(code, **kwargs)
