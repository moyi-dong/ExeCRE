"""Isolated code execution (fork + pipe) for LCB-style run/capture; used by TLE/TLEfree paths."""

import json
import time
import signal
import faulthandler
import resource
import math
import sys
import multiprocessing
from typing import Any, Optional, Tuple

from ..utils.path_manager import get_src_dir

src_dir = get_src_dir()
# Parent of the `lcb_runner` package so `import lcb_runner.*` works.
lcb_repo_root = src_dir / "benchmark_repo" / "LiveCodeBench"
lcb_repo_root_str = str(lcb_repo_root)

if lcb_repo_root_str not in sys.path:
    sys.path.insert(0, lcb_repo_root_str)

from lcb_runner.evaluation.testing_util import (
    import_string,
    compile_code,
    get_function,
    call_method,
    Capturing,
    clean_if_name,
    make_function,
    TimeoutException,
    reliability_guard,
)


def _timeout_handler(signum, frame):
    raise TimeoutException


def _subprocess_worker(fn_name, test_case, code, timeout, result_pipe):
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, resource.RLIM_INFINITY))
            resource.setrlimit(resource.RLIMIT_AS, (1 * 1024 * 1024 * 1024, resource.RLIM_INFINITY))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, resource.RLIM_INFINITY))
        except Exception:
            pass

        sys.setrecursionlimit(1000)

        signal.signal(signal.SIGALRM, _timeout_handler)
        reliability_guard()

        if fn_name and fn_name is not None:
            faulthandler.enable()

            safe_import_string = import_string.replace("sys.setrecursionlimit(50000)", "sys.setrecursionlimit(1000)")
            code = safe_import_string + "\n\n" + code

            try:
                compiled_sol = compile_code(code, timeout)
            except SyntaxError as e:
                result_pipe.send((False, None, f"Compilation Error: {e}"))
                return

            sys.setrecursionlimit(1000)

            if compiled_sol is None:
                result_pipe.send((False, None, "Compilation failed"))
                return

            method = get_function(compiled_sol, fn_name)
            if method is None:
                result_pipe.send((False, None, f"Function {fn_name} not found"))
                return

            input_data = [json.loads(line) for line in test_case.split("\n") if line.strip()]

            signal.alarm(timeout)
            try:
                prediction = method(*input_data)
                signal.alarm(0)

                if isinstance(prediction, tuple):
                    prediction = list(prediction)

                result_pipe.send((True, prediction, None))
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    result_pipe.send((False, None, "Time Limit Exceeded"))
                elif "recursion" in str(e).lower():
                    result_pipe.send((False, None, "Runtime Error (Recursion)"))
                else:
                    result_pipe.send((False, None, "Runtime Error"))
            finally:
                signal.alarm(0)
                faulthandler.disable()
        else:
            code = clean_if_name(code)
            code = make_function(code)

            try:
                compiled = compile_code(code, timeout)
            except SyntaxError as e:
                result_pipe.send((False, None, f"Compilation Error: {e}"))
                return

            sys.setrecursionlimit(1000)

            if compiled is None:
                result_pipe.send((False, None, "Compilation failed"))
                return

            method = get_function(compiled, "wrapped_function")
            if method is None:
                result_pipe.send((False, None, "wrapped_function not found"))
                return

            signal.alarm(timeout)
            faulthandler.enable()

            with Capturing() as cap:
                try:
                    call_method(method, test_case)
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    if "timeoutexception" in repr(e).lower():
                        result_pipe.send((False, None, "Time Limit Exceeded"))
                    else:
                        result_pipe.send((False, None, repr(e)))
                    return
                finally:
                    signal.alarm(0)
                    faulthandler.disable()

            output = cap[0]
            result_pipe.send((True, output, None))

    except Exception as e:
        try:
            result_pipe.send((False, None, f"Subprocess error: {repr(e)}"))
        except Exception:
            pass


def run_code_capture(
    fn_name: Optional[str],
    test_case: str,
    code: str,
    timeout: int = 2,
    test_index: int = 0,
) -> Tuple[bool, Any, Optional[str]]:
    """Run one test without comparing to expected output; subprocess-isolated."""
    try:
        timeout = max(1, int(math.ceil(timeout)))
    except Exception:
        timeout = 2

    process = None
    parent_conn = None
    child_conn = None

    try:
        ctx = multiprocessing.get_context('fork')
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        process = ctx.Process(
            target=_subprocess_worker,
            args=(fn_name, test_case, code, timeout, child_conn)
        )

        process.start()

        child_conn.close()
        child_conn = None

        process.join(timeout=timeout + 3)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)

            return False, None, "Time Limit Exceeded"

        exit_code = process.exitcode

        if exit_code != 0:
            error_msg = "Runtime Error"
            if exit_code == -11:
                error_msg = "Runtime Error (Segmentation Fault)"
            elif exit_code == -6:
                error_msg = "Runtime Error (Aborted)"
            elif exit_code == -9:
                error_msg = "Runtime Error (Killed)"
            elif exit_code is not None and exit_code < 0:
                error_msg = f"Runtime Error (Signal {-exit_code})"

            return False, None, error_msg

        try:
            if parent_conn.poll(timeout=1):
                result = parent_conn.recv()
                return result
            else:
                return False, None, "No result from subprocess"
        except EOFError:
            return False, None, "Runtime Error (subprocess crashed)"
        except Exception as e:
            return False, None, f"Error getting result: {repr(e)}"

    except Exception as e:
        return False, None, f"Error: {repr(e)}"

    finally:
        try:
            if parent_conn is not None:
                parent_conn.close()
        except Exception:
            pass
        try:
            if child_conn is not None:
                child_conn.close()
        except Exception:
            pass
        try:
            if process is not None and process.is_alive():
                process.kill()
                process.join(timeout=1)
        except Exception:
            pass
