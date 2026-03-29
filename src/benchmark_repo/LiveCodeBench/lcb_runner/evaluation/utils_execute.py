"""
LiveCodeBench execution utilities.

Base imports and correctness checks for code execution.
"""

import ast
import json
import sys
import faulthandler
import platform
import importlib
import signal
import numpy as np
from io import StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from enum import Enum
from decimal import Decimal
import time

# Base import preamble with common Python standard-library symbols
BASE_IMPORTS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def check_correctness(code_to_execute, timeout=3):
    """
    Check whether executed code runs without raising.

    Args:
        code_to_execute: Source code to exec.
        timeout: Execution timeout in seconds (reserved for callers; not enforced here).

    Returns:
        True if exec completes with no exception, else False.
    """
    try:
        exec_globals = {}
        exec_locals = {}
        exec(code_to_execute, exec_globals, exec_locals)
        return True
    except Exception:
        return False
