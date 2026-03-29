"""
LLM Analyze Verify Prompt Template

TextGrad-style two-stage verification Prompt Template
Used to prompt the LLM to analyze the correctness and time efficiency of the code.
Used to prompt the LLM to judge the correctness of the code based on the analysis result.
"""


# System Prompt for the analysis stage
ANALYSIS_SYSTEM_PROMPT = """You are an intelligent assistant used as an evaluator.
You will analyze a code implementation for a coding problem.
The code will be tested with harder tests, so think about edge cases.

Think about:
1. Correctness of the code logic - does it correctly solve the problem?
2. Time complexity and performance - will it pass harder test cases within time limits?
3. Edge cases handling - does it handle boundary conditions correctly?

Give very concise but thorough feedback. Focus on potential issues."""


# User Prompt Template for the analysis stage
ANALYSIS_USER_TEMPLATE = """**The coding problem:**
{problem_content}

**Code to analyze:**
```python
{code}
```

Analyze this code for correctness and runtime performance. Be concise but thorough."""


def build_analysis_prompt(problem_content: str, code: str) -> str:
    """Build the User Prompt for the analysis stage

    Args:
        problem_content: The problem description
        code: The code to analyze

    Returns:
        The formatted User Prompt
    """
    return ANALYSIS_USER_TEMPLATE.format(
        problem_content=problem_content,
        code=code
    )



# System Prompt for the judgment stage
JUDGE_SYSTEM_PROMPT = """Based on the code analysis provided, determine if the code is correct.
Consider both correctness and efficiency issues mentioned in the analysis.

You must respond with ONLY 'YES' or 'NO':
- 'YES' if the code correctly solves the problem (minor efficiency issues are acceptable)
- 'NO' if the code has logical errors, incorrect output, or severe efficiency issues that would cause TLE"""


# User Prompt Template for the judgment stage
JUDGE_USER_TEMPLATE = """## Problem
{problem_content}

## Code
```python
{code}
```

## Analysis
{analysis}

Based on the analysis above, is this code correct? Answer ONLY 'YES' or 'NO'."""


def build_judge_prompt(problem_content: str, code: str, analysis: str) -> str:
    """Build the User Prompt for the judgment stage

    Args:
        problem_content: The problem description
        code: The code to verify
        analysis: The analysis result of the first stage

    Returns:
        The formatted User Prompt
    """
    return JUDGE_USER_TEMPLATE.format(
        problem_content=problem_content,
        code=code,
        analysis=analysis
    )
