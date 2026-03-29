"""
LLM Direct Verify Prompt Template

Used to prompt the LLM to directly judge the correctness of the code.
"""

# System Prompt - Set the LLM as the code correctness judge
LLM_JUDGE_SYSTEM_PROMPT = """You are an expert code correctness judge. Your task is to determine whether a given code solution correctly solves a programming problem.

Guidelines:
1. Analyze the problem requirements carefully
2. Check if the code logic matches the problem requirements
3. Consider edge cases and constraints
4. Focus on correctness, not efficiency or style
5. Ignore minor issues like missing imports if the core logic is correct

You must respond with ONLY 'YES' or 'NO':
- 'YES' if the code correctly solves the problem
- 'NO' if the code has logical errors or doesn't solve the problem correctly"""


# User Prompt Template - Contains the problem description and the code to verify
LLM_JUDGE_USER_PROMPT_TEMPLATE = """## Problem Description
{problem_content}

## Code to Verify
```python
{code}
```

Does this code correctly solve the problem above?
Answer with ONLY 'YES' or 'NO'."""


def build_judge_prompt(problem_content: str, code: str) -> str:
    """Build the User Prompt for judging the correctness of the code

    Args:
        problem_content: The problem description
        code: The code to verify

    Returns:
        The formatted User Prompt
    """
    return LLM_JUDGE_USER_PROMPT_TEMPLATE.format(
        problem_content=problem_content,
        code=code
    )
