"""LiveCodeBench-style chat messages for codegen."""

from src.core.problem import Problem


class PromptConstants:
    """Shared LCB prompt fragments."""
    SYSTEM_MESSAGE_GENERIC = (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests."
    )
    
    FORMATTING_MESSAGE_WITH_STARTER_CODE = (
        "You will use the following starter code to write the solution to the problem "
        "and enclose your code within delimiters."
    )
    
    FORMATTING_WITHOUT_STARTER_CODE = (
        "Read the inputs from stdin solve the problem and write the answer to stdout "
        "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
        "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
    )


def format_livecodebench_prompt(problem: Problem) -> list:
    """OpenAI-style ``[system, user]`` messages for one LCB problem."""
    user_content = f"### Question:\n{problem.question_content}\n\n"
    
    if problem.starter_code:
        user_content += f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        user_content += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        user_content += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        user_content += "```python\n# YOUR CODE HERE\n```\n\n"
    
    user_content += "### Answer: (use the provided format with backticks)\n\n"

    chat_messages = [
        {
            "role": "system",
            "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    
    return chat_messages
