"""User template for bruteforce/simulation codegen."""

from src.core.problem import Problem


SYSTEM_PROMPT_FOR_SIMULATION_CODE = """You are an expert Python programmer specializing in brute force, simulation, and enumeration algorithms.  
In competitive programming, brute force, simulation, and enumeration are three fundamental strategies often used to ensure correctness in problem solving, even if the solution is not efficient.

- Brute force means trying all possible solutions, often using nested loops or recursion.  
- Simulation means reproducing the process described in the problem statement step by step.  
- Enumeration means listing all valid candidates and checking whether they satisfy the conditions.  

These methods are simple, intuitive, and useful for small input sizes or for verifying more optimized solutions.

Now, based on these principles, write a correct but not necessarily efficient solution to the following problem.  
Your goal is to ensure correctness using any of the above approaches, even if the time complexity is high.

Focus on implementing a solution that is correct and easy to understand, even if it's not the most efficient approach."""


def get_simulation_question_template_answer(problem: Problem) -> str:
    """Same structure as generic LCB template, for simulation system prompt."""
    prompt = f"### Question:\n{problem.question_content}\n\n"
    
    if problem.starter_code:
        prompt += f"### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    
    prompt += f"### Answer: (use the provided format with backticks, only contain the code)\n\n"
    return prompt

