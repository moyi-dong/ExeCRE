"""
Code Extraction Tool

Extract code from model output, supporting multiple formats.
"""


def extract_code(model_output: str) -> str:
    """
    Extract code from model output
    
    Supports the following formats:
    - Code block format: ```python ... ```
    - Pure code format: Return directly (if no code block marker)
    
    Args:
        model_output: The original output string of the model
    
    Returns:
        str: The extracted code, if no code block is found, return an empty string
    """
    if not model_output:
        return ""
    
    outputlines = model_output.split("\n")
    
    # Find code block markers
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    
    if len(indexlines) < 2:
        # If no complete code block is found, try to find the PYTHON] marker (CodeLLaMa format)
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            # If still not found, return an empty string
            return ""
    
    # Extract all code blocks and merge (support multiple code blocks)
    # This ensures that all function definitions are extracted, not just the last code block
    code_blocks = []
    for i in range(0, len(indexlines) - 1, 2):
        if i + 1 < len(indexlines):
            start_idx = indexlines[i] + 1
            end_idx = indexlines[i + 1]
            block_content = "\n".join(outputlines[start_idx:end_idx]).strip()
            if block_content:
                code_blocks.append(block_content)
    
    # Merge all code blocks
    if code_blocks:
        return "\n\n".join(code_blocks)
    
    # If no code blocks are extracted, try to extract the last code block (backward compatibility)
    if len(indexlines) >= 2:
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]]).strip()
    
    return ""
