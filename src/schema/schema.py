# This module parses schema JSON and generates randomized input data accordingly.
import json
import random
import re
from typing import Any, Dict, List, Tuple, Union, Optional
from collections import defaultdict, deque
from .config import default_config


def biased_random_int(min_val: int, max_val: int) -> int:
    """Generate random int with optional boundary bias."""
    if min_val == max_val:
        return min_val
    
    if default_config.boundary_bias:
        range_size = max_val - min_val
        boundary_size = max(1, range_size // 10)
    
        if random.random() < 0.2:
            end_boundary = min(min_val + boundary_size, max_val)
            return random.randint(min_val, end_boundary)
        else:
            start_boundary = max(max_val - boundary_size, min_val)
            return random.randint(start_boundary, max_val)
    else:
        return random.randint(min_val, max_val)

def biased_random_float(min_val: float, max_val: float) -> float:
    """Generate random float with optional boundary bias."""
    if min_val == max_val:
        return min_val
    
    if default_config.boundary_bias:
        range_size = max_val - min_val
        boundary_size = range_size * 0.1
        
        if random.random() < 0.2:
            end_boundary = min(min_val + boundary_size, max_val)
            return random.uniform(min_val, end_boundary)
        else:
            start_boundary = max(max_val - boundary_size, min_val)
            return random.uniform(start_boundary, max_val)
    else:
        return random.uniform(min_val, max_val)

def parse_schema(schema_json: str) -> Dict[str, Any]:
    """Parse JSON schema text into a dict."""
    try:
        schema_str = schema_json.strip()
        if schema_str.startswith("```json"):
            schema_str = schema_str[7:]
        elif schema_str.startswith("```"):
            schema_str = schema_str[3:]
        
        if schema_str.endswith("```"):
            schema_str = schema_str[:-3]
        
        schema_str = schema_str.strip()
        
        schema = json.loads(schema_str)
        
        required_fields = ["input_format", "parameters", "output_order"]
        for field in required_fields:
            if field not in schema:
                raise ValueError(f"Missing required field: {field}")
        
        if schema["input_format"] not in ["stdin", "function_args"]:
            raise ValueError("input_format must be 'stdin' or 'function_args'")
        if not isinstance(schema["parameters"], dict):
            raise ValueError("parameters must be a dict")
        if not isinstance(schema["output_order"], list):
            raise ValueError("output_order must be a list")
        
        return schema
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}")
    except Exception as e:
        raise ValueError(f"Schema parse error: {e}")

def extract_dependencies(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract top-level parameter dependencies."""
    dependencies = defaultdict(list)
    
    def extract_from_value(value):
        """Extract variable refs recursively."""
        if isinstance(value, str):
            pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?'
            matches = re.findall(pattern, value)
            return matches
        elif isinstance(value, list):
            refs = []
            for item in value:
                refs.extend(extract_from_value(item))
            return refs
        elif isinstance(value, dict):
            refs = []
            for v in value.values():
                refs.extend(extract_from_value(v))
            return refs
        return []
    
    for param_name, param_spec in schema['parameters'].items():
        dependent_refs = extract_from_value(param_spec)
        
        for ref in dependent_refs:
            if ref in schema['parameters']:
                dependencies[ref].append(param_name)
        
        if param_name not in dependencies:
            dependencies[param_name] = []
    
    for key in dependencies:
        dependencies[key] = list(set(dependencies[key]))
    
    return dict(dependencies)

def topological_sort(dependency_graph: Dict[str, List[str]]) -> List[str]:
    """Return topological order; raise on cycles."""
    in_degree = {node: 0 for node in dependency_graph}
    for node, neighbors in dependency_graph.items():
        for neighbor in neighbors:
            if neighbor in in_degree:
                in_degree[neighbor] += 1
    
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    result = []
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        for neighbor in dependency_graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(dependency_graph):
        raise ValueError("Cycle detected in dependency graph")
    
    return result

def evaluate_expression(expr: str, context: Dict[str, Any]) -> int:
    """Evaluate simple expressions with variable refs."""
    if not isinstance(expr, str):
        return expr
    
    pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?'
    
    def replace_variable(match):
        full_match = match.group(0)
        var_name = match.group(1)
        
        if '.' in full_match:
            property_name = full_match.split('.')[1]
            value = context.get(var_name)
            if value is None:
                raise ValueError(f"Variable not found in context: '{var_name}'")
            
            if property_name == 'length':
                if isinstance(value, (list, tuple, str)):
                    return str(len(value))
                else:
                    raise ValueError(f"Variable has no length property: '{var_name}'")
            else:
                raise ValueError(f"Unsupported property access: '{property_name}'")
        else:
            value = context.get(var_name)
            if value is None:
                raise ValueError(f"Variable not found in context: '{var_name}'")
            return str(value)
    
    expr_with_values = re.sub(pattern, replace_variable, expr)
    
    try:
        allowed_chars = set('0123456789+-*/()[]{}.,^eE minaxMINAX')
        if not all(c in allowed_chars for c in expr_with_values):
            raise ValueError("Expression contains disallowed characters")
        
        import re as _re
        word_pattern = r'[a-zA-Z]+'
        words = _re.findall(word_pattern, expr_with_values)
        allowed_functions = {'min', 'max', 'e', 'E'}
        for word in words:
            if word.lower() not in allowed_functions and word not in allowed_functions:
                raise ValueError(f"Expression contains disallowed token: '{word}'")
        
        expr_with_values = expr_with_values.replace('^', '**')
        
        safe_globals = {"__builtins__": {}, "min": min, "max": max}
        result = eval(expr_with_values, safe_globals)
        return int(result) if isinstance(result, (int, float)) else 0
    except Exception as e:
        raise ValueError(f"Expression evaluation failed: '{expr}' -> '{expr_with_values}', error: {str(e)}")


def generate_variable(param_spec: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """Generate Variable values."""
    data_type = param_spec.get("data_type")
    range_spec = param_spec.get("range")
    
    data_type_lower = str(data_type).lower() if data_type else ""
    
    if data_type_lower == "int":
        if isinstance(range_spec, list) and len(range_spec) == 2:
            min_val, max_val = range_spec
            min_val = evaluate_expression(min_val, context)
            max_val = evaluate_expression(max_val, context)
            
            config_min, config_max = default_config.get_variable_limit("int")
            min_val = max(min_val, config_min)
            max_val = min(max_val, config_max)
            
            return biased_random_int(min_val, max_val)
        else:
            print(f"Warning: invalid int range_spec {range_spec}, using default range")
            config_limit = default_config.get_variable_limit("int")
            return biased_random_int(config_limit[0], config_limit[1])
    
    elif data_type_lower == "float":
        config_min, config_max = default_config.get_variable_limit("float")
        if isinstance(range_spec, list) and len(range_spec) == 2:
            min_val, max_val = range_spec
            min_val = evaluate_expression(min_val, context)
            max_val = evaluate_expression(max_val, context)
            
            min_val = max(min_val, config_min)
            max_val = min(max_val, config_max)
            
            return biased_random_float(min_val, max_val)
        else:
            print(f"Warning: invalid float range_spec {range_spec}, using default range")
            return biased_random_float(config_min, config_max)
    
    elif data_type_lower == "char":
        if isinstance(range_spec, list):
            if len(range_spec) == 1 and isinstance(range_spec[0], list) and len(range_spec[0]) == 2:
                start_char, end_char = range_spec[0]
                start_ord = ord(start_char)
                end_ord = ord(end_char)
                if start_ord == end_ord:
                    return start_char
                return chr(random.randint(start_ord, end_ord))
            
            char_pool = ""
            for item in range_spec:
                item_lower = str(item).lower()
                if item_lower == "lowercase":
                    char_pool += "abcdefghijklmnopqrstuvwxyz"
                elif item_lower == "uppercase":
                    char_pool += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                elif item_lower in ["digit", "digits", "number", "numbers"]:
                    char_pool += "0123456789"
                else:
                    char_pool += str(item)
        
            if not char_pool:
                return "a"
            return random.choice(char_pool)
        else:
            print(f"Warning: invalid char range_spec {range_spec}, using lowercase default")
            return random.choice("abcdefghijklmnopqrstuvwxyz")
    
    elif data_type_lower == "string":
        if isinstance(range_spec, list):
            if not range_spec:
                return "0"
            return random.choice(range_spec)
        else:
            print(f"Warning: invalid string range_spec {range_spec}, using default")
            return "0"
    
    else:
        raise ValueError(f"Unknown data_type '{data_type}', spec: {param_spec}")


def generate_list(param_spec: Dict[str, Any], context: Dict[str, Any]) -> Tuple[List[Any], str]:
    """Generate List values and its splitter."""
    data_type = param_spec.get("data_type")
    range_spec = param_spec.get("range")
    split_char = param_spec.get("split", default_config.get_default_separator("list"))
    
    config_min, config_max = default_config.get_range_limit("list")
    if isinstance(range_spec, list) and len(range_spec) == 2:
        min_len, max_len = range_spec
        min_len = evaluate_expression(min_len, context)
        max_len = evaluate_expression(max_len, context)
        
        min_len = max(min_len, config_min)
        max_len = min(max_len, config_max)
        
        length = biased_random_int(min_len, max_len)
    else:
        print(f"Warning: invalid list range_spec {range_spec}, using default length")
        length = biased_random_int(config_min, config_max)
    
    result = []
    for i in range(length):
        if isinstance(data_type, dict):
            if data_type.get('type') == 'Group':
                element = generate_group(data_type, context)
            else:
                element = generate_parameter("", data_type, context)
        else:
            element = generate_variable({"data_type": data_type, "range": None}, context)
        result.append(element)
    
    return result, split_char


def generate_matrix(param_spec: Dict[str, Any], context: Dict[str, Any]) -> Tuple[List[List[Any]], str, str]:
    """Generate Matrix values with row/column separators."""
    data_type = param_spec.get("data_type")
    row_range = param_spec.get("row_range")
    column_range = param_spec.get("column_range")
    row_split = param_spec.get("row_split", default_config.get_default_separator("row_split"))
    column_split = param_spec.get("column_split", default_config.get_default_separator("column_split"))

    config_min, config_max = default_config.get_range_limit("matrix")

    if isinstance(row_range, list) and len(row_range) == 2:
        min_rows, max_rows = row_range
        min_rows = evaluate_expression(min_rows, context)
        max_rows = evaluate_expression(max_rows, context)
        
        min_rows = max(min_rows, config_min)
        max_rows = min(max_rows, config_max)
        
        rows = biased_random_int(min_rows, max_rows)
    else:
        print(f"Warning: invalid matrix row_range {row_range}, using default rows")
        rows = biased_random_int(config_min, config_max)
    
    if isinstance(column_range, list) and len(column_range) == 2:
        min_cols, max_cols = column_range
        min_cols = evaluate_expression(min_cols, context)
        max_cols = evaluate_expression(max_cols, context)
        
        min_cols = max(min_cols, config_min)
        max_cols = min(max_cols, config_max)
        
        cols = biased_random_int(min_cols, max_cols)
    else:
        print(f"Warning: invalid matrix column_range {column_range}, using default cols")
        cols = biased_random_int(config_min, config_max)
    
    result = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            if isinstance(data_type, dict):
                element = generate_parameter("", data_type, context)
            else:
                element = generate_variable({"data_type": data_type, "range": None}, context)
            row.append(element)
        result.append(row)
    
    return result, row_split, column_split


def generate_group(param_spec: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Generate Group values and output order."""
    parameters = param_spec.get("parameters", {})
    output_order = param_spec.get("output_order", [])
    
    group_context = context.copy()
    
    def extract_group_dependencies(group_params: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract dependencies inside a Group."""
        dependencies = defaultdict(list)
        
        def extract_from_value(value):
            if isinstance(value, str):
                pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?'
                matches = re.findall(pattern, value)
                return matches
            elif isinstance(value, list):
                refs = []
                for item in value:
                    refs.extend(extract_from_value(item))
                return refs
            elif isinstance(value, dict):
                refs = []
                for v in value.values():
                    refs.extend(extract_from_value(v))
                return refs
            return []
        
        for param_name, param_spec in group_params.items():
            dependent_refs = extract_from_value(param_spec)
            for ref in dependent_refs:
                if ref in group_params:
                    dependencies[ref].append(param_name)
        
        for param_name in group_params:
            if param_name not in dependencies:
                dependencies[param_name] = []
        
        return dict(dependencies)
    
    group_dependencies = extract_group_dependencies(parameters)
    group_param_order = topological_sort(group_dependencies)
    
    group_data = {}
    for param_name in group_param_order:
        param_spec = parameters[param_name]
        result = generate_parameter(param_name, param_spec, group_context)
        group_data[param_name] = result
    
    return group_data, output_order


def generate_parameter(param_name: str, param_spec: Dict[str, Any], context: Dict[str, Any]) -> Union[Any, Tuple[Any, str], Tuple[Any, str, str], Tuple[Any, List[Dict[str, str]]]]:
    """Dispatch generator by parameter type."""
    param_type = param_spec.get("type")
    if not param_type:
        raise ValueError(f"Parameter spec missing 'type': {param_spec}")
    
    param_type_lower = str(param_type).lower()
    
    if param_type_lower == "variable":
        result = generate_variable(param_spec, context)
    elif param_type_lower == "list":
        result = generate_list(param_spec, context)
    elif param_type_lower == "matrix":
        result = generate_matrix(param_spec, context)
    elif param_type_lower == "group":
        result = generate_group(param_spec, context)
    else:
        raise ValueError(f"Unsupported param type '{param_type}', spec: {param_spec}")
    
    if param_name and param_type_lower != "group":
        if isinstance(result, tuple):
            context[param_name] = result[0]
        else:
            context[param_name] = result
    
    return result


def format_output(schema: Dict[str, Any], generated_data: Dict[str, Any]) -> str:
    """Format output by schema input_format."""
    input_format = schema.get("input_format", "stdin")
    output_order = schema.get("output_order", [])
    
    if input_format == "function_args":
        return format_function_args(output_order, generated_data, schema)
    else:  # stdin
        return format_stdin(output_order, generated_data)


def format_function_args(output_order: List[Dict[str, str]], generated_data: Dict[str, Any], schema: Dict[str, Any] = None) -> str:
    """Format data for function_args."""
    result_parts = []
    
    for output_item in output_order:
        param_name = output_item["name"]
        if param_name in generated_data:
            data = generated_data[param_name]
            
            if isinstance(data, tuple):
                if len(data) == 2:
                    if isinstance(data[1], str):
                        formatted_data = format_list_for_function(data[0], data[1], schema, param_name)
                    elif isinstance(data[1], list):
                        raise ValueError(f"Group parameters are not supported in function_args: {param_name}")
                elif len(data) == 3:
                    formatted_data = format_matrix_for_function(data[0], data[1], data[2])
            else:
                formatted_data = format_variable_for_function(data, schema, param_name)
            
            result_parts.append(formatted_data)
    
    return "\n".join(result_parts)


def format_stdin(output_order: List[Dict[str, str]], generated_data: Dict[str, Any]) -> str:
    """Format data for stdin."""
    result_parts = []
    
    for i, output_item in enumerate(output_order):
        param_name = output_item["name"]
        separator = output_item.get("separator", "")
        
        if separator:
            separator = separator.encode().decode('unicode_escape')
        
        if param_name in generated_data:
            data = generated_data[param_name]
            
            if isinstance(data, tuple):
                if len(data) == 2:
                    if isinstance(data[1], str):
                        formatted_data = format_list_for_stdin(data[0], data[1])
                    elif isinstance(data[1], list):
                        formatted_data = format_group_for_stdin(data[0], data[1])
                elif len(data) == 3:
                    formatted_data = format_matrix_for_stdin(data[0], data[1], data[2])
            else:
                formatted_data = str(data)
            
            result_parts.append(formatted_data)
            if separator:
                result_parts.append(separator)
    
    return "".join(result_parts)


def format_variable_for_function(data: Any, schema: Dict[str, Any], param_name: str) -> str:
    """Format Variable values for function_args."""
    if schema and param_name in schema.get("parameters", {}):
        param_spec = schema["parameters"][param_name]
        data_type = param_spec.get("data_type", "")
        
        if isinstance(data_type, str):
            data_type_lower = data_type.lower()
        
        return str(data)
    else:
        return str(data)


def format_list_for_function(data: List[Any], split_char: str, schema: Dict[str, Any] = None, param_name: str = "") -> str:
    """Format List values for function_args."""
    if split_char == "":
        data = "".join(str(item) for item in data)
        return f'"{data}"'
    else:
        if data and isinstance(data[0], tuple) and len(data[0]) == 2 and isinstance(data[0][1], list):
            formatted_data = []
            for group_data, group_output_order in data:
                formatted_data.append(format_group_as_list(group_data, group_output_order))
            return str(formatted_data)
        else:
            if data and isinstance(data[0], tuple) and len(data[0]) == 2 and isinstance(data[0][1], str):
                formatted_data = []
                for item_tuple in data:
                    if isinstance(item_tuple, tuple) and len(item_tuple) == 2:
                        inner_data, inner_split = item_tuple
                        if isinstance(inner_data, list):
                            formatted_data.append(inner_data)
                        else:
                            formatted_data.append(inner_data)
                    else:
                        formatted_data.append(item_tuple)
                return str(formatted_data)
            
            if schema and param_name in schema.get("parameters", {}):
                param_spec = schema["parameters"][param_name]
                data_type = param_spec.get("data_type", {})
                if isinstance(data_type, dict):
                    element_type = data_type.get("data_type", "")
                    if isinstance(element_type, str) and element_type.lower() == "string":
                        return str([f'"{item}"' for item in data])
            
            return str(data)


def format_list_for_stdin(data: List[Any], split_char: str) -> str:
    """Format List values for stdin."""
    if split_char:
        split_char = split_char.encode().decode('unicode_escape')
    
    if split_char == "":
        return "".join(str(item) for item in data)
    else:
        if data and isinstance(data[0], tuple) and len(data[0]) == 2 and isinstance(data[0][1], list):
            formatted_groups = []
            for group_data, group_output_order in data:
                formatted_group = format_group_for_stdin(group_data, group_output_order)
                formatted_groups.append(formatted_group)
            return "".join(formatted_groups)
        else:
            return split_char.join(str(item) for item in data)


def format_matrix_for_function(data: List[List[Any]], row_split: str, column_split: str) -> str:
    """Format Matrix values for function_args."""
    return str(data)


def format_matrix_for_stdin(data: List[List[Any]], row_split: str, column_split: str) -> str:
    """Format Matrix values for stdin."""
    if row_split:
        row_split = row_split.encode().decode('unicode_escape')
    if column_split:
        column_split = column_split.encode().decode('unicode_escape')
    
    rows = []
    for row in data:
        row_str = column_split.join(str(item) for item in row)
        rows.append(row_str)
    return row_split.join(rows)


def format_group_for_function(data: Dict[str, Any], output_order: List[Dict[str, str]]) -> str:
    """Format Group values for function_args."""
    result_parts = []
    for output_item in output_order:
        param_name = output_item["name"]
        if param_name in data:
            result_parts.append(str(data[param_name]))
    return "\n".join(result_parts)


def format_group_as_list(data: Dict[str, Any], output_order: List[Dict[str, str]]) -> List[Any]:
    """Convert Group data into ordered list."""
    result = []
    for output_item in output_order:
        param_name = output_item["name"]
        if param_name in data:
            result.append(data[param_name])
    return result


def format_group_for_stdin(data: Dict[str, Any], output_order: List[Dict[str, str]]) -> str:
    """Format Group values for stdin."""
    result_parts = []
    for output_item in output_order:
        param_name = output_item["name"]
        separator = output_item.get("separator", "")
        
        if separator:
            separator = separator.encode().decode('unicode_escape')
        
        if param_name in data:
            param_data = data[param_name]
            if isinstance(param_data, tuple):
                if len(param_data) == 2:
                    if isinstance(param_data[1], str):
                        formatted_data = format_list_for_stdin(param_data[0], param_data[1])
                    elif isinstance(param_data[1], list):
                        formatted_data = format_group_for_stdin(param_data[0], param_data[1])
                elif len(param_data) == 3:
                    formatted_data = format_matrix_for_stdin(param_data[0], param_data[1], param_data[2])
            else:
                formatted_data = str(param_data)
            
            result_parts.append(formatted_data)
            if separator:
                result_parts.append(separator)
    
    return "".join(result_parts)


def generate_data_from_schema(schema_json: str) -> str:
    """Generate random data from schema JSON."""
    schema = parse_schema(schema_json)
    
    dependencies = extract_dependencies(schema)
    
    param_order = topological_sort(dependencies)
    
    context = {}
    generated_data = {}
    
    for param_name in param_order:
        param_spec = schema['parameters'][param_name]
        result = generate_parameter(param_name, param_spec, context)
        generated_data[param_name] = result
        
        if (isinstance(result, tuple) and len(result) == 2 and 
            isinstance(result[0], list) and result[0] and 
            isinstance(result[0][0], tuple) and len(result[0][0]) == 2 and 
            isinstance(result[0][0][1], list)):
            list_data, _ = result
            for i, (group_data, group_output_order) in enumerate(list_data, 1):
                group_name = f"{param_name}_{i}"
                generated_data[group_name] = (group_data, group_output_order)
    
    return format_output(schema, generated_data)



