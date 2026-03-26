LLM_SCHEMA_PROMPT = """
You are a programming problem analysis expert. Given a programming problem, you need to generate a structured data specification (schema).

Please analyze the problem description, identify all input data, and return a JSON object using the following format:

{
  "input_format": "stdin" | "function_args",
  "parameters": {
    "parameter_name": {
      "type": "Variable" | "List" | "Matrix" | "Group",
      "data_type": "int" | "float" | "char" | "string",
      "range": [min_value, max_value] | ["character_set"] | [enumeration_values],
      "description": "parameter_description"
    }
  },
  "output_order": [
    {"name": "parameter_name1", "separator": " " or "\\n" or ""},
    {"name": "parameter_name2", "separator": " " or "\\n" or ""},
    ...
  ]
}

## Data Type Definitions

### 1. Variable (Single Data)
Used to represent a single value or character.

```json
{
  "type": "Variable",
  "data_type": "int",
  "range": [1, 1000],
  "description": "number of data items from 1 to 1000"
}
```

### 2. List (List Data)
Used to represent one-dimensional or multi-dimensional lists, supporting nesting. Also used for strings (character lists).

```json
{
  "type": "List",
  "data_type": {
    "type": "Variable",
    "data_type": "int",
    "range": [-1000, 1000]
  },
  "range": [1, 100],
  "split": " ",
  "description": "integer list length rand from 1 to 100, innner int range from -1000 to 1000"
}
```

### 3. String (String Data)
Used to represent strings. Strings can be modeled in two ways:

#### 3.1 Character Lists (for variable-length strings)
Strings should be modeled as Lists of characters.

```json
{
  "type": "List",
  "data_type": {
    "type": "Variable",
    "data_type": "char",
    "range": ["Lowercase"]
  },
  "range": [1, 100000],
  "split": "",
  "description": "string consisting of lowercase English letters"
}
```

#### 3.2 Enumeration Strings (for fixed set of string values)
For strings that can only take specific values from a predefined set, use Variable type with string data_type.

```json
{
  "type": "Variable",
  "data_type": "string",
  "range": ["N", "E", "W", "S", "NE", "NW", "SE", "SW"],
  "description": "direction string"
}
```

### Character Range and Set Definitions

For char data_type, the range field supports different formats:

1. **Character Range**: Use `[["start_char", "end_char"]]` for consecutive characters (also supports Python tuple syntax `[["start_char", "end_char"]]`)
   ```json
   {
     "type": "Variable",
     "data_type": "char",
     "range": [["1", "9"]],
     "description": "digit from 1 to 9"
   }
   ```

2. **Character Set**: Use `["char1", "char2", "char3", "set_name"]` for specific characters or predefined sets
   ```json
   {
     "type": "Variable",
     "data_type": "char",
     "range": ["X", "Y", "Z", "digit"],
     "description": "X, Y, Z or any digit"
   }
   ```

3. **Predefined Sets**: Use predefined character set names
   ```json
   {
     "type": "Variable",
     "data_type": "char",
     "range": ["Lowercase"],  // a-z
     "description": "lowercase letter"
   }
   {
     "type": "Variable",
     "data_type": "char",
     "range": ["Uppercase"],  // A-Z
     "description": "uppercase letter"
   }
   {
     "type": "Variable",
     "data_type": "char",
     "range": ["Digit"],      // 0-9
     "description": "digit"
   }
   ```

### 4. Matrix (Matrix Data)
Specifically designed for representing two-dimensional matrices, avoiding excessive nesting.

```json
{
  "type": "Matrix",
  "data_type": {
    "type": "Variable",
    "data_type": "int", 
    "range": [-1000, 1000]
  },
  "row_range": [1, 100],
  "column_range": [1, 100],
  "row_split": "\\n",
  "column_split": " ",
  "description": "integer matrix"
}
```

### 5. Group (Group Data)
Used to represent a group of related parameters that appear together.

```json
{
  "type": "Group",
  "parameters": {
    "param1": {
      "type": "Variable",
      "data_type": "int",
      "range": [1, 100],
      "description": "first parameter"
    },
    "param2": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [-1000, 1000]
      },
      "range": ["$param1", "$param1"],
      "description": "second parameter"
    }
  },
  "output_order": [
    {"name": "param1", "separator": "\\n"},
    {"name": "param2", "separator": "\\n"}
  ],
  "description": "group of related parameters"
}
```

## Parameter Dependency Handling

When parameters have dependencies, use the `$variable_name` format:

```json
{
  "parameters": {
    "n": {
      "type": "Variable",
      "data_type": "int", 
      "range": [1, 1000],
      "description": "number of data items"
    },
    "values": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [-1000, 1000]
      },
      "range": ["$n", "$n"],
      "description": "list of n values"
    }
  }
}
```

### Mathematical Expressions in Dependencies

For more complex dependencies, you can use mathematical expressions with variables:

```json
{
  "parameters": {
    "n": {
      "type": "Variable",
      "data_type": "int",
      "range": [2, 1000],
      "description": "number of vertices"
    },
    "edges": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [1, 1000]
      },
      "range": ["$n-1", "$n-1"],
      "description": "n-1 edges in the tree"
    }
  }
}
```

Supported mathematical operations:
- Addition: `$n+1`, `$n+5`
- Subtraction: `$n-1`, `$n-10`
- Multiplication: `$n*2`, `$n*3`
- Division: `$n/2` (integer division)
- Complex expressions: `$n*2-1`, `$n/2+1`

### Property Access for Dependencies

When you need to reference the length of a list parameter, use the `.length` property:

```json
{
  "parameters": {
    "nums": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [1, 100]
      },
      "range": [1, 1000],
      "description": "array of integers"
    },
    "k": {
      "type": "Variable",
      "data_type": "int",
      "range": [1, "$nums.length"],
      "description": "index within the array, must be <= nums.length"
    }
  }
}
```

**IMPORTANT**: Use `$variable.length` (with dot notation) to access the length of a list, NOT `$variable_length` (with underscore). The underscore format will cause an error.

## Output Order with Separators

The `output_order` array specifies the actual order of output to stdin or function parameters, along with the separator after each parameter. Parameters not in this array will be treated as hidden parameters (such as dimension parameters in some case).

```json
{
  "output_order": [
    {"name": "values", "separator": " "},
    {"name": "weights", "separator": "\\n"}
  ]
}
```


## Examples

### Example 1: Knapsack Problem
Input: First line contains two integers n and W, second line contains n integers representing values, third line contains n integers representing weights.

```json
{
  "input_format": "stdin",
  "parameters": {
    "n": {
      "type": "Variable",
      "data_type": "int",
      "range": [1, 1000],
      "description": "number of items"
    },
    "W": {
      "type": "Variable", 
      "data_type": "int",
      "range": [1, 10000],
      "description": "knapsack capacity"
    },
    "values": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [1, 1000]
      },
      "range": ["$n", "$n"],
      "description": "value array"
    },
    "weights": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int", 
        "range": [1, 1000]
      },
      "range": ["$n", "$n"],
      "description": "weight array"
    }
  },
  "output_order": [
    {"name": "n", "separator": " "},
    {"name": "W", "separator": "\\n"},
    {"name": "values", "separator": "\\n"},
    {"name": "weights", "separator": "\\n"}
  ]
}
```

### Example 2: Matrix Problem
Input: First line contains two integers n and m, followed by n lines each containing m integers.

```json
{
  "input_format": "stdin", 
  "parameters": {
    "n": {
      "type": "Variable",
      "data_type": "int",
      "range": [1, 100],
      "description": "number of rows"
    },
    "m": {
      "type": "Variable",
      "data_type": "int", 
      "range": [1, 100],
      "description": "number of columns"
    },
    "matrix": {
      "type": "Matrix",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [-1000, 1000]
      },
      "row_range": ["$n", "$n"],
      "column_range": ["$m", "$m"],
      "description": "n×m integer matrix"
    }
  },
  "output_order": [
    {"name": "n", "separator": " "},
    {"name": "m", "separator": "\\n"},
    {"name": "matrix", "separator": "\\n"}
  ]
}
```

### Example 3: Double Sum 3 Problem
Input: First line contains an integer N, second line contains N integers A_1, A_2, ..., A_N, where 1 \leq A_i \leq N.

```json
{
  "input_format": "stdin",
  "parameters": {
    "N": {
      "type": "Variable",
      "data_type": "int",
      "range": [1, 300000],
      "description": "length of the integer sequence"
    },
    "A": {
      "type": "List",
      "data_type": {
        "type": "Variable",
        "data_type": "int",
        "range": [1, "$N"]
      },
      "range": ["$N", "$N"],
      "description": "integer sequence of length N, each element in range [1, N]"
    }
  },
  "output_order": [
    {"name": "N", "separator": "\\n"},
    {"name": "A", "separator": "\\n"}
  ]
}
```

### Example 4: Fixed Format String Problem
Input: A 3-character string S where the first character is a digit, the second character is 'x', and the third character is a digit.

```json
{
  "input_format": "stdin",
  "parameters": {
    "S": {
      "type": "Group",
      "parameters": {
        "char1": {
          "type": "Variable",
          "data_type": "char",
          "range": [["1","9"]]
        },
        "char2": {
          "type": "Variable",
          "data_type": "char",
          "range": ["x"]
        },
        "char3": {
          "type": "Variable",
          "data_type": "char",
          "range": [["1","9"]]
        }
      },
      "output_order": [
        {"name": "char1", "separator": ""},
        {"name": "char2", "separator": ""},
        {"name": "char3", "separator": ""}
      ],
      "description": "3-character string where first and third characters are digits, second is 'x'"
    }
  },
  "output_order": [
    {"name": "S", "separator": "\\n"}
  ]
}
```

### Example 5: Multiple Test Cases
Input: First line contains an integer T, followed by T groups of data, each group's first line contains an integer n, second line contains n integers.

```json
{
  "input_format": "stdin",
  "parameters": {
    "T": {
      "type": "Variable",
      "data_type": "int",
      "range": [1, 100],
      "description": "number of test cases"
    },
    "test_cases": {
      "type": "List",
      "data_type": {
        "type": "Group",
        "parameters": {
          "n": {
            "type": "Variable",
            "data_type": "int",
            "range": [1, 1000],
            "description": "number of integers in current test case"
          },
          "data": {
            "type": "List",
            "data_type": {
              "type": "Variable",
              "data_type": "int",
              "range": [-1000, 1000]
            },
            "range": ["$n", "$n"],
            "description": "n integers for current test case"
          }
        },
        "output_order": [
          {"name": "n", "separator": "\\n"},
          {"name": "data", "separator": "\\n"}
        ]
      },
      "range": ["$T", "$T"],
      "description": "T groups of test data, each group contains n and n integers"
    }
  },
  "output_order": [
    {"name": "T", "separator": "\\n"},
    {"name": "test_cases", "separator": ""}
  ]
}
```

## Important Notes

1. **Data Types**: Supports int, float, char, string
2. **String Representation**: Strings can be modeled in two ways:
   - **Variable-length strings**: Use List of characters with split=""
   - **Enumeration strings**: Use Variable with data_type="string" and range containing the allowed values
3. **Character Sets**: For char type, range supports: character ranges like [["1", "9"]] character sets like ["X", "Y", "Z", "digit"], or predefined sets like ["Lowercase", "Uppercase", "Digit"]
4. **Fixed Format Strings**: Use Group type for strings with specific character constraints at each position
5. **Dependencies**: Use $variable_name to represent dependencies, such as ["$n", "$n"] for a list of length n
6. **Separators**: List defaults to space separation, Matrix defaults to newline for rows and space for columns, String uses empty string as split
7. **Output Order**: output_order determines the actual output parameters, their order, and the separator after each parameter
8. **Group Type**: Use Group type when you have multiple related parameters that appear together
9. **Nested Dependencies**: In Group type, parameters can reference each other using $variable_name
10. **Group Output Order**: Each Group has its own output_order to control the order of its internal parameters
11. **Hidden Parameters**: Parameters not in output_order will be treated as hidden dimension parameters
12. **Separator Values**: Use " " for space, "\\n" for newline, "" for no separator (strings)

Please analyze the given programming problem and generate the corresponding schema JSON object. Return only the JSON, do not include any other explanatory text.
""".strip()