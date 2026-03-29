"""GSM8K parameterized ``solve`` + ``ans = solve(...)``: LLM code, AST parse, execute."""

import ast
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.baselines.base_solver import BaseSolver
from src.core.problem import Problem
from src.core.solution import Solution
from src.engine import get_engine

def _load_pot_module():
    _path = Path(__file__).resolve().parent.parent / "Program-of-Thoughts" / "pot_solver.py"
    spec = importlib.util.spec_from_file_location("gsm8k_pot_solver", _path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pot = _load_pot_module()
safe_execute = _pot.safe_execute
floatify_ans = _pot.floatify_ans
_extract_code = _pot._extract_code

_PM_FEW_SHOT_EXAMPLES = """\
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

def solve(total_eggs, eaten_eggs, baked_eggs, dollars_per_egg):
    sold_eggs = total_eggs - eaten_eggs - baked_eggs
    return sold_eggs * dollars_per_egg

ans = solve(16, 3, 4, 2)

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

def solve(bolts_of_blue_fiber):
    bolts_of_white_fiber = bolts_of_blue_fiber / 2
    return bolts_of_blue_fiber + bolts_of_white_fiber

ans = solve(2)

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?

def solve(cost_of_original_house, increase_rate_percent, cost_of_repair):
    increase_rate = increase_rate_percent / 100
    value_of_house = (1 + increase_rate) * cost_of_original_house
    return value_of_house - cost_of_repair - cost_of_original_house

ans = solve(80000, 150, 50000)

Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?

def solve(num_of_chickens, cups_for_each_chicken, cups_in_the_morning, cups_in_the_afternoon):
    cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
    return cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon

ans = solve(20, 3, 15, 25)

Question: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?

def solve(num_glasses, first_glass_cost, discount_rate):
    second_glass_cost = first_glass_cost * discount_rate
    total = 0
    for i in range(num_glasses):
        if i % 2 == 0:
            total += first_glass_cost
        else:
            total += second_glass_cost
    return total

ans = solve(16, 5, 0.6)

Question: Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?

def solve(total_trail_miles, average_mile_per_hour, first_segment, second_segment, hours_elapsed):
    remaining_miles = total_trail_miles - first_segment - second_segment
    total_hours = total_trail_miles / average_mile_per_hour
    remaining_hours = total_hours - hours_elapsed
    return remaining_miles / remaining_hours

ans = solve(12, 4, 4, 2, 2)

Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?

def solve(planting_cost, cost_of_watering_and_feeding, cost_of_each_lemon, num_of_lemon_per_year):
    total_cost = planting_cost
    years = 0
    while total_cost > 0:
        total_cost += cost_of_watering_and_feeding
        total_cost -= num_of_lemon_per_year * cost_of_each_lemon
        years += 1
    return years

ans = solve(90, 3, 1.5, 7)

Question: When Freda cooks canned tomatoes into sauce, they lose half their volume. Each 16 ounce can of tomatoes that she uses contains three tomatoes. Freda's last batch of tomato sauce made 32 ounces of sauce. How many tomatoes did Freda use?

def solve(lose_rate, tomatoes_per_can, ounces_per_can, ounce_sauce_in_last_batch):
    tomato_per_ounce_sauce = tomatoes_per_can / ounces_per_can
    tomato_in_last_batch = ounce_sauce_in_last_batch * tomato_per_ounce_sauce
    return tomato_in_last_batch / (1 - lose_rate)

ans = solve(0.5, 3, 16, 32)

Question: Jordan wanted to surprise her mom with a homemade birthday cake. From reading the instructions, she knew it would take 20 minutes to make the cake batter and 30 minutes to bake the cake. The cake would require 2 hours to cool and an additional 10 minutes to frost the cake. If she plans to make the cake all on the same day, what is the latest time of day that Jordan can start making the cake to be ready to serve it at 5:00 pm?

def solve(minutes_to_make_batter, minutes_to_bake, hours_to_cool, minutes_to_frost, serve_hour):
    total_minutes = minutes_to_make_batter + minutes_to_bake + hours_to_cool * 60 + minutes_to_frost
    total_hours = total_minutes / 60
    return serve_hour - total_hours

ans = solve(20, 30, 2, 10, 5)"""

PM_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math word problems by writing Python code.\n"
    "Write a Python **function** named `solve` that takes the problem's numerical "
    "inputs as parameters and returns the final numeric answer.\n"
    "After the function definition, call it with the original values from the problem "
    "and store the result in `ans`.\n"
    "Output ONLY the Python code with no explanation."
)

def parse_pm_output(code: str) -> Dict[str, Any]:
    """Parse ``solve`` def, param names, and literal args from ``ans = solve(...)``."""
    result: Dict[str, Any] = {
        "function_code": "",
        "param_names": [],
        "original_values": [],
        "full_code": code,
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result

    lines = code.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            result["param_names"] = [arg.arg for arg in node.args.args]
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else len(lines)
            result["function_code"] = "\n".join(lines[start:end])

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id == "ans"):
                    continue
                if not isinstance(node.value, ast.Call):
                    continue
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "solve":
                    result["original_values"] = [
                        _ast_literal_value(a) for a in call.args
                    ]

    return result


def _ast_literal_value(node: ast.expr) -> Any:
    """Literal from AST expr (best effort)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_literal_value(node.operand)
        if val is not None:
            return -val
    try:
        return ast.literal_eval(ast.unparse(node))
    except Exception:
        return None


class GSM8KPMSolver(BaseSolver):
    """LLM writes PM code; run and return formatted answer."""

    def __init__(
        self,
        name: str = "GSM8KPM",
        model_name: str = "gpt-4o",
        **kwargs,
    ):
        super().__init__(name, model_name=model_name, **kwargs)
        self.model_name = model_name
        self._engine = None
        self.engine_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        }

    def solve(self, problem: Problem) -> Solution:
        try:
            engine = self._get_engine()

            user_prompt = self._build_prompt(problem.question_content.strip())
            response = engine.generate(
                user_prompt,
                system_prompt=PM_SYSTEM_PROMPT,
                **self.engine_kwargs,
            )

            code_text = _extract_code(response.strip())

            parsed = parse_pm_output(code_text)

            ans = safe_execute(code_text)
            executed = floatify_ans(ans)

            if executed is not None:
                output_text = f"{code_text}\n\n#### {executed}"
            else:
                output_text = code_text

            return Solution(
                code=output_text,
                problem_id=problem.question_id,
                is_normal_end=bool(code_text),
                metadata={
                    "raw_code": code_text,
                    "function_code": parsed["function_code"],
                    "param_names": parsed["param_names"],
                    "original_values": parsed["original_values"],
                    "executed_answer": executed,
                },
            )
        except Exception as e:
            print(f"Error in GSM8KPMSolver.solve() for {problem.question_id}: {e}")
            return Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
                metadata={"error": str(e)},
            )

    @staticmethod
    def _build_prompt(question: str) -> str:
        return (
            "Here are some examples of solving math problems with a Python function.\n"
            "Each example shows the question followed by a `solve` function and a call "
            "with the original values.\n\n"
            + _PM_FEW_SHOT_EXAMPLES
            + f"\nQuestion: {question}\n"
        )

    def _get_engine(self):
        if self._engine is None:
            self._engine = get_engine(self.model_name, **self.engine_kwargs)
        return self._engine
