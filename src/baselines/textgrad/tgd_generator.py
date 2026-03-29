import argparse
import sys
import os
from typing import List, Dict, Any

textgrad_path = os.path.dirname(os.path.abspath(__file__))
if textgrad_path not in sys.path:
    sys.path.insert(0, textgrad_path)

from inference_pipelines.LiveCodeBench.lcb_generator import livecodebench_generator
from inference_pipelines.Textgrad.textgrad.variable import Variable
from inference_pipelines.Textgrad.textgrad.optimizer.optimizer import TextualGradientDescent
from inference_pipelines.Textgrad.prompts import CODE_INSTANCE_ROLE_DESCRIPTION, CodeTestTimewithTests
from inference_pipelines.Textgrad.textgrad.engine import get_engine
from inference_pipelines.Textgrad.py_eval import evaluate
import textgrad
from utls import build_result_dict

ENGINE_API = None

def initialize_engine(model_name: str):
    """Lazy singleton TextGrad engine."""
    global ENGINE_API
    if ENGINE_API is None:
        ENGINE_API = get_engine(model_name)
        textgrad.set_backward_engine(ENGINE_API, override=True)
    return ENGINE_API

def optimization_one_iteration(optimizer, instance_var, prompt, test_string):
    """One optimizer step from eval feedback."""
    optimizer.zero_grad()
    loss_fn = CodeTestTimewithTests(engine=ENGINE_API)
    test_time_loss = loss_fn(prompt, instance_var, test_string)
    test_time_loss.backward()
    optimizer.step()
    return 

def textgrad_generator(problem, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Pipeline entry: see ``utls.build_result_dict`` for result dict shape."""
    initialize_engine(args.model)
    
    results = []
    
    llm_first_result = livecodebench_generator(problem, args)

    instance_var = Variable(llm_first_result[0]["Solution_Code"], requires_grad=True,
                        role_description=CODE_INSTANCE_ROLE_DESCRIPTION)
    
    optimizer = TextualGradientDescent(engine=ENGINE_API,
                                    parameters=[instance_var],
                                    constraints=["Do not add asserts to the code",
                                                "Code must contain imports"])
    
    ### evaluate ###
    passed, test_string = evaluate(instance_var.value, problem.public_test_cases, problem.metadata)
    
    llm_first_result[0]["Local_Passed"] = passed
    llm_first_result[0]["Local_Result_Type"] = test_string
    
    results.extend(llm_first_result)
    
    # print(instance_var.value, problem.public_test_cases, problem.metadata)
    # print(passed, test_string)
    for iter in range(args.TEXTGRAD_MAX_ITERS):
        # if all test passed we early stop
        if ((iter != 0) and passed):
            break

        print(f"{problem.question_id} iter {iter + 1} before_opt passed={passed}")
        optimization_one_iteration(optimizer, instance_var, problem.question_content, test_string)
        passed, test_string = evaluate(instance_var.value, problem.public_test_cases, problem.metadata)
        
        result_dict = build_result_dict(
            problem=problem,
            solution_code=instance_var.value,
            is_normal_end=True,
            round_index=iter + 1,
            local_passed=passed,
            local_result_type=test_string
        )
        results.append(result_dict)
        
    
    return results