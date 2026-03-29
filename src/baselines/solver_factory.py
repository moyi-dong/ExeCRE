"""Factory for constructing baseline solvers by name."""

from src.baselines.base_solver import BaseSolver


def create_solver(
    solver_type: str,
    **kwargs,
) -> BaseSolver:
    """
    Args:
        solver_type: One of DirectAnswer, Bruteforce, ExeCRE, Textgrad, GSM8KPM, GSM8KExeCRE.
        **kwargs: Passed through to the concrete solver.
    """
    if solver_type == "DirectAnswer":
        from src.baselines.direct_solve.solver import DirectAnswerSolver

        return DirectAnswerSolver(**kwargs)

    elif solver_type == "Bruteforce":
        from src.baselines.bruteforce_solve.solver import BruteforceSolver

        return BruteforceSolver(**kwargs)

    elif solver_type in ("ExeCRE", "TrustTestEM"):
        from src.baselines.execre.solver import TrustTestSolver

        return TrustTestSolver(**kwargs)

    elif solver_type == "Textgrad":
        from src.baselines.textgrad.solver import TextgradSolver

        return TextgradSolver(**kwargs)

    elif solver_type == "GSM8KPM":
        from src.baselines.gsm8k.pm_solver import GSM8KPMSolver

        return GSM8KPMSolver(**kwargs)

    elif solver_type == "GSM8KExeCRE":
        from src.baselines.gsm8k.execre_solver import GSM8KExeCRESolver

        return GSM8KExeCRESolver(**kwargs)

    else:
        raise ValueError(f"Unsupported solver type: {solver_type}")


def list_available_solvers() -> list[str]:
    return [
        "DirectAnswer",
        "Bruteforce",
        "ExeCRE",
        "Textgrad",
        "GSM8KPM",
        "GSM8KExeCRE",
    ]
