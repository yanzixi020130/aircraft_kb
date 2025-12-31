import uvicorn
import os
import sys
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

# Allow importing from src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from engine import find_formulas_by_quantity, solve_targets_auto  # noqa: E402

app = FastAPI()


class FindFormulasByQuantityRequest(BaseModel):
    category: str
    quantity_id: str


@app.post("/formulas/by-quantity")
def api_find_formulas_by_quantity(
    payload: FindFormulasByQuantityRequest,
) -> Dict[str, Any]:
    return find_formulas_by_quantity(
        category=payload.category,
        quantity_id=payload.quantity_id,
    )


class SolveTargetsAutoRequest(BaseModel):
    category: str
    known_inputs: Dict[str, str]
    targets: list[str]
    formula_overrides: Dict[str, str] | None = None
    max_steps: int = 50


@app.post("/solve/targets-auto")
def api_solve_targets_auto(
    payload: SolveTargetsAutoRequest,
) -> Dict[str, Any]:
    return solve_targets_auto(
        category=payload.category,
        known_inputs=payload.known_inputs,
        targets=payload.targets,
        formula_overrides=payload.formula_overrides,
        max_steps=payload.max_steps,
    )

if __name__ == "__main__":
    uvicorn.run(
        app='main:app', 
        host="0.0.0.0", 
        port=1420, 
        reload=False,
    )
