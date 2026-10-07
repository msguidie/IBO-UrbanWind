"""The model menu: public `--model` name -> recipe.

    ibo         IBO on the importance-weighted (invsqrt) partition
    ibo_equal   IBO architecture, adaptive partition with unit weights
    ibo_grid    IBO architecture, regular 32 x 32 x 16 grid
    abupt       AB-UPT (Alkin et al., 2025), without the geometry branch
    transolver  Transolver-3 (Zhou et al., 2026)
    spider      SpiderSolver (Qi et al., 2025)
    erwin       Erwin (Zhdanov et al., 2025)

GAOT is not on the UrbanWind menu: at its released configuration its
radius gathering over millions of points does not fit one GPU.
"""
from __future__ import annotations

from .recipes.abupt import AbuptRecipe
from .recipes.base import Recipe
from .recipes.erwin import ErwinRecipe
from .recipes.ibo import IBOEqualRecipe, IBOGridRecipe, IBORecipe
from .recipes.spider import SpiderRecipe
from .recipes.transolver import TransolverRecipe

_RECIPES: dict[str, type[Recipe]] = {
    "ibo": IBORecipe,
    "ibo_equal": IBOEqualRecipe,
    "ibo_grid": IBOGridRecipe,
    "abupt": AbuptRecipe,
    "transolver": TransolverRecipe,
    "spider": SpiderRecipe,
    "erwin": ErwinRecipe,
}

MENU = tuple(_RECIPES)


def get_recipe(name: str) -> Recipe:
    if name not in _RECIPES:
        raise SystemExit(
            f"unknown model {name!r}. Available: " + ", ".join(MENU))
    return _RECIPES[name]()


def recipe_from_config(cfg) -> Recipe:
    """The recipe a run's own config_used.yaml names."""
    return get_recipe(cfg.run.recipe)
