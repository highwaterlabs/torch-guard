"""Rule registry.

Importing this package registers every built-in rule. Adding a rule means dropping a
module here and decorating the class with ``@register`` — nothing else to wire up.
"""

from .base import RULES, Rule, RuleDispatcher, all_rules, register

# Import for side effects: each module registers its rule on import.
from . import tg001_graph_retention  # noqa: F401
from . import tg002_missing_no_grad  # noqa: F401
from . import tg003_missing_zero_grad  # noqa: F401
from . import tg004_dataloader  # noqa: F401
from . import tg005_softmax_ce  # noqa: F401
from . import tg010_projected_oom  # noqa: F401

__all__ = ["RULES", "Rule", "RuleDispatcher", "all_rules", "register"]
