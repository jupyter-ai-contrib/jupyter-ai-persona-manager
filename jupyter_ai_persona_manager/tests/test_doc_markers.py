"""Conformance check: every public BasePersona member carries a doc marker.

The persona API marks each public member of :class:`BasePersona` with a contract
level (Required / Recommended / Optional / Available to subclasses / Available to
consumers) — see ``jupyter_ai_persona_manager.doc_markers``. The auto-generated
API docs group and badge members by that level, so an *unmarked* member would
silently render without a category.

This test enforces the invariant at CI time: if you add a public method or
property to ``BasePersona`` without marking it (``@mark_required`` /
``@mark_recommended`` / ``@mark_optional`` / ``@mark_subclass_api`` /
``@mark_consumer_api``), this fails with the offending names. It also guards the
two hard requirements (``defaults``, ``process_message``) so a refactor can't
silently downgrade them.
"""

from __future__ import annotations

import inspect

import pytest

from jupyter_ai_persona_manager import BasePersona, ContractLevel
from jupyter_ai_persona_manager.doc_markers import get_contract_level


def _public_members() -> dict[str, object]:
    """Public members defined directly on BasePersona (not inherited)."""
    return {
        name: obj
        for name, obj in vars(BasePersona).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or isinstance(obj, property))
    }


def test_every_public_member_is_marked():
    unmarked = [
        name
        for name, obj in _public_members().items()
        if get_contract_level(obj) is None
    ]
    assert not unmarked, (
        "These public BasePersona members are missing a doc marker "
        "(@mark_required / @mark_recommended / @mark_optional / "
        "@mark_subclass_api / @mark_consumer_api): " + ", ".join(sorted(unmarked))
    )


@pytest.mark.parametrize("name", ["defaults", "process_message"])
def test_required_members_stay_required(name):
    obj = vars(BasePersona)[name]
    assert get_contract_level(obj) is ContractLevel.REQUIRED
    # A Required member must also actually be abstract, so the class can't be
    # instantiated without it.
    target = obj.fget if isinstance(obj, property) else obj
    assert getattr(target, "__isabstractmethod__", False), (
        f"{name} is tagged REQUIRED but is not abstract"
    )


def test_only_required_members_are_abstract():
    """Every abstract member is REQUIRED, and vice versa — the tag and the
    `abstractmethod` reality can't drift apart."""
    for name, obj in _public_members().items():
        target = obj.fget if isinstance(obj, property) else obj
        is_abstract = getattr(target, "__isabstractmethod__", False)
        level = get_contract_level(obj)
        if is_abstract:
            assert level is ContractLevel.REQUIRED, (
                f"{name} is abstract but tagged {level}; abstract members must be REQUIRED"
            )
