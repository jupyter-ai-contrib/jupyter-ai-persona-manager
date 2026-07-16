"""Documentation markers for the persona API.

**These decorators are for documentation builds only — they do not change any
runtime behavior.** Each ``mark_*`` decorator simply stamps a metadata attribute
(``__contract_level__``) on the function and returns it unchanged. Nothing at
runtime reads that attribute; it exists purely so the API-reference docs can
group members by contract level, and a CI check can assert every public member
is marked.

Every public member of :class:`BasePersona` is marked with a **contract level**
that says what a persona *author* is expected to do with it, using RFC 2119
language:

- ``REQUIRED``    — MUST be implemented (the class is abstract without it).
- ``RECOMMENDED`` — SHOULD be implemented; there is a default, but most real
  personas override it.
- ``OPTIONAL``    — MAY be implemented; a safe default covers personas that
  don't need it.
- ``SUBCLASS``    — provided by ``BasePersona`` for a persona author to *call*
  from inside their persona (may be overridden).
- ``CONSUMER``    — provided by ``BasePersona`` for *consumers* (the
  ``PersonaManager`` and other extensions that interact with personas) to call
  on a persona; should generally not be overridden.

The markers are introspected by:

- the docs build, to group and badge members by level, and
- a CI check, to assert every public member carries exactly one marker.

This module is self-contained and dependency-free by design: to document another
package's API the same way, **copy this file into that package** and mark its
members. (There is no shared home for it yet.)

Usage — mark the underlying function; for a ``property`` mark the getter, and for
an ``abstractmethod`` the decorators compose in any order::

    class BasePersona(...):
        @mark_required
        @property
        @abstractmethod
        def defaults(self) -> PersonaDefaults: ...

        @mark_required
        @abstractmethod
        async def process_message(self, message): ...

        @mark_subclass_api
        def send_message(self, body: str) -> None: ...

        @mark_consumer_api
        def as_user(self) -> User: ...
"""

from __future__ import annotations

import enum
from typing import Callable, TypeVar


class ContractLevel(enum.Enum):
    """How a persona author / consumer is expected to treat a member."""

    REQUIRED = "required"        # MUST implement
    RECOMMENDED = "recommended"  # SHOULD implement
    OPTIONAL = "optional"        # MAY implement
    SUBCLASS = "subclass"        # provided; a persona calls it (may override)
    CONSUMER = "consumer"        # provided; consumers call it (don't override)

    @property
    def label(self) -> str:
        return {
            "required": "Required",
            "recommended": "Recommended",
            "optional": "Optional",
            "subclass": "Available to subclasses",
            "consumer": "Available to consumers",
        }[self.value]

    @property
    def rfc2119(self) -> str:
        return {
            "required": "MUST be implemented by every persona.",
            "recommended": "SHOULD be implemented; most personas override the default.",
            "optional": "MAY be implemented; a safe default is used otherwise.",
            "subclass": "Provided by BasePersona for a persona to call; may be overridden.",
            "consumer": "Provided by BasePersona for consumers to call; should not be overridden.",
        }[self.value]


#: Attribute name used to stamp the level onto a function/getter.
CONTRACT_ATTR = "__contract_level__"

_T = TypeVar("_T")


def _tag(level: ContractLevel) -> Callable[[_T], _T]:
    def decorator(obj: _T) -> _T:
        # For a property, tag the underlying getter (properties are read-only
        # objects, so we stamp fget); otherwise tag the function directly.
        target = obj.fget if isinstance(obj, property) else obj
        if target is None:
            raise TypeError(f"cannot tag {obj!r} with a contract level")
        setattr(target, CONTRACT_ATTR, level)
        return obj

    return decorator


def mark_required(obj: _T) -> _T:
    """Mark a member as :attr:`ContractLevel.REQUIRED` (MUST implement)."""
    return _tag(ContractLevel.REQUIRED)(obj)


def mark_recommended(obj: _T) -> _T:
    """Mark a member as :attr:`ContractLevel.RECOMMENDED` (SHOULD implement)."""
    return _tag(ContractLevel.RECOMMENDED)(obj)


def mark_optional(obj: _T) -> _T:
    """Mark a member as :attr:`ContractLevel.OPTIONAL` (MAY implement)."""
    return _tag(ContractLevel.OPTIONAL)(obj)


def mark_subclass_api(obj: _T) -> _T:
    """Mark a member as :attr:`ContractLevel.SUBCLASS` (a persona calls it)."""
    return _tag(ContractLevel.SUBCLASS)(obj)


def mark_consumer_api(obj: _T) -> _T:
    """Mark a member as :attr:`ContractLevel.CONSUMER` (consumers call it)."""
    return _tag(ContractLevel.CONSUMER)(obj)


def get_contract_level(obj: object) -> ContractLevel | None:
    """Return the contract level tagged on a member, or None if untagged.

    Accepts a function, a ``property`` (checks its getter), or any object; used
    by both the docs build and the CI conformance check.
    """
    target = obj.fget if isinstance(obj, property) else obj
    return getattr(target, CONTRACT_ATTR, None)
