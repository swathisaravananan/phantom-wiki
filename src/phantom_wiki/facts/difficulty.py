"""Structured difficulty computation for Prolog query templates.

Decomposes difficulty into two independent axes:

- **hops**: number of base-level Prolog predicate evaluations needed to
  traverse the relation chain.  Derived relations (e.g. ``nephew``) are
  recursively expanded into base predicates and counted.
- **constraints**: number of attribute-filter predicates (``hobby``, ``job``,
  ``dob``) that appear in the query.

A scalar **composite** (hops + constraints) is provided for simple sorting.
"""

from __future__ import annotations

import functools
import re

from .attributes.constants import ATTRIBUTE_TYPES
from .family.constants import FAMILY_RELATION_DIFFICULTY
from .friends.constants import FRIENDSHIP_RELATION

# Predicates that are gender filters, not relational hops
GENDER_FILTERS = frozenset(["male", "female", "nonbinary", "gender"])


@functools.lru_cache(maxsize=1)
def derive_relation_hop_counts() -> dict[str, int]:
    """Return {relation_name: hop_count} sourced from FAMILY_RELATION_DIFFICULTY
    plus friendship relations (1 hop each).
    """
    hops = dict(FAMILY_RELATION_DIFFICULTY)
    for rel in FRIENDSHIP_RELATION:
        hops[rel] = 1
    return hops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Attribute predicates that count as constraints (not hops)
_ATTRIBUTE_SET = frozenset(ATTRIBUTE_TYPES) | {"dob", "date_of_birth"}
_AGGREGATE_RE = re.compile(r"aggregate_all\(")
_COMPARISON_RE = re.compile(r"@[<>]|\\+|\\=|==|>|<")


def _parse_predicate(query_atom: str) -> tuple[str | None, list[str]]:
    """Extract predicate name and arguments from a Prolog query atom.

    Returns (name, args) or (None, []) for non-predicate atoms.
    """
    query_atom = query_atom.strip()
    if not query_atom:
        return None, []
    if query_atom.startswith("\\+"):
        return None, []
    if _COMPARISON_RE.match(query_atom):
        return None, []
    m = re.match(r"(\w+)\(", query_atom)
    if not m:
        return None, []

    name = m.group(1)
    # Extract the argument string (everything between the outermost parens)
    start = m.end()  # position right after "name("
    depth = 1
    i = start
    while i < len(query_atom) and depth > 0:
        ch = query_atom[i]
        if ch == '"':
            i += 1
            while i < len(query_atom) and query_atom[i] != '"':
                i += 1
        elif ch == "'":
            i += 1
            while i < len(query_atom) and query_atom[i] != "'":
                i += 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1

    arg_str = query_atom[start:i]

    # Split args on commas, respecting quotes and nested parens
    args: list[str] = []
    current: list[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    for ch in arg_str:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())

    return name, args


def _parse_predicate_name(query_atom: str) -> str | None:
    """Extract the predicate name from a Prolog query atom."""
    name, _ = _parse_predicate(query_atom)
    return name


def _is_quoted_literal(s: str) -> bool:
    """True if the string is a quoted literal value (double or single quotes)."""
    s = s.strip()
    return (len(s) >= 2 and s[0] == '"' and s[-1] == '"') or \
           (len(s) >= 2 and s[0] == "'" and s[-1] == "'")


_VARIABLE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
_QUOTED_STR_RE = re.compile(r'"[^"]*"')


def _extract_variables(atom: str) -> set[str]:
    """Extract all Prolog variable names from an atom (uppercase-starting identifiers).

    Quoted string literals (e.g. person names like "Deane Smock") are stripped
    first so their words are not mistaken for Prolog variables.
    """
    atom_no_strings = _QUOTED_STR_RE.sub("", atom)
    return set(_VARIABLE_RE.findall(atom_no_strings)) - GENDER_FILTERS


def _split_comparison_branches(query_template: list[str]) -> tuple[list[str], list[str]] | None:
    """Split a CmpDL/CmpDR comparison query into left and right branch atoms.

    Returns (left_atoms, right_atoms) if the query uses the CmpDL/CmpDR pattern,
    or None if it is not a two-branch comparison query.  Atoms that are part of
    the comparison plumbing (the @< operator, the dob value-retrieval atoms that
    bind CmpDL/CmpDR) are excluded from both branches.
    """
    left_root = right_root = None
    cmp_vars: set[str] = set()

    for atom in query_template:
        if "CmpDL" not in atom and "CmpDR" not in atom:
            continue
        if "@<" in atom or "@>" in atom:
            cmp_vars.update(_extract_variables(atom))
            continue
        # dob(Y_2, CmpDL_6) — binds a person var to a comparison var
        pred, args = _parse_predicate(atom)
        if pred == "dob" and len(args) >= 2:
            var_name = args[0].strip()
            cmp_var = args[1].strip()
            if "CmpDL" in cmp_var:
                left_root = var_name
            elif "CmpDR" in cmp_var:
                right_root = var_name
            cmp_vars.add(var_name)
            cmp_vars.update(_extract_variables(atom))

    if left_root is None or right_root is None:
        return None

    # Build variable connectivity via union-find to assign atoms to branches.
    # Start with left_root connected to "L" and right_root connected to "R".
    _SENTINELS = {"L_BRANCH", "R_BRANCH"}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Always keep sentinel labels as the root
            if rb in _SENTINELS:
                parent[ra] = rb
            else:
                parent[rb] = ra

    # Seed the two branches with sentinel labels
    parent[left_root] = "L_BRANCH"
    parent["L_BRANCH"] = "L_BRANCH"
    parent[right_root] = "R_BRANCH"
    parent["R_BRANCH"] = "R_BRANCH"

    # Collect content atoms (excluding comparison plumbing)
    content_atoms: list[tuple[str, set[str]]] = []
    for atom in query_template:
        if "@<" in atom or "@>" in atom:
            continue
        atom_vars = _extract_variables(atom)
        if atom_vars & cmp_vars and _parse_predicate(atom)[0] == "dob":
            # Skip dob value-retrieval atoms that bind CmpDL/CmpDR
            if any(v.startswith("Cmp") for v in atom_vars):
                continue
        content_atoms.append((atom, atom_vars))

    # Union variables that co-occur in the same atom
    for _, atom_vars in content_atoms:
        var_list = [v for v in atom_vars if not v.startswith("Cmp")]
        for i in range(1, len(var_list)):
            union(var_list[0], var_list[i])

    left_atoms: list[str] = []
    right_atoms: list[str] = []
    for atom, atom_vars in content_atoms:
        var_list = [v for v in atom_vars if not v.startswith("Cmp")]
        if not var_list:
            continue
        root = find(var_list[0])
        if root == "L_BRANCH":
            left_atoms.append(atom)
        elif root == "R_BRANCH":
            right_atoms.append(atom)

    return left_atoms, right_atoms


def _score_atoms(atoms: list[str], hop_counts: dict[str, int]) -> tuple[int, int]:
    """Compute (hops, constraints) for a list of Prolog query atoms."""
    hops = 0
    constraints = 0

    for atom in atoms:
        pred, args = _parse_predicate(atom)
        if pred is None:
            continue

        if pred == "aggregate_all":
            inner = re.search(r"distinct\((\w+)\(", atom)
            if inner:
                inner_pred = inner.group(1)
                if inner_pred in _ATTRIBUTE_SET:
                    constraints += 1
                elif inner_pred in hop_counts:
                    hops += hop_counts[inner_pred]
                else:
                    hops += 1
            else:
                hops += 1
            continue

        if pred in _ATTRIBUTE_SET:
            if len(args) >= 2 and _is_quoted_literal(args[1]):
                constraints += 1
        elif pred in hop_counts:
            hops += hop_counts[pred]
        elif pred in GENDER_FILTERS:
            pass
        else:
            hops += 1

    return hops, constraints


def compute_difficulty(query_template: list[str]) -> dict:
    """Compute structured difficulty from a Prolog query template.

    For two-branch comparison queries (CmpDL/CmpDR pattern), hops and
    constraints are computed per-branch and the **max** of each axis is
    used — an agent can resolve both branches in parallel, so difficulty
    is determined by the harder branch, not the sum.

    Args:
        query_template: List of Prolog query atoms as produced by
            ``generate_templates()`` or sampled questions.

    Returns:
        Dict with keys ``hops``, ``constraints``, ``composite``.
    """
    hop_counts = derive_relation_hop_counts()

    branches = _split_comparison_branches(query_template)
    if branches is not None:
        left_atoms, right_atoms = branches
        l_hops, l_cons = _score_atoms(left_atoms, hop_counts)
        r_hops, r_cons = _score_atoms(right_atoms, hop_counts)
        hops = max(l_hops, r_hops)
        constraints = max(l_cons, r_cons)
    else:
        hops, constraints = _score_atoms(query_template, hop_counts)

    return {
        "hops": hops,
        "constraints": constraints,
        "composite": hops + constraints,
    }
