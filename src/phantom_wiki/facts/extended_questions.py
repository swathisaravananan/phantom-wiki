"""Extended question types: comparison, multi-constraint, and superlative questions.

These question types go beyond the base CFG (Who is / What is / How many) and support:

1. **Comparison**: "Who is older, <name_1> or <name_2>?"
   / "Who has more <relation_plural>, <name_1> or <name_2>?"

2. **Multi-constraint**: "Who is the person whose <attr_1> is <val_1> and whose <attr_2> is <val_2>?"

3. **Superlative**: "Who is the oldest <relation> of <name>?"
   / "Who has the most <relation_plural> among the <relation_plural> of <name>?"

Each function generates (question, prolog_query, answer_variable) tuples that can be
sampled and answered using the same pipeline as the base CFG questions.
"""

import re
from copy import copy

from numpy.random import Generator

from .attributes.constants import ATTRIBUTE_ALIASES, ATTRIBUTE_TYPES
from .database import Database
from .family.constants import FAMILY_RELATION_ALIAS, FAMILY_RELATION_DIFFICULTY, FAMILY_RELATION_PLURAL_ALIAS
from .friends.constants import (
    FRIENDSHIP_RELATION,
    FRIENDSHIP_RELATION_ALIAS,
    FRIENDSHIP_RELATION_PLURAL_ALIAS,
)
from .sample import RELATION, RELATION_ALIAS, RELATION_EASY, RELATION_PLURAL_ALIAS, get_relation_bank, get_vals_and_update_cache

# ---------------------------------------------------------------------------
# Question type identifiers
# ---------------------------------------------------------------------------
COMPARISON_AGE_TYPE = "comparison_age"
COMPARISON_COUNT_TYPE = "comparison_count"
MULTI_CONSTRAINT_TYPE = "multi_constraint"
SUPERLATIVE_OLDEST_TYPE = "superlative_oldest"
SUPERLATIVE_YOUNGEST_TYPE = "superlative_youngest"
SUPERLATIVE_MOST_TYPE = "superlative_most"

EXTENDED_QUESTION_TYPES = [
    COMPARISON_AGE_TYPE,
    COMPARISON_COUNT_TYPE,
    MULTI_CONSTRAINT_TYPE,
    SUPERLATIVE_OLDEST_TYPE,
    SUPERLATIVE_YOUNGEST_TYPE,
    SUPERLATIVE_MOST_TYPE,
]


def is_extended_question(question: str) -> bool:
    """Returns True if the question matches an extended question pattern."""
    patterns = [
        r"^Who is older,",
        r"^Who is younger,",
        r"^Who has more ",
        r"^Who is the person whose .+ and whose ",
        r"^Who is the oldest ",
        r"^Who is the youngest ",
        r"^Who has the most ",
    ]
    return any(re.match(p, question.strip()) for p in patterns)


# ---------------------------------------------------------------------------
# Comparison: age
# ---------------------------------------------------------------------------
def sample_comparison_age_question(
    rng: Generator,
    db: Database,
    person_name_bank: list[str],
    person_name2attr_name_and_val: dict[str, list[tuple[str, str]]],
    num_sampling_attempts: int = 100,
) -> tuple[str, list[str]] | None:
    """Sample: 'Who is older, <name_1> or <name_2>?'

    Prolog query:
        dob("<name_1>", D1), dob("<name_2>", D2), D1 @< D2
    Answer variable: Result (the name of the older person)

    We resolve the answer directly: compare DOBs, whoever has the earlier DOB is older.
    The Prolog query is structured so that the answer is deterministic.
    """
    for _ in range(num_sampling_attempts):
        if len(person_name_bank) < 2:
            return None
        idxs = rng.choice(len(person_name_bank), size=2, replace=False)
        name1 = person_name_bank[idxs[0]]
        name2 = person_name_bank[idxs[1]]

        # Check both have DOBs
        dob1_results = db.query(f'dob("{name1}", D)')
        dob2_results = db.query(f'dob("{name2}", D)')
        if not dob1_results or not dob2_results:
            continue

        question = f"Who is older, {name1} or {name2}?"
        # Prolog: earlier DOB = older person
        # We use @< for string comparison (dates are in YYYY-MM-DD format)
        query = [
            f'dob("{name1}", D1)',
            f'dob("{name2}", D2)',
            "D1 @< D2",  # name1 is older if their DOB is earlier
        ]
        # The "answer" here needs special handling - it's a computed result
        # We'll store the answer as a sentinel and compute it during answer extraction
        return question, query

    return None


# ---------------------------------------------------------------------------
# Comparison: count of relations
# ---------------------------------------------------------------------------
def sample_comparison_count_question(
    rng: Generator,
    db: Database,
    person_name_bank: list[str],
    person_name2relation_and_related: dict[str, list[tuple[str, str]]],
    easy_mode: bool = False,
    num_sampling_attempts: int = 100,
    difficulty_level: str | None = None,
) -> tuple[str, list[str]] | None:
    """Sample: 'Who has more <relation_plural>, <name_1> or <name_2>?'

    Prolog query:
        aggregate_all(count, distinct(<relation>("<name_1>", X)), C1),
        aggregate_all(count, distinct(<relation>("<name_2>", Y)), C2),
        C1 > C2
    """
    if difficulty_level is not None:
        relation_bank = get_relation_bank(difficulty_level)
    else:
        relation_bank = RELATION_EASY if easy_mode else RELATION

    for _ in range(num_sampling_attempts):
        if len(person_name_bank) < 2:
            return None
        idxs = rng.choice(len(person_name_bank), size=2, replace=False)
        name1 = person_name_bank[idxs[0]]
        name2 = person_name_bank[idxs[1]]

        relation = relation_bank[rng.integers(0, len(relation_bank))]
        relation_plural = RELATION_PLURAL_ALIAS.get(relation, relation + "s")

        # Check both have at least one of this relation and counts differ
        r1 = db.query(f'distinct({relation}("{name1}", X))')
        r2 = db.query(f'distinct({relation}("{name2}", X))')

        c1 = len(r1)
        c2 = len(r2)
        if c1 == 0 and c2 == 0:
            continue
        if c1 == c2:
            continue  # need a clear winner

        question = f"Who has more {relation_plural}, {name1} or {name2}?"
        query = [
            f'aggregate_all(count, distinct({relation}("{name1}", X)), C1)',
            f'aggregate_all(count, distinct({relation}("{name2}", Y)), C2)',
            "C1 > C2",
        ]
        return question, query

    return None


# ---------------------------------------------------------------------------
# Multi-constraint
# ---------------------------------------------------------------------------
def sample_multi_constraint_question(
    rng: Generator,
    db: Database,
    person_name_bank: list[str],
    person_name2attr_name_and_val: dict[str, list[tuple[str, str]]],
    num_procs: int,
    num_sampling_attempts: int = 100,
) -> tuple[str, list[str]] | None:
    """Sample: 'Who is the person whose <attr_1> is <val_1> and whose <attr_2> is <val_2>?'

    Prolog query:
        <attr_1>(X, "<val_1>"), <attr_2>(X, "<val_2>")
    Answer variable: X
    """
    for _ in range(num_sampling_attempts):
        person_name = person_name_bank[rng.integers(0, len(person_name_bank))]

        attr_name_and_vals: list[tuple[str, str]] = get_vals_and_update_cache(
            cache=person_name2attr_name_and_val,
            key=person_name,
            db=db,
            query_bank=ATTRIBUTE_TYPES,
            num_procs=num_procs,
        )

        if len(attr_name_and_vals) < 2:
            continue

        # Pick two distinct attributes
        idxs = rng.choice(len(attr_name_and_vals), size=2, replace=False)
        attr1_name, attr1_val = attr_name_and_vals[idxs[0]]
        attr2_name, attr2_val = attr_name_and_vals[idxs[1]]

        # Build question using aliases
        attr1_alias = ATTRIBUTE_ALIASES[attr1_name]
        attr2_alias = ATTRIBUTE_ALIASES[attr2_name]

        question = f"Who is the person whose {attr1_alias} is {attr1_val} and whose {attr2_alias} is {attr2_val}?"
        query = [
            f'{attr1_name}(X, "{attr1_val}")',
            f'{attr2_name}(X, "{attr2_val}")',
        ]
        return question, query

    return None


# ---------------------------------------------------------------------------
# Superlative: oldest / youngest relation
# ---------------------------------------------------------------------------
def sample_superlative_age_question(
    rng: Generator,
    db: Database,
    person_name_bank: list[str],
    person_name2relation_and_related: dict[str, list[tuple[str, str]]],
    easy_mode: bool = False,
    oldest: bool = True,
    num_sampling_attempts: int = 100,
    difficulty_level: str | None = None,
) -> tuple[str, list[str]] | None:
    """Sample: 'Who is the oldest/youngest <relation> of <name>?'

    Prolog query (oldest):
        <relation>("<name>", X), dob(X, D),
        \\+ (<relation>("<name>", Y), dob(Y, D2), Y \\= X, D2 @< D)
    The negation ensures no other relative has an earlier DOB (i.e., X has the earliest = oldest).
    """
    if difficulty_level is not None:
        relation_bank = get_relation_bank(difficulty_level)
    else:
        relation_bank = RELATION_EASY if easy_mode else RELATION

    for _ in range(num_sampling_attempts):
        person_name = person_name_bank[rng.integers(0, len(person_name_bank))]
        relation = relation_bank[rng.integers(0, len(relation_bank))]
        relation_alias = RELATION_ALIAS.get(relation, relation)

        # Check this person has at least 2 of this relation (otherwise superlative is trivial)
        results = db.query(f'distinct({relation}("{person_name}", X))')
        if len(results) < 2:
            continue

        adj = "oldest" if oldest else "youngest"
        op = "@<" if oldest else "@>"  # oldest = earliest DOB = @<, youngest = latest DOB = @>

        question = f"Who is the {adj} {relation_alias} of {person_name}?"
        # Prolog: find X such that no Y has a DOB that is earlier (oldest) / later (youngest)
        query = [
            f'{relation}("{person_name}", X)',
            "dob(X, D)",
            f'\\+ ({relation}("{person_name}", Y), dob(Y, D2), Y \\= X, D2 {op} D)',
        ]
        return question, query

    return None


# ---------------------------------------------------------------------------
# Superlative: most relations
# ---------------------------------------------------------------------------
def sample_superlative_most_question(
    rng: Generator,
    db: Database,
    person_name_bank: list[str],
    person_name2relation_and_related: dict[str, list[tuple[str, str]]],
    easy_mode: bool = False,
    num_sampling_attempts: int = 100,
    difficulty_level: str | None = None,
) -> tuple[str, list[str]] | None:
    """Sample: 'Who has the most <relation_plural> among the <relation_plural_2> of <name>?'

    E.g., "Who has the most children among the siblings of Alice?"

    Prolog query:
        <relation_2>("<name>", X),
        aggregate_all(count, distinct(<relation_1>(X, Z)), C),
        \\+ (<relation_2>("<name>", Y), Y \\= X,
             aggregate_all(count, distinct(<relation_1>(Y, W)), C2), C2 > C)
    """
    if difficulty_level is not None:
        relation_bank = get_relation_bank(difficulty_level)
    else:
        relation_bank = RELATION_EASY if easy_mode else RELATION

    for _ in range(num_sampling_attempts):
        person_name = person_name_bank[rng.integers(0, len(person_name_bank))]
        # relation_2 is the group relation (e.g., "siblings of name")
        relation_2 = relation_bank[rng.integers(0, len(relation_bank))]
        # relation_1 is what we count (e.g., "children")
        relation_1 = relation_bank[rng.integers(0, len(relation_bank))]

        relation_1_plural = RELATION_PLURAL_ALIAS.get(relation_1, relation_1 + "s")
        relation_2_plural = RELATION_PLURAL_ALIAS.get(relation_2, relation_2 + "s")

        # Check person has >= 2 of relation_2
        group = db.query(f'distinct({relation_2}("{person_name}", X))')
        if len(group) < 2:
            continue

        # Check at least one member has relation_1 and counts vary
        from ..utils import decode

        counts = {}
        for r in group:
            member = decode(r["X"])
            c = len(db.query(f'distinct({relation_1}("{member}", Z))'))
            counts[member] = c

        if max(counts.values()) == 0:
            continue
        # Ensure there's a unique maximum
        max_count = max(counts.values())
        if list(counts.values()).count(max_count) > 1:
            continue

        question = f"Who has the most {relation_1_plural} among the {relation_2_plural} of {person_name}?"
        query = [
            f'{relation_2}("{person_name}", X)',
            f"aggregate_all(count, distinct({relation_1}(X, Z)), C)",
            f'\\+ ({relation_2}("{person_name}", Y), Y \\= X, '
            f"aggregate_all(count, distinct({relation_1}(Y, W)), C2), C2 > C)",
        ]
        return question, query

    return None


# ---------------------------------------------------------------------------
# Master sampling function
# ---------------------------------------------------------------------------
def sample_extended_question(
    question_type: str,
    rng: Generator,
    db: Database,
    person_name_bank: list[str],
    person_name2attr_name_and_val: dict[str, list[tuple[str, str]]],
    person_name2relation_and_related: dict[str, list[tuple[str, str]]],
    num_procs: int,
    easy_mode: bool = False,
    num_sampling_attempts: int = 100,
    difficulty_level: str | None = None,
) -> tuple[str, list[str]] | None:
    """Sample an extended question of the given type.

    Returns (question_string, prolog_query_list) or None if sampling fails.
    """
    if question_type == COMPARISON_AGE_TYPE:
        return sample_comparison_age_question(
            rng, db, person_name_bank, person_name2attr_name_and_val, num_sampling_attempts
        )
    elif question_type == COMPARISON_COUNT_TYPE:
        return sample_comparison_count_question(
            rng, db, person_name_bank, person_name2relation_and_related,
            easy_mode, num_sampling_attempts, difficulty_level=difficulty_level,
        )
    elif question_type == MULTI_CONSTRAINT_TYPE:
        return sample_multi_constraint_question(
            rng, db, person_name_bank, person_name2attr_name_and_val, num_procs, num_sampling_attempts
        )
    elif question_type == SUPERLATIVE_OLDEST_TYPE:
        return sample_superlative_age_question(
            rng, db, person_name_bank, person_name2relation_and_related,
            easy_mode, True, num_sampling_attempts, difficulty_level=difficulty_level,
        )
    elif question_type == SUPERLATIVE_YOUNGEST_TYPE:
        return sample_superlative_age_question(
            rng, db, person_name_bank, person_name2relation_and_related,
            easy_mode, False, num_sampling_attempts, difficulty_level=difficulty_level,
        )
    elif question_type == SUPERLATIVE_MOST_TYPE:
        return sample_superlative_most_question(
            rng, db, person_name_bank, person_name2relation_and_related,
            easy_mode, num_sampling_attempts, difficulty_level=difficulty_level,
        )
    else:
        raise ValueError(f"Unknown extended question type: {question_type}")


def get_extended_answer(
    question: str,
    query: list[str],
    question_type: str,
    db: Database,
) -> list[str]:
    """Execute an extended question's Prolog query and return the answer.

    For comparison questions, the answer is one of the two names.
    For multi-constraint, the answer is the person(s) matching all constraints.
    For superlative, the answer is the person with the extreme value.
    """
    from ..utils import decode

    if question_type == COMPARISON_AGE_TYPE:
        # Query: dob("name1", D1), dob("name2", D2), D1 @< D2
        # If this succeeds, name1 is older; otherwise name2 is older
        joined = ", ".join(query)
        results = list(db.prolog.query(joined))
        # Extract names from the question
        import re

        m = re.match(r"Who is older, (.+?) or (.+?)\?", question)
        if not m:
            return []
        name1, name2 = m.group(1), m.group(2)
        if results:
            return [name1]  # D1 @< D2 succeeded, name1 is older
        else:
            return [name2]

    elif question_type == COMPARISON_COUNT_TYPE:
        joined = ", ".join(query)
        results = list(db.prolog.query(joined))
        import re

        m = re.match(r"Who has more .+?, (.+?) or (.+?)\?", question)
        if not m:
            return []
        name1, name2 = m.group(1), m.group(2)
        if results:
            return [name1]  # C1 > C2 succeeded
        else:
            return [name2]

    elif question_type == MULTI_CONSTRAINT_TYPE:
        joined = ", ".join(query)
        results = list(db.prolog.query(joined))
        return sorted({decode(r["X"]) for r in results})

    elif question_type in (SUPERLATIVE_OLDEST_TYPE, SUPERLATIVE_YOUNGEST_TYPE):
        joined = ", ".join(query)
        results = list(db.prolog.query(joined))
        return sorted({decode(r["X"]) for r in results})

    elif question_type == SUPERLATIVE_MOST_TYPE:
        joined = ", ".join(query)
        results = list(db.prolog.query(joined))
        return sorted({decode(r["X"]) for r in results})

    return []
