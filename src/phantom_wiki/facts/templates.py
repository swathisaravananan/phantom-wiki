"""Generates formal question templates and corresponding Prolog queries using a context-free grammar (CFG).


*Question templates* are based on the QA_GRAMMAR_STRING and can be generated at varying recursion depths.
The templates are not complete questions or valid queries, as they contain the <placeholder> tokens intended
to be replaced with their instantiations that depend on their type and the Prolog database.

For example, at a low recursion depth the grammar string may generate two questions:

    > Question: Who is the person whose hobby is reading?
    > Template: Who is the person whose <attribute_name>_1 is <attribute_value>_1 ?

    > Question: How many children does John have?
    > Template: How many <relation_plural>_2 does <name>_1 have?

Therefore a template of a question is an abstraction of many possible questions of the same type.

Note the exact numbering of the <placeholder> tokens may differ based on the chosen recursion depth, and is
there to distinguish tokens potentially generated at the same recursion depth but representing different
tokens.

Each question template has a corresponding template for Prolog query used to obtain the ground truth answer.
For example:

    > Template: Who is the person whose <attribute_name>_1 is <attribute_value>_1 ?
    > Query: <attribute_name>_1(Y_2, <attribute_value>_1).
    > Answer: Y_2

    > Template: How many <relation_plural>_2 does <name>_1 have?
    > Query: aggregate_all(count, distinct(<relation_plural>_2(<name>_1, Y_3)), Count_3).
    > Answer: Count_3

`phantom_wiki.facts.templates.generate_templates` generates tuples of all possible question and Prolog query
templates at a particular recursion depth from the context-free grammar as defined by QA_GRAMMAR_STRING.

This template generation is based on the `nltk.parse.generate` function from the NLTK project, see:
    Source: https://github.com/nltk/nltk/blob/develop/nltk/parse/generate.py
    Natural Language Toolkit: Generating from a CFG

    Copyright (C) 2001-2024 NLTK Project
    Author: Steven Bird <stevenbird1@gmail.com>
            Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
            Eric Kafe <kafe.eric@gmail.com>
    URL: https://www.nltk.org/
    For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
"""

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from nltk import CFG, Nonterminal

QA_GRAMMAR_STRING = """
    S -> 'Who is' R '?' | 'What is' A '?' | 'How many' RN_p 'does' R_c 'have' '?'
    R -> 'the' RN 'of' R_c | 'the person whose' AN 'is' AV
    R_c -> R | N
    A -> 'the' AN 'of' R
    RN -> '<relation>'
    RN_p -> '<relation_plural>'
    AN -> '<attribute_name>'
    AV -> '<attribute_value>'
    N -> '<name>'
    """

# Question type categories for filtering
QUESTION_TYPE_BASE = "base"
QUESTION_TYPE_COMPARISON = "comparison"
QUESTION_TYPE_MULTI_CONSTRAINT = "multi_constraint"
QUESTION_TYPE_SUPERLATIVE = "superlative"
ALL_QUESTION_TYPES = [
    QUESTION_TYPE_BASE,
    QUESTION_TYPE_COMPARISON,
    QUESTION_TYPE_MULTI_CONSTRAINT,
    QUESTION_TYPE_SUPERLATIVE,
]

# Extended question subtypes (used as answer_info markers and in generate_dataset)
COMPARISON_AGE_SUBTYPE = "comparison_age"
COMPARISON_AGE_YOUNGER_SUBTYPE = "comparison_age_younger"
COMPARISON_BORN_FIRST_SUBTYPE = "comparison_born_first"
MULTI_CONSTRAINT_2_SUBTYPE = "multi_constraint_2"
MULTI_CONSTRAINT_3_SUBTYPE = "multi_constraint_3"
COMPARISON_AGE_MC2_SUBTYPE = "comparison_age_mc2"
COMPARISON_AGE_YOUNGER_MC2_SUBTYPE = "comparison_age_younger_mc2"
COMPARISON_BORN_FIRST_MC2_SUBTYPE = "comparison_born_first_mc2"
SUPERLATIVE_OLDEST_MC2_SUBTYPE = "superlative_oldest_mc2"
SUPERLATIVE_YOUNGEST_MC2_SUBTYPE = "superlative_youngest_mc2"
SUPERLATIVE_OLDEST_SUBTYPE = "superlative_oldest"
SUPERLATIVE_YOUNGEST_SUBTYPE = "superlative_youngest"


def classify_question_type(question_template: list[str]) -> str:
    """Classify a question template into one of the extended types based on its structure."""
    joined = " ".join(question_template)
    if any(kw in joined for kw in ["older,", "younger,", "born first,"]):
        return QUESTION_TYPE_COMPARISON
    if "and whose" in joined:
        return QUESTION_TYPE_MULTI_CONSTRAINT
    if any(kw in joined for kw in ["oldest", "youngest"]):
        return QUESTION_TYPE_SUPERLATIVE
    return QUESTION_TYPE_BASE


def is_aggregation_question(question: str) -> bool:
    """
    Returns True if the question is an aggregation question.

    Definition of aggregation question:
    -----------------------------------
    1. Starts with `"How many"`
    -----------------------------------
    For example, "How many children does John have?" is an aggregation question.
    """
    return question.strip().startswith("How many")


def generate_templates(grammar: CFG = None, depth=4, question_types=None) -> Iterable:
    """Generates an iterator of all question templates and corresponding Prolog queries from a CFG.

    To generate valid Prolog queries, the grammar is assumed to contain <placeholder> terminals with
    <relation>s (with <relation_plural>s for counting queries), <attribute_name>s (corresponding to
    concepts like "job" or "hobby") and matching <attribute_value>s (e.g., "architect" or "running").

    Args:
        grammar: The CFG used to generate questions and queries.
            By default, the grammar is based on QA_GRAMMAR_STRING.
        depth: The maximal depth of the generated tree.
            Default value 4, minimum depth of QA_GRAMMAR_STRING.
        question_types: List of question type strings to include. If None, only base types.
            Use ALL_QUESTION_TYPES or a subset like ["base", "comparison", "superlative"].

    Returns:
        A list of tuples. Base templates are 3-tuples: (question_template, query_template, answer).
        Extended templates are 4-tuples: (question_template, query_template, answer_info, question_subtype).
    """
    if grammar is None:
        grammar = CFG.fromstring(QA_GRAMMAR_STRING)

    start = grammar.start()
    if depth is None:
        # Safe default, assuming the grammar may be recursive:
        depth = (sys.getrecursionlimit() // 3) - 3

    if question_types is None:
        question_types = [QUESTION_TYPE_BASE]

    templates = []

    # Generate base templates from the standard grammar
    if QUESTION_TYPE_BASE in question_types:
        fragments = _generate_tail_template_fragments(grammar, [start], depth, depth)
        for fragment in fragments:
            templates.append((fragment.q_fragment, fragment.p_fragment, fragment.p_answer))

    # Generate extended templates by composing R_c fragments from the base grammar
    extended_types = set(question_types) - {QUESTION_TYPE_BASE}
    if extended_types:
        rc_nonterminal = Nonterminal("R_c")
        templates += _generate_extended_templates(grammar, rc_nonterminal, depth, extended_types)

    return templates


def _generate_extended_templates(
    grammar: CFG,
    rc_nonterminal: Nonterminal,
    depth: int,
    extended_types: set[str],
) -> list[tuple]:
    """Generate extended question templates by composing R_c fragments from the base grammar.

    Uses R_c fragments (which recurse via R_c -> R | N) as building blocks for comparison,
    multi-constraint, and superlative question templates. Each R_c fragment already has the
    correct question text and Prolog query for its chain depth.

    Also feeds multi-constraint fragments (0 hops, 2 constraints) into the comparison and
    superlative builders so that one branch can carry 2 constraints while the other carries
    hops — unlocking (hops>=1, constraints>=2) cells.

    Returns 4-tuples: (question_template, query_template, answer_info, question_subtype).
    """
    templates = []

    rc_fragments = _generate_head_template_fragments(grammar, rc_nonterminal, depth, depth)
    mc2_fragments = _build_mc2_fragments(depth)
    # Deep R_c fragments bypass the CFG depth=6 cap so mc2 builders can produce
    # (hops>=3, constraints>=2) templates, and the superlative builder can
    # produce (hops>=4, constraints=1) templates.
    deep_rc_fragments = _build_deep_rc_fragments()

    if QUESTION_TYPE_COMPARISON in extended_types:
        templates += _build_comparison_templates(grammar, rc_fragments, depth)
        templates += _build_comparison_mc2_templates(
            rc_fragments + deep_rc_fragments, mc2_fragments, depth
        )

    if QUESTION_TYPE_MULTI_CONSTRAINT in extended_types:
        templates += _build_multi_constraint_templates(depth)

    if QUESTION_TYPE_SUPERLATIVE in extended_types:
        templates += _build_superlative_templates(
            grammar, rc_fragments + deep_rc_fragments, depth
        )
        templates += _build_superlative_mc2_templates(mc2_fragments, depth)

    return templates


def _build_mc2_fragments(depth: int) -> list["Fragment"]:
    """Build Fragment objects representing a 2-attribute multi-constraint person lookup.

    Each fragment represents: 'the person whose AN1 is AV1 and whose AN2 is AV2'
    with query atoms [AN1(Y_person, AV1), AN2(Y_person, AV2)] and answer Y_person.

    These are used as R_c substitutes in comparison/superlative builders so that one
    branch can carry 2 constraints, enabling (hops>=1, constraints>=2) questions.
    """
    # Use a high offset to avoid variable collisions with base R_c fragments
    offset = depth * 10
    an1 = f"<attribute_name>_{offset}"
    av1 = f"<attribute_value>_{offset}"
    an2 = f"<attribute_name>_{offset + 1}"
    av2 = f"<attribute_value>_{offset + 1}"
    person_var = f"Y_{offset}"

    q = ["the person whose", an1, "is", av1, "and whose", an2, "is", av2]
    p = [f"{an1}({person_var}, {av1})", f"{an2}({person_var}, {av2})"]
    return [Fragment(q_fragment=q, p_fragment=p, p_answer=person_var)]


def _renumber_fragment(fragment: "Fragment", offset: int) -> "Fragment":
    """Add offset to all numeric subscripts (_N) in a fragment to avoid variable collisions."""
    pattern = re.compile(r"_(\d+)")

    def add_offset(s):
        return pattern.sub(lambda m: f"_{int(m.group(1)) + offset}", s)

    return Fragment(
        q_fragment=[add_offset(t) for t in fragment.q_fragment],
        p_fragment=[add_offset(t) for t in fragment.p_fragment],
        p_answer=add_offset(fragment.p_answer) if fragment.p_answer else None,
    )


def _get_person_var(fragment: "Fragment") -> str:
    """Get the variable/placeholder that resolves to a person from an R_c fragment.

    For N (terminal name): the placeholder like '<name>_1'
    For R (chain): the answer variable like 'Y_3'
    """
    if fragment.p_answer:
        return fragment.p_answer
    return fragment.p_fragment[0]


def _count_hops(fragment: "Fragment") -> int:
    """Count the number of relation hops in an R_c fragment.

    N (name) -> 0 hops
    R (attr only) -> 0 hops (attribute constraint, no relation traversal)
    R (chain) -> number of relation predicates in p_fragment
    """
    return sum(
        1 for atom in fragment.p_fragment
        if re.match(r"<relation>_\d+", atom)
    )


def _build_explicit_chain_rc_fragment(
    n_hops: int,
    anchor_kind: str,
    offset: int,
) -> "Fragment":
    """Build an R_c fragment with exactly n_hops relation placeholders.

    Bypasses the CFG depth limit to unlock (hops>=3, constraints>=2) cells.
    Used by the mc2 and superlative builders to add deeper chain variants.

    Args:
        n_hops: Number of chained <relation> placeholders (>= 1).
        anchor_kind: 'name' or 'attr' — the innermost anchor.
        offset: Starting index for all placeholder/variable subscripts.
            Must be chosen to avoid collisions with CFG-generated fragments
            (indices <= ~20 at depth=6) and mc2 fragments (60-61 at depth=6).

    Chain semantics (Prolog reads ``rel(X, Y)`` as "Y is rel of X"):
        answer = Y_{offset}
        rel_0(Y_1, Y_0), rel_1(Y_2, Y_1), ..., rel_{n-1}(anchor, Y_{n-1})
    """
    assert n_hops >= 1, "n_hops must be >= 1"
    assert anchor_kind in ("name", "attr")

    q: list[str] = []
    p: list[str] = []

    for i in range(n_hops):
        q.extend(["the", f"<relation>_{offset + i}", "of"])

    if anchor_kind == "name":
        anchor_arg = f"<name>_{offset + n_hops}"
        q.append(anchor_arg)
    else:  # attr
        attr_n = f"<attribute_name>_{offset + n_hops}"
        attr_v = f"<attribute_value>_{offset + n_hops}"
        anchor_arg = f"Y_{offset + n_hops}"
        q.extend(["the person whose", attr_n, "is", attr_v])

    for i in range(n_hops):
        outer = f"Y_{offset + i}"
        inner = anchor_arg if i == n_hops - 1 else f"Y_{offset + i + 1}"
        p.append(f"<relation>_{offset + i}({inner}, {outer})")

    if anchor_kind == "attr":
        p.append(f"<attribute_name>_{offset + n_hops}({anchor_arg}, <attribute_value>_{offset + n_hops})")

    return Fragment(q_fragment=q, p_fragment=p, p_answer=f"Y_{offset}")


def _build_deep_rc_fragments() -> list["Fragment"]:
    """Build explicit R_c fragments for 2- and 3-hop chains (name and attr anchors).

    Provides the deeper chains the CFG grammar cannot reach at depth=6, so that
    mc2 and superlative builders can generate (hops>=3, constraints>=2) and
    (hops>=4, constraints=1) templates.
    """
    fragments: list[Fragment] = []
    # Each (n_hops, anchor) gets a 20-wide subscript slot starting at 200 to
    # stay clear of CFG (<=20) and mc2 (60-61) fragment subscripts.
    specs = [(2, "name"), (2, "attr"), (3, "name"), (3, "attr")]
    for i, (n_hops, anchor) in enumerate(specs):
        offset = 200 + i * 20
        fragments.append(_build_explicit_chain_rc_fragment(n_hops, anchor, offset))
    return fragments


def _build_comparison_mc2_templates(
    rc_fragments: list["Fragment"],
    mc2_fragments: list["Fragment"],
    depth: int,
) -> list[tuple]:
    """Build comparison templates where one branch is a multi-constraint (2-attr) fragment.

    Pairs each R_c fragment (which contributes hops) with a mc2 fragment (which contributes
    2 constraints), producing questions like:
      'Who is older, the wife of the person whose hobby is X
       or the person whose job is Y and whose hobby is Z?'

    This unlocks (hops>=1, constraints>=2) cells since max(rc_constraints=1, mc2_constraints=2)=2
    while max(rc_hops>=1, mc2_hops=0)>=1.
    """
    templates = []
    dob_l_var = f"CmpDL_{depth}"
    dob_r_var = f"CmpDR_{depth}"

    for rc_frag in rc_fragments:
        if _count_hops(rc_frag) == 0:
            continue  # skip name/attribute-only frags — no hops to pair with

        for mc2_frag in mc2_fragments:
            mc2_var = _get_person_var(mc2_frag)
            rc_var = _get_person_var(rc_frag)

            # rc on the left, mc2 on the right
            age_query = (
                [f"{dob_l_var} @< {dob_r_var}",
                 f"dob({rc_var}, {dob_l_var})",
                 f"dob({mc2_var}, {dob_r_var})"]
                + rc_frag.p_fragment
                + mc2_frag.p_fragment
            )
            for subtype, prefix in [
                (COMPARISON_AGE_MC2_SUBTYPE, ["Who is older,"]),
                (COMPARISON_AGE_YOUNGER_MC2_SUBTYPE, ["Who is younger,"]),
                (COMPARISON_BORN_FIRST_MC2_SUBTYPE, ["Who was born first,"]),
            ]:
                q = prefix + rc_frag.q_fragment + ["or"] + mc2_frag.q_fragment + ["?"]
                templates.append((q, age_query, (rc_var, mc2_var), subtype))

            # Also mc2 on the left, rc on the right (reversed)
            age_query_rev = (
                [f"{dob_l_var} @< {dob_r_var}",
                 f"dob({mc2_var}, {dob_l_var})",
                 f"dob({rc_var}, {dob_r_var})"]
                + mc2_frag.p_fragment
                + rc_frag.p_fragment
            )
            for subtype, prefix in [
                (COMPARISON_AGE_MC2_SUBTYPE, ["Who is older,"]),
                (COMPARISON_AGE_YOUNGER_MC2_SUBTYPE, ["Who is younger,"]),
                (COMPARISON_BORN_FIRST_MC2_SUBTYPE, ["Who was born first,"]),
            ]:
                q = prefix + mc2_frag.q_fragment + ["or"] + rc_frag.q_fragment + ["?"]
                templates.append((q, age_query_rev, (mc2_var, rc_var), subtype))

    return templates


def _build_superlative_mc2_templates(
    mc2_fragments: list["Fragment"],
    depth: int,
    chain_depths: list[int] = (1, 2, 3),
) -> list[tuple]:
    """Build superlative templates where the anchor chain is a multi-constraint fragment.

    'Who is the oldest [relation] of the person whose [attr1] is X and whose [attr2] is Y?'
    'Who is the oldest [rel1] of the [rel2] of the person whose [attr1] is X and ...?'

    The mc2 fragment (0 hops, 2 constraints) provides the innermost anchor person; one or
    more fresh relation placeholders (chain depth N) are stacked on top to add hops,
    producing (hops=N, constraints=2) questions. Depth-1 chains were the original
    behavior; depths 2+ unlock (hops>=2, constraints=2) cells.
    """
    templates = []
    sup_d1 = f"SupD_{depth}"
    sup_d2 = f"SupD2_{depth}"
    sup_z = f"SupZ_{depth}"

    for chain_depth in chain_depths:
        assert chain_depth >= 1
        # Subscript base for this chain's relation/answer variables.
        # Use 300+ to stay clear of CFG (<=20), mc2 (60-61), deep-rc (200-280).
        base = 300 + chain_depth * 10
        answer_var = f"Y_{base}"
        # The chain: answer = Y_base, then Y_{base+1}, ..., Y_{base+chain_depth-1}
        # With rel_i(inner, outer) semantics:
        # rel_0(Y_{base+1}, Y_base), rel_1(Y_{base+2}, Y_{base+1}), ...,
        # rel_{N-1}(mc2_person, Y_{base+N-1})

        for mc2_frag in mc2_fragments:
            mc2_person = _get_person_var(mc2_frag)

            # Build chain atoms (in query order: innermost first so reversal executes outermost first)
            chain_q: list[str] = []
            chain_p: list[str] = []
            for i in range(chain_depth):
                chain_q.extend(["the", f"<relation>_{base + i}", "of"])
                outer = f"Y_{base + i}"
                inner = mc2_person if i == chain_depth - 1 else f"Y_{base + i + 1}"
                chain_p.append(f"<relation>_{base + i}({inner}, {outer})")

            # Outer superlative wraps the chain: "the oldest <rel>_base of (chain of mc2)"
            # For chain_depth >= 2, the outer <rel>_base is already in chain_p[0].
            # The negation-as-failure uses the SAME outer relation and the chain inner target.
            # If chain_depth == 1: outer inner = mc2_person directly.
            # If chain_depth >= 2: outer's inner is Y_{base+1} (the answer of rel_1).
            if chain_depth == 1:
                nf_inner = mc2_person
            else:
                nf_inner = f"Y_{base + 1}"
            rel_outer = f"<relation>_{base}"

            for subtype, adj, op in [
                (SUPERLATIVE_OLDEST_MC2_SUBTYPE, "oldest", "@<"),
                (SUPERLATIVE_YOUNGEST_MC2_SUBTYPE, "youngest", "@>"),
            ]:
                # Question: "Who is the oldest <rel>_base of the <rel>_{base+1} of ... of <mc2>?"
                # For chain_depth=1: "Who is the oldest <rel>_base of <mc2>?" (strip leading "the")
                q_chain = chain_q.copy()
                # Replace the very first "the" with the adjective marker form
                # The original builder uses ["Who is", f"the {adj}", rel, "of", ...]
                # So for our chain, we replace the first ["the", rel_base, "of"] with
                # ["the ", adj, " ", rel_base, "of"] effectively as ["the {adj}", rel_base, "of"].
                q_chain[0] = f"the {adj}"  # change "the" -> "the oldest"/"the youngest"

                q = ["Who is"] + q_chain + mc2_frag.q_fragment + ["?"]
                query = [
                    f"\\+ ({rel_outer}({nf_inner}, {sup_z}), dob({sup_z}, {sup_d2}), "
                    f"{sup_z} \\= {answer_var}, {sup_d2} {op} {sup_d1})",
                    f"dob({answer_var}, {sup_d1})",
                ] + chain_p + mc2_frag.p_fragment
                templates.append((q, query, answer_var, subtype))

    return templates


def _build_comparison_templates(
    grammar: CFG, rc_fragments: list["Fragment"], depth: int
) -> list[tuple]:
    """Build comparison question templates from pairs of R_c fragments.

    Comparison types:
    - "Who is older, R_c or R_c?" (age comparison)
    - "Who is younger, R_c or R_c?" (age comparison, reversed)
    - "Who was born first, R_c or R_c?" (same as older)
    - "Who has more RN_p, R_c or R_c?" (count comparison)
    - "Who has fewer RN_p, R_c or R_c?" (count comparison, reversed)
    """
    templates = []

    for left_frag in rc_fragments:
        # Renumber right operand to avoid variable collisions with left
        for right_frag_orig in rc_fragments:
            right_frag = _renumber_fragment(right_frag_orig, depth)
            left_var = _get_person_var(left_frag)
            right_var = _get_person_var(right_frag)

            # Age comparisons: older, younger, born first
            # Query: chain queries + dob lookups + comparison
            # After reversal in get_answer, execution order is:
            # chains -> dob lookups -> comparison
            dob_l_var = f"CmpDL_{depth}"
            dob_r_var = f"CmpDR_{depth}"

            age_query = (
                [f"{dob_l_var} @< {dob_r_var}", f"dob({left_var}, {dob_l_var})", f"dob({right_var}, {dob_r_var})"]
                + left_frag.p_fragment
                + right_frag.p_fragment
            )

            for subtype, prefix in [
                (COMPARISON_AGE_SUBTYPE, ["Who is older,"]),
                (COMPARISON_AGE_YOUNGER_SUBTYPE, ["Who is younger,"]),
                (COMPARISON_BORN_FIRST_SUBTYPE, ["Who was born first,"]),
            ]:
                q = prefix + left_frag.q_fragment + ["or"] + right_frag.q_fragment + ["?"]
                # answer_info encodes left/right vars for answer extraction
                answer_info = (left_var, right_var)
                templates.append((q, age_query, answer_info, subtype))

    return templates


def _build_multi_constraint_templates(depth: int) -> list[tuple]:
    """Build multi-constraint question templates.

    Multi-constraint: "Who is the person whose AN is AV and whose AN is AV?"
    This is a leaf production (R) that resolves to a person matching multiple attributes.
    It composes with base questions via R_c -> R.

    The templates use unique depth-based numbering for each attribute pair.
    """
    templates = []
    d = depth

    # 2-attribute multi-constraint: "Who is the person whose AN1 is AV1 and whose AN2 is AV2?"
    an1 = f"<attribute_name>_{d}"
    av1 = f"<attribute_value>_{d}"
    an2 = f"<attribute_name>_{d + 1}"
    av2 = f"<attribute_value>_{d + 1}"
    person_var = f"Y_{d}"

    q_2 = ["Who is", "the person whose", an1, "is", av1, "and whose", an2, "is", av2, "?"]
    query_2 = [f"{an1}({person_var}, {av1})", f"{an2}({person_var}, {av2})"]
    templates.append((q_2, query_2, person_var, MULTI_CONSTRAINT_2_SUBTYPE))

    # 3-attribute multi-constraint
    an3 = f"<attribute_name>_{d + 2}"
    av3 = f"<attribute_value>_{d + 2}"
    q_3 = [
        "Who is",
        "the person whose",
        an1,
        "is",
        av1,
        "and whose",
        an2,
        "is",
        av2,
        "and whose",
        an3,
        "is",
        av3,
        "?",
    ]
    query_3 = [f"{an1}({person_var}, {av1})", f"{an2}({person_var}, {av2})", f"{an3}({person_var}, {av3})"]
    templates.append((q_3, query_3, person_var, MULTI_CONSTRAINT_3_SUBTYPE))

    return templates


def _build_superlative_templates(
    grammar: CFG, rc_fragments: list["Fragment"], depth: int
) -> list[tuple]:
    """Build superlative question templates from R_c fragments.

    Superlative types:
    - "Who is the oldest/youngest RN of R_c?" (age-based)
    - "Who is the RN of R_c with the most/fewest RN_p?" (count-based)

    The superlative itself is an R production that resolves to a person,
    and it composes with R_c via R_c -> R.
    """
    templates = []

    for rc_frag in rc_fragments:
        rc_person = _get_person_var(rc_frag)

        # Age-based superlatives: oldest/youngest
        # R -> 'the oldest/youngest' RN 'of' R_c
        rel_d = f"<relation>_{depth + 1}"
        answer_var = f"Y_{depth + 1}"
        sup_z = f"SupZ_{depth}"
        sup_d1 = f"SupD_{depth}"
        sup_d2 = f"SupD2_{depth}"

        for subtype, adj, op in [
            (SUPERLATIVE_OLDEST_SUBTYPE, "oldest", "@<"),
            (SUPERLATIVE_YOUNGEST_SUBTYPE, "youngest", "@>"),
        ]:
            q = ["Who is", f"the {adj}", rel_d, "of"] + rc_frag.q_fragment + ["?"]
            # Query order: negation, dob, relation, then chain queries
            # After reversal in get_answer: chain -> relation -> dob -> negation
            query = [
                f"\\+ ({rel_d}({rc_person}, {sup_z}), dob({sup_z}, {sup_d2}), "
                f"{sup_z} \\= {answer_var}, {sup_d2} {op} {sup_d1})",
                f"dob({answer_var}, {sup_d1})",
                f"{rel_d}({rc_person}, {answer_var})",
            ] + rc_frag.p_fragment
            templates.append((q, query, answer_var, subtype))

    return templates


@dataclass
class Fragment:
    """Fragment of the question and Prolog query template.

    Represents a subsequence of the CFG.
    If the subsequence is empty, it will be of the form ([], [], None).
    If the subsequence is a single terminal, it will be of the form (['terminal'], [], None).
    If the subsequence is a single <placeholder> terminal, it will be of the form
        (['<placeholder>_*'], ['<placeholder>_*'], None).
    Otherwise, a general subsequence is of the form
        (['Who is', 'the', '<relation>_*', ...], ['<relation>_*(...)', ...], Y_*)

    Attributes:
        q_fragment: Question template of the current fragment
        p_fragment: Prolog query template of the current fragment
        p_answer: The variable in the Prolog query template corresponding to the answer
    """

    q_fragment: list[str] = field(default_factory=list)
    p_fragment: list[str] = field(default_factory=list)
    p_answer: str = None

    def is_empty(self):
        """End-of-production fragment."""
        return not self.q_fragment

    def is_terminal(self):
        """Regular non<placeholder> terminal."""
        return not self.p_fragment

    def is_placeholder(self):
        """Test for <placeholder> terminal."""
        return len(self.p_fragment) == 1 and not self.p_answer

    def get_question_template(self) -> str:
        # TODO remove space before question mark
        return " ".join(self.q_fragment)

    def get_query_template(self) -> str:
        return ", ".join(self.p_fragment)

    def get_query_answer(self) -> str:
        return self.p_answer


def _generate_tail_template_fragments(
    grammar: CFG, items: list[Nonterminal | Any], depth: int, total_depth: int
) -> list[Fragment]:
    """Generates fragments for a list of symbols (`items`) in the grammar.

    Calls `_generate_head_template_fragments` to process the first symbol in `items` and then makes a
    recursive call to process the "remaining symbols" (tail) of the list.
    Then combines all valid subsequences (fragments) resulting from the tail call with all the valid
    subsequences produced from the first symbol.
    """
    if items:
        try:
            fragments = []
            for frag1 in _generate_head_template_fragments(grammar, items[0], depth, total_depth):
                for frag2 in _generate_tail_template_fragments(grammar, items[1:], depth, total_depth):
                    fragments.append(_combine_fragments(frag1, frag2, depth, total_depth))
        except RecursionError as error:
            # Helpful error message while still showing the recursion stack.
            raise RuntimeError(
                "The grammar has rule(s) that yield infinite recursion!\n\
                    Eventually use a lower 'depth', or a higher 'sys.setrecursionlimit()'."
            ) from error
        return fragments
    else:
        # End of production
        return [Fragment()]


def _generate_head_template_fragments(
    grammar: CFG, item: Nonterminal | Any, depth: int, total_depth: int
) -> list[Fragment]:
    """Generates fragments for the current `item` symbol of the grammar.

    Called as a subroutine to process the first symbol (head) of the current production (`item`).
    Recursively calls `_generate_tail_template_fragments` if `item` is a nonterminal in the CFG, for all its
    possible productions.
    If `item` is a <placeholder> or terminal, generates its Fragment.
    """
    if depth > 0:
        if isinstance(item, Nonterminal):
            fragments = []
            for prod in grammar.productions(lhs=item):
                fragments += _generate_tail_template_fragments(grammar, prod.rhs(), depth - 1, total_depth)
            return fragments

        elif re.match(r"<.*?>", item):
            # <placeholder> terminal
            d = total_depth - depth
            return [Fragment([f"{item}_{d}"], [f"{item}_{d}"], None)]
        else:
            # non<placeholder> terminal
            return [Fragment([item], [], None)]

    return []


def _combine_fragments(f1: Fragment, f2: Fragment, depth, total_depth) -> Fragment:
    """Combines two Fragments.

    This merges two CFG subsequences, e.g. 'Who is' and 'the <relation> of ...'.
    The question fragments are combined straightforwardly by concatenating the symbols.
    The Prolog query fragments are combined based on the <placeholder> and query type.

    Examples:
    > (['Who is'], [], None) + (['<name>'], ['<name>'], None) -> (['Who is', '<name>'], ['<name>'], None)
    > (['<attribute_name>_3'], ['<attribute_name>_3'], None)
        + (['of the', '<relation>_2', 'of', '<name>_1', '?'], ['<relation>_2(<name>_1, Y_3)'], 'Y_3')
        -> (['<attribute_name>_3', 'of the', ...],
            ['<attribute_name>_3(Y_3, Y_4)', '<relation>_2(<name>_1, Y_3)'],
            'Y_4')
    """

    q_fragment = f1.q_fragment + f2.q_fragment

    if f1.is_empty():
        return Fragment(q_fragment, f2.p_fragment, f2.p_answer)

    elif f1.is_terminal():
        if f2.is_empty():
            return Fragment(q_fragment, f1.p_fragment, f1.p_answer)
        else:
            return Fragment(q_fragment, f2.p_fragment, f2.p_answer)

    # <placeholder> f1 case (e.g. '<relation>', '<name>',...)
    #   Note: the grammar currently does not allow f1 to be a subquery,
    #   but f2 can be either a subquery, <placeholder>, terminal, or empty
    else:
        if f2.is_empty() or f2.is_terminal():
            return Fragment(q_fragment, f1.p_fragment, f1.p_answer)

        assert f1.is_placeholder()
        placeholder = f1.p_fragment[0]
        subquery = None
        answer = None

        d = total_depth - depth

        if re.match(r"<relation_plural>_(\d+)", placeholder):
            if f2.is_placeholder():
                # ... how many brothers does Alice have ...
                subquery = [
                    (f"aggregate_all(count, distinct({placeholder}({f2.p_fragment[0]}, Y_{d})), Count_{d})")
                ]
            else:
                # ... how many brothers does the sister of Alice have ...
                subquery = [
                    f"aggregate_all(count, distinct({placeholder}({f2.p_answer}, Y_{d})), Count_{d})"
                ] + f2.p_fragment
            answer = f"Count_{d}"

        elif re.match(r"<relation>_(\d+)", placeholder):
            if f2.is_placeholder():
                # ... who is the mother of Alice ...
                subquery = [f"{placeholder}({f2.p_fragment[0]}, Y_{d})"]
            else:
                # ... who is the mother of the mother of Alice ...
                subquery = [f"{placeholder}({f2.p_answer}, Y_{d})"] + f2.p_fragment
            answer = f"Y_{d}"

        elif re.match(r"<attribute_name>_(\d+)", placeholder):
            if placeholder.replace("name", "value") in f2.p_fragment[0]:
                # ... whose hobby is running ...
                assert f2.is_placeholder()
                subquery = [f"{placeholder}(Y_{d}, {f2.p_fragment[0]})"]
            else:
                # ...the hobby of the mother of Alice ...
                subquery = [f"{placeholder}({f2.p_answer}, Y_{d})"] + f2.p_fragment
            answer = f"Y_{d}"

        return Fragment(q_fragment, subquery, answer)
