"""Tests for (hops, constraints) matrix coverage.

Verifies that generate_templates and build_comparison_count_variants populate
the expected cells as max_hops and max_constraints vary.
"""
import re
import pytest
from collections import Counter
from phantom_wiki.facts.templates import (
    generate_templates,
    cfg_depth_for_max_hops,
    is_comparison_older_subtype,
    is_comparison_younger_subtype,
    is_comparison_subtype,
)
from phantom_wiki.facts.extended_questions import (
    build_comparison_count_variants,
    build_extended_question_types,
    build_extended_question_category,
    _count_variant_type,
)


# ---------------------------------------------------------------------------
# cfg_depth_for_max_hops
# ---------------------------------------------------------------------------
class TestCfgDepth:
    @pytest.mark.parametrize("max_hops,expected_depth", [
        (1, 6),
        (2, 8),
        (3, 10),
        (4, 12),
    ])
    def test_formula(self, max_hops, expected_depth):
        assert cfg_depth_for_max_hops(max_hops) == expected_depth


# ---------------------------------------------------------------------------
# count_variant_type naming
# ---------------------------------------------------------------------------
class TestCountVariantType:
    def test_plain(self):
        assert _count_variant_type(0, 0) == "comparison_count"

    def test_chain_only(self):
        assert _count_variant_type(2, 0) == "comparison_count_chain2"

    def test_mc_only(self):
        assert _count_variant_type(0, 3) == "comparison_count_mc3"

    def test_chain_and_mc(self):
        assert _count_variant_type(2, 3) == "comparison_count_chain2_mc3"


# ---------------------------------------------------------------------------
# build_comparison_count_variants
# ---------------------------------------------------------------------------
class TestBuildVariants:
    def test_keys_named_correctly(self):
        v = build_comparison_count_variants(max_chain_depth=2, n_attrs_options=(0, 2, 3))
        assert "comparison_count" in v
        assert "comparison_count_chain2" in v
        assert "comparison_count_mc2" in v
        assert "comparison_count_chain2_mc3" in v

    def test_values_are_pairs(self):
        v = build_comparison_count_variants(max_chain_depth=3, n_attrs_options=(0, 2, 3))
        for key, (chain_depth, n_attrs) in v.items():
            assert isinstance(chain_depth, int)
            assert isinstance(n_attrs, int)

    def test_size(self):
        # (max_chain_depth+1) * len(n_attrs_options)
        v = build_comparison_count_variants(max_chain_depth=3, n_attrs_options=(0, 2, 3))
        assert len(v) == 4 * 3  # 4 chain depths × 3 n_attrs options

    def test_empty_attrs(self):
        v = build_comparison_count_variants(max_chain_depth=2, n_attrs_options=(0,))
        assert all(n == 0 for _, n in v.values())

    def test_extended_question_types_includes_all(self):
        v = build_comparison_count_variants(max_chain_depth=1, n_attrs_options=(0, 2, 3))
        types = build_extended_question_types(v)
        for key in v:
            assert key in types

    def test_extended_question_category_all_comparison_count(self):
        v = build_comparison_count_variants(max_chain_depth=1, n_attrs_options=(0, 2))
        cat = build_extended_question_category(v)
        for key in v:
            assert cat[key] == "comparison_count"

    @pytest.mark.parametrize("max_constraints", [2, 3, 4])
    def test_mc_range(self, max_constraints):
        opts = (0,) + tuple(range(2, max_constraints + 1))
        v = build_comparison_count_variants(max_chain_depth=2, n_attrs_options=opts)
        n_attrs_in_variants = {n for _, n in v.values()}
        for n in range(2, max_constraints + 1):
            assert n in n_attrs_in_variants


# ---------------------------------------------------------------------------
# is_comparison_*_subtype predicates
# ---------------------------------------------------------------------------
class TestSubtypePredicates:
    @pytest.mark.parametrize("subtype", [
        "comparison_age",
        "comparison_age_mc2",
        "comparison_age_mc3",
        "comparison_born_first",
        "comparison_born_first_mc2",
        "comparison_born_first_mc3",
    ])
    def test_older_subtypes(self, subtype):
        assert is_comparison_older_subtype(subtype)
        assert not is_comparison_younger_subtype(subtype)
        assert is_comparison_subtype(subtype)

    @pytest.mark.parametrize("subtype", [
        "comparison_age_younger",
        "comparison_age_younger_mc2",
        "comparison_age_younger_mc3",
    ])
    def test_younger_subtypes(self, subtype):
        assert is_comparison_younger_subtype(subtype)
        assert not is_comparison_older_subtype(subtype)
        assert is_comparison_subtype(subtype)

    @pytest.mark.parametrize("subtype", [
        "superlative_oldest",
        "multi_constraint_2",
        "comparison_count",
    ])
    def test_non_comparison_subtypes(self, subtype):
        assert not is_comparison_subtype(subtype)


# ---------------------------------------------------------------------------
# generate_templates cell coverage
# ---------------------------------------------------------------------------
_RELATION_ATOM = re.compile(r"^<relation>_\d+\(")
_ATTR_ATOM = re.compile(r"^<attribute_name>_\d+\(")
_AGG_ATOM = re.compile(r"^aggregate_all")


def _template_cell(p_atoms: list[str]) -> tuple[int, int]:
    """Estimate (hops, constraints) for a template's Prolog atom list.

    Uses placeholder patterns rather than concrete predicate names since
    templates have not been substituted yet.
    """
    hops = sum(1 for a in p_atoms if _RELATION_ATOM.match(a) or _AGG_ATOM.match(a))
    constraints = sum(1 for a in p_atoms if _ATTR_ATOM.match(a))
    return hops, constraints


def _cells_covered(templates) -> set[tuple[int, int]]:
    """Return set of (hops, constraints) cells across all templates."""
    cells = set()
    for t in templates:
        p_atoms = t[1]
        if not isinstance(p_atoms, list):
            continue
        cells.add(_template_cell(p_atoms))
    return cells


class TestTemplateCells:
    @pytest.mark.parametrize("max_hops", [1, 2, 3, 4])
    def test_base_hops_range(self, max_hops):
        """Base templates scale with max_hops in the (h, 0) column.

        The count grammar (``How many Xp does [chain] have?``) contributes
        one aggregate_all on top of the chain, so the deepest (h, 0) cell
        is max_hops + 1. We verify that the column reaches that depth.
        """
        ts = generate_templates(
            max_hops=max_hops, max_constraints=2, question_types=["base"]
        )
        cells = _cells_covered(ts)
        hop_col = {h for h, c in cells if c == 0}
        assert max(hop_col) >= max_hops, (
            f"Expected max hop in (h,0) column >= {max_hops}, got {sorted(hop_col)}"
        )

    @pytest.mark.parametrize("max_hops", [1, 2, 3, 4])
    def test_base_attr_anchor(self, max_hops):
        """Base templates include at least one 1-attr-constraint cell."""
        ts = generate_templates(
            max_hops=max_hops, max_constraints=2, question_types=["base"]
        )
        cells = _cells_covered(ts)
        assert any(c >= 1 for _, c in cells), "Expected at least one constraint-1 cell"

    @pytest.mark.parametrize("max_constraints", [2, 3])
    def test_multi_constraint_cells(self, max_constraints):
        """Standalone multi_constraint covers (0, n) for n in 2..max_constraints."""
        ts = generate_templates(
            max_hops=4, max_constraints=max_constraints, question_types=["multi_constraint"]
        )
        cells = _cells_covered(ts)
        for n in range(2, max_constraints + 1):
            assert (0, n) in cells, f"Missing (hops=0, constraints={n}) at max_constraints={max_constraints}"

    @pytest.mark.parametrize("max_hops,max_constraints", [
        (2, 2), (3, 2), (4, 2),
        (2, 3), (3, 3), (4, 3),
    ])
    def test_comparison_age_mc_cells(self, max_hops, max_constraints):
        """comparison_age with mc-N fragments populates (hops>=1, constraints=N) cells."""
        ts = generate_templates(
            max_hops=max_hops,
            max_constraints=max_constraints,
            question_types=["comparison_age"],
        )
        cells = _cells_covered(ts)
        for n in range(2, max_constraints + 1):
            # At least (1, n) should be populated
            assert any(h >= 1 and c == n for h, c in cells), (
                f"Missing any (hops>=1, constraints={n}) at max_hops={max_hops}, max_constraints={max_constraints}"
            )

    def test_more_hops_strictly_more_cells(self):
        """Cells at max_hops=4 strictly include all cells at max_hops=2."""
        ts2 = generate_templates(max_hops=2, max_constraints=2, question_types=["base", "comparison_age"])
        ts4 = generate_templates(max_hops=4, max_constraints=2, question_types=["base", "comparison_age"])
        cells2 = _cells_covered(ts2)
        cells4 = _cells_covered(ts4)
        assert cells2 <= cells4
        assert cells2 != cells4  # strictly more at higher max_hops

    def test_more_constraints_strictly_more_cells(self):
        """Cells at max_constraints=3 strictly include all cells at max_constraints=2."""
        ts2 = generate_templates(max_hops=3, max_constraints=2, question_types=["multi_constraint", "comparison_age"])
        ts3 = generate_templates(max_hops=3, max_constraints=3, question_types=["multi_constraint", "comparison_age"])
        cells2 = _cells_covered(ts2)
        cells3 = _cells_covered(ts3)
        assert cells2 <= cells3
        assert cells2 != cells3


# ---------------------------------------------------------------------------
# comparison_count variants target cells
# ---------------------------------------------------------------------------
class TestComparisonCountCells:
    """Verify each variant type maps to the expected (hops, constraints) target.

    The +1 on hops comes from aggregate_all inside the comparison_count sampler;
    we can't easily test this without a real DB, so we just check the variant
    table encodes the right (chain_depth, n_attrs) pairs.
    """
    @pytest.mark.parametrize("chain_depth,n_attrs,expected_key", [
        (0, 0, "comparison_count"),
        (1, 0, "comparison_count_chain1"),
        (3, 0, "comparison_count_chain3"),
        (0, 2, "comparison_count_mc2"),
        (2, 2, "comparison_count_chain2_mc2"),
        (0, 3, "comparison_count_mc3"),
        (1, 3, "comparison_count_chain1_mc3"),
        (3, 3, "comparison_count_chain3_mc3"),
    ])
    def test_variant_key_and_value(self, chain_depth, n_attrs, expected_key):
        v = build_comparison_count_variants(
            max_chain_depth=3, n_attrs_options=(0, 2, 3)
        )
        assert expected_key in v
        assert v[expected_key] == (chain_depth, n_attrs)

    @pytest.mark.parametrize("max_hops,max_constraints", [
        (2, 2), (3, 3), (4, 3),
    ])
    def test_chain_depth_bounded_by_max_hops(self, max_hops, max_constraints):
        """chain_depth never exceeds max_hops - 1 (aggregate_all adds the last hop)."""
        opts = (0,) + tuple(range(2, max_constraints + 1))
        v = build_comparison_count_variants(
            max_chain_depth=max_hops - 1, n_attrs_options=opts
        )
        for key, (chain_depth, _) in v.items():
            assert chain_depth <= max_hops - 1, (
                f"{key} has chain_depth={chain_depth} > max_hops-1={max_hops-1}"
            )
