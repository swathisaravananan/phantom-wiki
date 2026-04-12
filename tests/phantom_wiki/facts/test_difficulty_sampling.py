"""Tests for difficulty-level-based sampling (sample_questions / describe_pool)."""

import pytest

from phantom_wiki.facts.balanced_sampling import (
    DIFFICULTY_LEVELS,
    _classify_broad_type,
    _get_difficulty_level,
    _matches_step_range,
    describe_pool,
    sample_questions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_questions(
    difficulties: list[int],
    categories: list[str] | None = None,
) -> list[dict]:
    """Create question dicts with given difficulties and optional categories."""
    if categories is None:
        categories = ["base"] * len(difficulties)
    return [
        {
            "id": str(i),
            "question": f"Q{i}",
            "difficulty": d,
            "question_category": categories[i],
        }
        for i, d in enumerate(difficulties)
    ]


# ---------------------------------------------------------------------------
# DIFFICULTY_LEVELS config
# ---------------------------------------------------------------------------
class TestDifficultyLevels:
    def test_levels_are_non_overlapping(self):
        """Ranges must not overlap."""
        ranges = []
        for level, (lo, hi) in DIFFICULTY_LEVELS.items():
            if hi is None:
                hi = 999
            ranges.append((lo, hi, level))
        ranges.sort()
        for i in range(len(ranges) - 1):
            assert ranges[i][1] < ranges[i + 1][0], (
                f"Overlap between {ranges[i][2]} and {ranges[i+1][2]}"
            )

    def test_levels_cover_expected_keys(self):
        assert set(DIFFICULTY_LEVELS.keys()) == {
            "trivial", "easy", "medium", "hard", "extreme",
        }

    def test_trivial_range(self):
        assert DIFFICULTY_LEVELS["trivial"] == (1, 2)

    def test_easy_range(self):
        assert DIFFICULTY_LEVELS["easy"] == (3, 4)

    def test_medium_range(self):
        assert DIFFICULTY_LEVELS["medium"] == (5, 7)

    def test_hard_range(self):
        assert DIFFICULTY_LEVELS["hard"] == (8, 11)

    def test_extreme_range(self):
        assert DIFFICULTY_LEVELS["extreme"] == (12, None)


# ---------------------------------------------------------------------------
# _get_difficulty_level
# ---------------------------------------------------------------------------
class TestGetDifficultyLevel:
    def test_trivial(self):
        assert _get_difficulty_level(1) == "trivial"
        assert _get_difficulty_level(2) == "trivial"

    def test_easy(self):
        assert _get_difficulty_level(3) == "easy"
        assert _get_difficulty_level(4) == "easy"

    def test_medium(self):
        assert _get_difficulty_level(5) == "medium"
        assert _get_difficulty_level(7) == "medium"

    def test_hard(self):
        assert _get_difficulty_level(8) == "hard"
        assert _get_difficulty_level(11) == "hard"

    def test_extreme(self):
        assert _get_difficulty_level(12) == "extreme"
        assert _get_difficulty_level(100) == "extreme"


# ---------------------------------------------------------------------------
# _matches_step_range
# ---------------------------------------------------------------------------
class TestMatchesStepRange:
    def test_no_bounds(self):
        assert _matches_step_range(5, None, None) is True

    def test_min_only(self):
        assert _matches_step_range(5, 3, None) is True
        assert _matches_step_range(2, 3, None) is False

    def test_max_only(self):
        assert _matches_step_range(5, None, 7) is True
        assert _matches_step_range(8, None, 7) is False

    def test_both_bounds(self):
        assert _matches_step_range(5, 3, 7) is True
        assert _matches_step_range(2, 3, 7) is False
        assert _matches_step_range(8, 3, 7) is False

    def test_boundary_inclusive(self):
        assert _matches_step_range(3, 3, 7) is True
        assert _matches_step_range(7, 3, 7) is True


# ---------------------------------------------------------------------------
# _classify_broad_type
# ---------------------------------------------------------------------------
class TestClassifyBroadType:
    def test_comparison(self):
        assert _classify_broad_type("comparison_age") == "comparison"

    def test_multi_constraint(self):
        assert _classify_broad_type("multi_constraint_hobby") == "multi_constraint"

    def test_superlative(self):
        assert _classify_broad_type("superlative_oldest") == "superlative"

    def test_base(self):
        assert _classify_broad_type("base") == "base"
        assert _classify_broad_type("") == "base"


# ---------------------------------------------------------------------------
# sample_questions
# ---------------------------------------------------------------------------
class TestSampleQuestions:
    def test_basic_sample(self):
        questions = _make_questions([1, 2, 3, 4, 5, 6, 7, 8])
        result = sample_questions(questions, count=3)
        assert len(result) == 3

    def test_exact_count(self):
        questions = _make_questions([5, 6, 7])
        result = sample_questions(questions, count=3)
        assert len(result) == 3

    def test_adds_reasoning_steps_field(self):
        questions = _make_questions([5, 6, 7])
        result = sample_questions(questions, count=2)
        for q in result:
            assert "reasoning_steps" in q
            assert q["reasoning_steps"] == q["difficulty"]

    def test_filter_by_difficulty_level(self):
        # trivial=1-2, easy=3-4, medium=5-7
        questions = _make_questions([1, 2, 3, 4, 5, 6, 7])
        result = sample_questions(questions, count=3, difficulty="medium")
        assert len(result) == 3
        assert all(5 <= q["difficulty"] <= 7 for q in result)

    def test_filter_by_min_max_steps(self):
        questions = _make_questions([1, 2, 3, 4, 5, 6, 7, 8])
        result = sample_questions(questions, count=2, min_steps=3, max_steps=5)
        assert len(result) == 2
        assert all(3 <= q["difficulty"] <= 5 for q in result)

    def test_mutual_exclusion_difficulty_and_steps(self):
        questions = _make_questions([1, 2, 3])
        with pytest.raises(ValueError, match="Cannot specify both"):
            sample_questions(questions, count=1, difficulty="easy", min_steps=1)

    def test_mutual_exclusion_difficulty_and_max_steps(self):
        questions = _make_questions([1, 2, 3])
        with pytest.raises(ValueError, match="Cannot specify both"):
            sample_questions(questions, count=1, difficulty="easy", max_steps=5)

    def test_insufficient_pool_raises(self):
        questions = _make_questions([1, 2])
        with pytest.raises(ValueError, match="Requested 5 questions but only 2"):
            sample_questions(questions, count=5)

    def test_insufficient_pool_with_filters(self):
        questions = _make_questions([1, 2, 5, 6])
        with pytest.raises(ValueError, match="only 2 match"):
            sample_questions(questions, count=3, difficulty="medium")

    def test_filter_by_question_types_insufficient(self):
        categories = ["base", "comparison_age", "superlative_oldest", "base"]
        questions = _make_questions([1, 2, 3, 4], categories=categories)
        with pytest.raises(ValueError, match="only 1 match"):
            sample_questions(questions, count=2, question_types=["comparison"])

    def test_filter_by_question_types_exact(self):
        categories = ["base", "comparison_age", "comparison_born", "superlative_oldest"]
        questions = _make_questions([1, 2, 3, 4], categories=categories)
        result = sample_questions(questions, count=2, question_types=["comparison"])
        assert len(result) == 2
        assert all("comparison" in q["question_category"] for q in result)

    def test_filter_by_multiple_types(self):
        categories = ["base", "comparison_age", "superlative_oldest", "multi_constraint_hobby"]
        questions = _make_questions([1, 2, 3, 4], categories=categories)
        result = sample_questions(
            questions, count=2, question_types=["comparison", "superlative"]
        )
        assert len(result) == 2

    def test_reproducible_with_seed(self):
        questions = _make_questions(list(range(1, 21)))
        r1 = sample_questions(questions, count=5, seed=42)
        r2 = sample_questions(questions, count=5, seed=42)
        assert [q["id"] for q in r1] == [q["id"] for q in r2]

    def test_different_seed_different_result(self):
        questions = _make_questions(list(range(1, 21)))
        r1 = sample_questions(questions, count=5, seed=42)
        r2 = sample_questions(questions, count=5, seed=99)
        # Very unlikely to be the same
        assert [q["id"] for q in r1] != [q["id"] for q in r2]

    def test_unknown_difficulty_level_raises(self):
        questions = _make_questions([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown difficulty level"):
            sample_questions(questions, count=1, difficulty="impossible")

    def test_combined_difficulty_and_type_filter(self):
        categories = ["base", "comparison_age", "base", "comparison_born"]
        questions = _make_questions([1, 5, 6, 7], categories=categories)
        result = sample_questions(
            questions, count=1, difficulty="medium", question_types=["comparison"]
        )
        assert len(result) == 1
        assert 5 <= result[0]["difficulty"] <= 7
        assert "comparison" in result[0]["question_category"]


# ---------------------------------------------------------------------------
# describe_pool
# ---------------------------------------------------------------------------
class TestDescribePool:
    def test_basic_structure(self):
        questions = _make_questions([1, 5, 12])
        result = describe_pool(questions)
        assert "total" in result
        assert "by_difficulty" in result
        assert "by_type" in result
        assert "by_difficulty_and_type" in result

    def test_total_count(self):
        questions = _make_questions([1, 2, 3, 4, 5])
        result = describe_pool(questions)
        assert result["total"] == 5

    def test_by_difficulty_counts(self):
        questions = _make_questions([1, 2, 5, 6, 12])
        result = describe_pool(questions)
        assert result["by_difficulty"]["trivial"] == 2
        assert result["by_difficulty"]["medium"] == 2
        assert result["by_difficulty"]["extreme"] == 1
        assert result["by_difficulty"]["easy"] == 0
        assert result["by_difficulty"]["hard"] == 0

    def test_by_type_counts(self):
        categories = ["base", "comparison_age", "comparison_born", "superlative_oldest"]
        questions = _make_questions([1, 2, 3, 4], categories=categories)
        result = describe_pool(questions)
        assert result["by_type"]["base"] == 1
        assert result["by_type"]["comparison"] == 2
        assert result["by_type"]["superlative"] == 1

    def test_by_difficulty_and_type(self):
        categories = ["base", "comparison_age"]
        questions = _make_questions([1, 5], categories=categories)
        result = describe_pool(questions)
        assert result["by_difficulty_and_type"]["trivial"]["base"] == 1
        assert result["by_difficulty_and_type"]["medium"]["comparison"] == 1

    def test_empty_pool(self):
        result = describe_pool([])
        assert result["total"] == 0
        assert all(v == 0 for v in result["by_difficulty"].values())
        assert result["by_type"] == {}

    def test_all_difficulty_levels_present(self):
        result = describe_pool([])
        assert set(result["by_difficulty"].keys()) == set(DIFFICULTY_LEVELS.keys())
        assert set(result["by_difficulty_and_type"].keys()) == set(DIFFICULTY_LEVELS.keys())
