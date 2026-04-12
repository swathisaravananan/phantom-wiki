import copy
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .core.article import get_articles
from .facts import get_database
from .facts.attributes import db_generate_attributes
from .facts.balanced_sampling import balanced_sample, filter_by_difficulty, sample_questions
from .facts.balanced_sampling import describe_pool as _describe_pool
from .facts.extended_questions import (
    EXTENDED_QUESTION_TYPES,
    get_extended_answer,
    is_extended_question,
    sample_extended_question,
)
from .facts.family import db_generate_family
from .facts.friends import db_generate_friendships
from .facts.question_difficulty import calculate_query_difficulty
from .facts.sample import sample_question
from .facts.templates import (
    ALL_QUESTION_TYPES,
    COMPARISON_AGE_SUBTYPE,
    COMPARISON_AGE_YOUNGER_SUBTYPE,
    COMPARISON_BORN_FIRST_SUBTYPE,
    COMPARISON_COUNT_FEWER_SUBTYPE,
    COMPARISON_COUNT_MORE_SUBTYPE,
    QUESTION_TYPE_BASE,
    classify_question_type,
    generate_templates,
    is_aggregation_question,
)
from .utils import blue, generate_unique_id
from .utils.get_answer import get_answer


def _get_extended_cfg_answer(
    question: str,
    query: list[str],
    answer_info,
    question_subtype: str,
    db,
) -> list[str]:
    """Extract answers for extended CFG question types.

    For comparison questions: checks if the Prolog query succeeds to determine the winner.
    For superlative/multi-constraint: runs the Prolog query and extracts the answer variable.
    """
    from .utils import decode

    joined = ", ".join(reversed(query))

    # Comparison types: query includes a comparison predicate.
    # If query succeeds, left operand wins. If fails, right operand wins.
    if question_subtype in (
        COMPARISON_AGE_SUBTYPE,
        COMPARISON_BORN_FIRST_SUBTYPE,
        COMPARISON_COUNT_MORE_SUBTYPE,
    ):
        results = list(db.prolog.query(joined))
        left_var, right_var = answer_info
        if results:
            # Left operand wins — extract its name
            if left_var in results[0]:
                return [str(decode(results[0][left_var]))]
            # left_var is a <name> placeholder already resolved in the query
            import re

            m = re.match(r'Who is older, (.+?) or', question) or re.match(
                r'Who was born first, (.+?) or', question
            ) or re.match(r'Who has more .+?, (.+?) or', question)
            return [m.group(1)] if m else []
        else:
            # Right operand wins
            # Try to get the right person's name from a query without the comparison
            no_cmp_query = ", ".join(reversed([q for q in query if "@<" not in q and ">" not in q]))
            results2 = list(db.prolog.query(no_cmp_query))
            if results2 and right_var in results2[0]:
                return [str(decode(results2[0][right_var]))]
            m = re.match(r'.+ or (.+?)\?', question)
            return [m.group(1)] if m else []

    elif question_subtype in (COMPARISON_AGE_YOUNGER_SUBTYPE, COMPARISON_COUNT_FEWER_SUBTYPE):
        # "younger" / "fewer" — query has @< / >, so if it succeeds, left has EARLIER dob / MORE count.
        # For "younger": left born earlier means left is OLDER, so right is younger (the answer).
        # For "fewer": left has more, so right has fewer (the answer).
        results = list(db.prolog.query(joined))
        left_var, right_var = answer_info
        if results:
            # Left won the comparison (older / more) → right is the answer (younger / fewer)
            no_cmp_query = ", ".join(reversed([q for q in query if "@<" not in q and ">" not in q]))
            results2 = list(db.prolog.query(no_cmp_query))
            if results2 and right_var in results2[0]:
                return [str(decode(results2[0][right_var]))]
            import re

            m = re.match(r'.+ or (.+?)\?', question)
            return [m.group(1)] if m else []
        else:
            # Left did NOT win → left is younger / fewer → left is the answer
            no_cmp_query = ", ".join(reversed([q for q in query if "@<" not in q and ">" not in q]))
            results2 = list(db.prolog.query(no_cmp_query))
            if results2 and left_var in results2[0]:
                return [str(decode(results2[0][left_var]))]
            import re

            m = re.match(r'Who is younger, (.+?) or', question) or re.match(
                r'Who has fewer .+?, (.+?) or', question
            )
            return [m.group(1)] if m else []

    else:
        # Superlative and multi-constraint: answer_info is a variable name like 'Y_5'
        results = list(db.prolog.query(joined))
        if not results:
            return []
        return sorted({str(decode(r[answer_info])) for r in results if answer_info in r})


def _filter_by_hops(questions: list[dict], min_hops: int = None, max_hops: int = None) -> list[dict]:
    """Filter questions by hop count (number of Prolog predicates in the query)."""
    filtered = []
    for q in questions:
        prolog_query = q.get("prolog", {}).get("query", [])
        if isinstance(prolog_query, list):
            # Count hops as number of chain predicates (exclude static predicates)
            hops = sum(
                1
                for stmt in prolog_query
                if not stmt.strip().startswith("\\+") and "dob(" not in stmt and "@<" not in stmt and "@>" not in stmt
            )
        else:
            hops = 1
        if min_hops is not None and hops < min_hops:
            continue
        if max_hops is not None and hops > max_hops:
            continue
        filtered.append(q)
    return filtered


def generate_dataset(
    max_branching_factor: int = 5,
    max_family_tree_depth: int = 5,
    max_family_tree_size: int = 25,
    num_family_trees: int = 1,
    stop_prob: int = 0,
    duplicate_names: bool = False,
    friendship_k: int = 3,
    friendship_seed: int = 1,
    num_questions_per_type: int = 10,
    num_sampling_attempts: int = 100,
    question_depth: int = 6,
    easy_mode: bool = False,
    skip_solution_traces: bool = False,
    debug: bool = False,
    quiet: bool = False,
    visualize: bool = False,
    use_multithreading: bool = False,
    seed: int = 1,
    output_dir: str = "./out",
    article_format: str = "txt",
    question_format: str = "json_by_type",
    balanced: bool = False,
    min_difficulty: int = None,
    max_difficulty: int = None,
    question_types: str = None,
    min_hops: int = None,
    max_hops: int = None,
    sample_count: int = None,
    sample_difficulty: str = None,
    sample_min_steps: int = None,
    sample_max_steps: int = None,
    sample_types: str = None,
    describe_pool: bool = False,
) -> None:
    """
    Generate a PhantomWiki dataset consisting of family trees, friendship networks,
    articles, and reasoning questions with answers.

    Args:
        max_branching_factor (int): The maximum number of children that any person
            in a family tree may have. (default=5)
        max_family_tree_depth (int): The maximum depth that a family tree may have.
            (default=5)
        max_family_tree_size (int): The maximum number of people that may appear
            in a family tree. (default=25)
        num_family_trees (int): The number of family trees to generate. (default=1)
        stop_prob (float): The probability of stopping to further extend a family tree
            after a person has been added. (default=0)
        duplicate_names (bool): Allow/prevent duplicate names in the generation.
            (default=False)
        friendship_k (int): Average degree in friendship graph. (default=3)
        friendship_seed (int): Seed for friendship generation. (default=1)
        num_questions_per_type (int): Number of questions to generate per question type
            (i.e., template). (default=10)
        num_sampling_attempts (int): Number of attempts to sample a valid question.
            (default=100)
        question_depth (int): Depth of the question template. (default=6)
        easy_mode (bool): Sample from easy relations (hard mode is default).
            (default=False)
        skip_solution_traces (bool): Do not include solution traces in the dataset.
            (default=False)
        debug (bool): Enable debug output (DEBUG level). (default=False)
        quiet (bool): Enable quiet (no) output (WARNING level). (default=False)
        visualize (bool): Whether or not to visualize the friendship & family graphs.
            (default=False)
        use_multithreading (bool): Use multithreading for querying the database when
            generating questions/answers. Note: This flag works for Windows and Linux,
            but not for MacOS. Also very intensive for high universe size. (default=False)
        seed (int): Global seed for random number generator. (default=1)
        output_dir (str): Path to the output folder. (default="./out")
        article_format (str): Format to save the generated articles. Options: 'txt', 'json'.
            (default="txt")
        question_format (str): Format to save the generated questions and answers.
            Options: 'json_by_type', 'json'. (default="json_by_type")

    Returns:
        None, The function saves all generated data to the output directory as well as a
        `timings.csv` and does not return any value.
    """
    assert article_format in ["txt", "json"], "Article format not supported, use 'txt' or 'json'."
    assert question_format in [
        "json_by_type",
        "json",
    ], "Question format not supported, use 'json_by_type' or 'json'."

    if quiet:
        log_level = logging.WARNING
    elif debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler()])

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    # create dictionary to store timings
    timings = {}
    global_start = time.time()

    #
    # Step 1. Generate facts
    #
    db = get_database()

    blue("Generating facts")
    start = time.time()
    # generate family tree
    db_generate_family(
        db,
        seed,
        duplicate_names,
        debug,
        output_dir,
        visualize,
        max_family_tree_depth,
        max_branching_factor,
        max_family_tree_size,
        stop_prob,
        num_family_trees,
    )

    # generate friend relationships between people in the database
    db_generate_friendships(db, friendship_k, friendship_seed, visualize, output_dir)

    # generate jobs, hobbies for each person in the database
    db_generate_attributes(db, seed)

    timings["facts_generate"] = time.time() - start

    db_path = os.path.join(output_dir, "facts.pl")
    blue(f"Saving Prolog database to {db_path}")
    facts_time = time.time()
    db.save_to_disk(db_path)
    timings["facts_save"] = time.time() - facts_time

    #
    # Step 2. Generate articles
    # Currently, the articles comprise a list of facts.
    #
    blue("Generating articles")
    start = time.time()
    articles = get_articles(db, db.get_person_names())
    timings["articles_generate"] = time.time() - start

    blue("Saving articles")
    start = time.time()
    if article_format == "txt":
        article_dir = os.path.join(output_dir, "articles")
        logging.info(f"Saving articles to: {article_dir}")
        os.makedirs(article_dir, exist_ok=True)
        for name, (article, facts) in articles.items():
            with open(os.path.join(article_dir, f"{name}.txt"), "w") as file:
                file.write(article)
            with open(os.path.join(article_dir, f"{name}_facts.txt"), "w") as file:
                file.write("\n".join(facts))
    elif article_format == "json":
        save_path = os.path.join(output_dir, "articles.json")
        logging.info(f"Saving articles to: {save_path}")
        with open(save_path, "w") as file:
            json.dump(
                [
                    {"title": name, "article": article, "facts": facts}
                    for name, (article, facts) in articles.items()
                ],
                file,
                indent=4,
            )
    else:
        raise ValueError(f"Article format {article_format} not supported!")
    timings["articles_save"] = time.time() - start

    #
    # Step 3. Generate question-answer pairs
    #
    blue("Generating question answer pairs")
    start = time.time()

    # Parse question types
    if question_types is not None:
        parsed_qtypes = [t.strip() for t in question_types.split(",")]
    else:
        parsed_qtypes = list(ALL_QUESTION_TYPES)

    # generate question templates with a given depth
    templates = generate_templates(depth=question_depth, question_types=parsed_qtypes)

    # Separate base templates (3-tuples) from extended templates (4-tuples)
    base_templates = [t for t in templates if len(t) == 3]
    extended_cfg_templates = [t for t in templates if len(t) == 4]

    # sample questions for each template (i.e., type)
    if question_format == "json_by_type":
        question_dir = os.path.join(output_dir, "questions")
        logging.info(f"Saving questions to: {question_dir}")
        os.makedirs(question_dir, exist_ok=True)

    progbar = tqdm(enumerate(base_templates), desc="Generating questions", total=len(base_templates))

    # Populate person name bank for the universe. The list is static across generating questions
    # so create it once and pass it to the question generation function
    person_name_bank: list[str] = db.get_person_names()

    # Create caches for person -> (attr name, attr value) and person -> (relation, related person) pairs
    # When we iterate over multiple questions, we can reuse the same cache to avoid recomputing
    # e.g. "John" -> [("dob", "1990-01-01"), ("job", "teacher"), ("hobby", "reading"),
    # ("hobby", "swimming"), ...]
    # NOTE: Invariant: (attr name, attr value) pairs are unique
    person_name2attr_name_and_val: dict[str, list[tuple[str, str]]] = {}
    # e.g. "John" -> [("child", "Alice"), ("child", "Bob"), ("friend", "Charlie"), ...]
    # NOTE: Invariant: (relation, related person) pairs are unique
    person_name2relation_and_related: dict[str, list[tuple[str, str]]] = {}

    # To store all the questions and queries for all templates
    all_questions = []
    all_queries = []

    for i, (question_template, query_template, answer) in progbar:
        # Reset the seed at the start of each question type
        # so that sampled questions are the same for each question type
        rng = np.random.default_rng(seed)

        # To store the questions and queries for the given template
        questions = []
        queries = []

        # for _ in range(args.num_questions_per_type):
        while (
            len(questions)
            < num_questions_per_type
            # TODO: handle potential edge cases where templates repeatedly fail to generate,
            # resulting in an infinite loop
        ):  # TODO: temporary fix to make sure that we generate the same number of questions for each template
            # sample a question
            question, query = sample_question(
                question_template,
                query_template,
                rng,
                db,
                person_name_bank,
                person_name2attr_name_and_val,
                person_name2relation_and_related,
                easy_mode=easy_mode,
                num_sampling_attempts=num_sampling_attempts,
            )

            questions.append(question)
            queries.append(query)

        all_questions.append(questions)
        all_queries.append(queries)

    # Generate extended CFG-based question types (comparison, multi-constraint, superlative)
    extended_cfg_questions_data = []
    if extended_cfg_templates:
        blue("Generating extended CFG question types")
        for tmpl_idx, tmpl in enumerate(extended_cfg_templates):
            question_template, query_template, answer_info, question_subtype = tmpl
            rng = np.random.default_rng(seed)
            questions = []
            queries = []
            attempts = 0
            max_attempts = num_questions_per_type * num_sampling_attempts

            while len(questions) < num_questions_per_type and attempts < max_attempts:
                attempts += 1
                try:
                    result = sample_question(
                        question_template,
                        query_template,
                        rng,
                        db,
                        person_name_bank,
                        person_name2attr_name_and_val,
                        person_name2relation_and_related,
                        easy_mode=easy_mode,
                        num_sampling_attempts=1,
                    )
                    if result is not None:
                        question, query = result
                        questions.append(question)
                        queries.append(query)
                except (ValueError, AssertionError):
                    continue

            # Get answers for extended CFG questions
            for j in range(len(questions)):
                answer_list = _get_extended_cfg_answer(
                    questions[j], queries[j], answer_info, question_subtype, db
                )
                question_difficulty = calculate_query_difficulty(queries[j])
                q_type = classify_question_type(question_template)
                extended_cfg_questions_data.append(
                    {
                        "id": generate_unique_id(),
                        "question": questions[j],
                        "solution_traces": json.dumps([]),
                        "answer": answer_list,
                        "prolog": {"query": queries[j], "answer": str(answer_info)},
                        "template": question_template,
                        "type": len(base_templates) + tmpl_idx,
                        "difficulty": question_difficulty,
                        "is_aggregation_question": False,
                        "question_category": question_subtype,
                    }
                )

    # Generate legacy extended question types (backward compat, standalone 1-hop)
    extended_questions_data = []
    if QUESTION_TYPE_BASE in parsed_qtypes:
        # Only generate legacy extended questions when base types are included
        blue("Generating extended question types")
        for qtype in EXTENDED_QUESTION_TYPES:
            rng = np.random.default_rng(seed)
            questions = []
            queries = []
            attempts = 0
            max_attempts = num_questions_per_type * num_sampling_attempts
            while len(questions) < num_questions_per_type and attempts < max_attempts:
                attempts += 1
                result = sample_extended_question(
                    qtype,
                    rng,
                    db,
                    person_name_bank,
                    person_name2attr_name_and_val,
                    person_name2relation_and_related,
                    easy_mode=easy_mode,
                    num_sampling_attempts=num_sampling_attempts,
                )
                if result is not None:
                    question, query = result
                    questions.append(question)
                    queries.append(query)

            # Get answers for extended questions
            for j in range(len(questions)):
                answer_list = get_extended_answer(questions[j], queries[j], qtype, db)
                question_difficulty = calculate_query_difficulty(queries[j])
                extended_questions_data.append(
                    {
                        "id": generate_unique_id(),
                        "question": questions[j],
                        "solution_traces": json.dumps([]),
                        "answer": answer_list,
                        "prolog": {"query": queries[j], "answer": "X"},
                        "template": [qtype],
                        "type": len(base_templates) + len(extended_cfg_templates) + EXTENDED_QUESTION_TYPES.index(qtype),
                        "difficulty": question_difficulty,
                        "is_aggregation_question": False,
                        "question_category": qtype,
                    }
                )

    # Get all possible answers/solution traces for the base queries
    answers = [t[2] for t in base_templates]
    all_solution_traces, all_final_results = get_answer(
        copy.deepcopy(all_queries),
        db,
        answers,
        skip_solution_traces=skip_solution_traces,
        multi_threading=use_multithreading,
    )

    all_full_questions = []
    progbar = tqdm(enumerate(base_templates), desc="Generating questions #2", total=len(base_templates))

    for i, (question_template, query_template, answer) in progbar:
        questions = []

        for j in range(num_questions_per_type):
            # get the difficulty of the question
            question = all_questions[i][j]
            query = all_queries[i][j]
            question_difficulty = calculate_query_difficulty(query)

            questions.append(
                {
                    "id": generate_unique_id(),
                    "question": question,
                    "solution_traces": json.dumps(
                        all_solution_traces[i][j]
                    ),  # NOTE: serialize list of dicts so that it can be saved on HF
                    "answer": all_final_results[i][j],
                    "prolog": {"query": query, "answer": answer},
                    "template": question_template,
                    "type": i,  # this references the template type
                    "difficulty": question_difficulty,
                    "is_aggregation_question": is_aggregation_question(question),
                }
            )
            if question_format == "json_by_type":
                with open(os.path.join(question_dir, f"type{i}.json"), "w") as file:
                    json.dump(questions, file, indent=4)

        all_full_questions.extend(questions)

        # update progbar
        progbar.set_description(f"Template ({i+1}/{len(base_templates)})")

    # Append extended CFG questions
    all_full_questions.extend(extended_cfg_questions_data)

    # Append legacy extended questions
    all_full_questions.extend(extended_questions_data)
    if question_format == "json_by_type" and extended_questions_data:
        for qtype in EXTENDED_QUESTION_TYPES:
            type_qs = [q for q in extended_questions_data if q.get("question_category") == qtype]
            if type_qs:
                type_idx = len(base_templates) + len(extended_cfg_templates) + EXTENDED_QUESTION_TYPES.index(qtype)
                with open(os.path.join(question_dir, f"type{type_idx}.json"), "w") as file:
                    json.dump(type_qs, file, indent=4)

    # Apply hop-count filtering
    if min_hops is not None or max_hops is not None:
        blue("Filtering questions by hop count")
        all_full_questions = _filter_by_hops(all_full_questions, min_hops, max_hops)
        logging.info(f"After hop filtering: {len(all_full_questions)} questions")

    # Apply difficulty-based filtering and balanced sampling
    if min_difficulty is not None or max_difficulty is not None:
        blue("Filtering questions by difficulty range")
        all_full_questions = filter_by_difficulty(all_full_questions, min_difficulty, max_difficulty)
        logging.info(f"After filtering: {len(all_full_questions)} questions")

    if balanced:
        blue("Applying difficulty-balanced sampling")
        all_full_questions = balanced_sample(all_full_questions)
        logging.info(f"After balanced sampling: {len(all_full_questions)} questions")

    # Describe pool (print breakdown and exit)
    if describe_pool:
        pool_info = _describe_pool(all_full_questions)
        blue("Question Pool Breakdown")
        logging.info(f"Total questions: {pool_info['total']}")
        logging.info("")
        logging.info("By difficulty level:")
        for level, count in pool_info["by_difficulty"].items():
            logging.info(f"  {level:>10s}: {count}")
        logging.info("")
        logging.info("By question type:")
        for qtype, count in pool_info["by_type"].items():
            logging.info(f"  {qtype:>20s}: {count}")
        logging.info("")
        logging.info("By difficulty and type:")
        for level, type_counts in pool_info["by_difficulty_and_type"].items():
            if type_counts:
                logging.info(f"  {level}:")
                for qtype, count in type_counts.items():
                    logging.info(f"    {qtype:>20s}: {count}")
        return

    # Apply exact sampling (--sample-count)
    if sample_count is not None:
        blue("Sampling exact number of questions")
        parsed_sample_types = None
        if sample_types is not None:
            parsed_sample_types = [t.strip() for t in sample_types.split(",")]
        all_full_questions = sample_questions(
            all_full_questions,
            count=sample_count,
            difficulty=sample_difficulty,
            question_types=parsed_sample_types,
            min_steps=sample_min_steps,
            max_steps=sample_max_steps,
            seed=seed,
        )
        logging.info(f"After sampling: {len(all_full_questions)} questions")

    timings["questions_generate"] = time.time() - start

    blue("Saving questions")
    start = time.time()
    if question_format == "json":
        # save all questions to a single file
        save_path = os.path.join(output_dir, "questions.json")
        logging.info(f"Saving questions to: {save_path}")
        with open(save_path, "w") as file:
            json.dump(all_full_questions, file, indent=4)
    timings["questions_save"] = time.time() - start

    timings["total"] = time.time() - global_start

    logging.info("Benchmarking results:")
    df_timings = pd.DataFrame([timings])
    logging.info(df_timings.T.to_markdown())
    timings_path = os.path.join(output_dir, "timings.csv")
    logging.info(f"Saving timings to {timings_path}")
    df_timings.to_csv(timings_path, index=False)
    blue("Done!")
