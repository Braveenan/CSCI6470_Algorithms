"""
Author: Braveenan Sritharan
Date Created: 2025-04-06

"""

import os
import random
import itertools
import sys


def ensure_config_dir():
    """
    Ensures the 'config' directory exists. Creates it if missing.
    """
    os.makedirs("config", exist_ok=True)


def generate_config1_file():
    """
    Generates config1.txt with random scores (1–5) for all label pairs.
    Includes a randomly chosen delta score (-3 to -1) at the end.
    """
    ensure_config_dir()
    labels = ['A', 'C', 'G', 'T']
    score_range = [1, 2, 3, 4, 5]
    combinations = itertools.product(labels, repeat=2)

    with open("config/config1.txt", "w") as f:
        for a, b in combinations:
            score = random.choice(score_range)
            f.write(f"{a}\t{b}\t{score}\n")

        delta_score = random.randint(-3, -1)
        f.write(f"delta\t{delta_score}\n")


def generate_config2_file():
    """
    Generates config2.txt where:
    - Identical label pairs have one fixed score.
    - Different label pairs have another fixed score.
    Includes a delta score (-3 to -1).
    """
    ensure_config_dir()
    labels = ['A', 'C', 'G', 'T']
    combinations = itertools.product(labels, repeat=2)

    same_score = random.randint(1, 5)
    diff_score = random.randint(1, 5)

    with open("config/config2.txt", "w") as f:
        for a, b in combinations:
            score = same_score if a == b else diff_score
            f.write(f"{a}\t{b}\t{score}\n")

        delta_score = random.randint(-3, -1)
        f.write(f"delta\t{delta_score}\n")


def generate_config3_file():
    """
    Generates config3.txt with symmetric random scores for label pairs.
    Ensures (A, C) and (C, A) get the same score.
    Includes a delta score (-3 to -1).
    """
    ensure_config_dir()
    labels = ['A', 'C', 'G', 'T']
    score_range = [1, 2, 3, 4, 5]
    base_combinations = itertools.combinations_with_replacement(labels, 2)

    score_map = {}
    for a, b in base_combinations:
        score = random.choice(score_range)
        score_map[(a, b)] = score
        score_map[(b, a)] = score

    combinations = itertools.product(labels, repeat=2)

    with open("config/config3.txt", "w") as f:
        for a, b in combinations:
            score = score_map[(a, b)]
            f.write(f"{a}\t{b}\t{score}\n")

        delta_score = random.randint(-3, -1)
        f.write(f"delta\t{delta_score}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3"}:
        print("Usage: python file.py [1|2|3]")
    elif sys.argv[1] == "1":
        generate_config1_file()
        print("config/config1.txt generated.")
    elif sys.argv[1] == "2":
        generate_config2_file()
        print("config/config2.txt generated.")
    else:
        generate_config3_file()
        print("config/config3.txt generated.")
