"""
Title: Pairwise Alignment
Author: Braveenan Sritharan
Date Created: 2025-04-06

"""

import os

# -------------------------
# Auxiliary Functions
# -------------------------

def ensure_directories():
    """
    Ensures required folders exist: input/, config/, output/
    """
    os.makedirs("input", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)


def read_config_file(filepath):
    """
    Reads the scoring configuration file.
    Returns a dictionary with scores for character pairs and delta gap penalty.
    """
    score = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == "delta":
                score["delta"] = int(parts[1])
            else:
                a, b, val = parts
                score[(a, b)] = int(val)
    return score


def read_two_lines(filepath):
    """
    Reads the first two lines of a file and returns them as a tuple.
    Raises an error if fewer than two lines are present.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError("Input file must contain at least two lines.")
        return lines[0].strip(), lines[1].strip()


# -------------------------
# Step 1: Objective Function
# -------------------------

# F(i, j) = max(
#     F(i-1, j-1) + S(x[i], y[j]),  # match/mismatch
#     F(i-1, j)   + delta,          # gap in y (deletion)
#     F(i, j-1)   + delta           # gap in x (insertion)
# )


# -------------------------
# Step 2: Fill DP Table
# -------------------------

def fill_dp_table(x, y, score):
    """
    Fills the dynamic programming table based on the defined objective function.
    Returns the score table and the backtrace previous table.
    """
    m, n = len(x), len(y)

    score_table = [[0] * (n + 1) for _ in range(m + 1)]
    prev_table = [[(-1, -1)] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        score_table[i][0] = score_table[i-1][0] + score["delta"]
        prev_table[i][0] = (i-1, 0)

    for j in range(1, n + 1):
        score_table[0][j] = score_table[0][j-1] + score["delta"]
        prev_table[0][j] = (0, j-1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_table[i-1][j-1] + score[(x[i-1], y[j-1])]
            delete = score_table[i-1][j] + score["delta"]
            insert = score_table[i][j-1] + score["delta"]

            best = max(match, delete, insert)
            score_table[i][j] = best

            if best == match:
                prev_table[i][j] = (i-1, j-1)
            elif best == delete:
                prev_table[i][j] = (i-1, j)
            else:
                prev_table[i][j] = (i, j-1)

    return score_table, prev_table


# -------------------------
# Step 3: Traceback and Output
# -------------------------

def traceback_alignment(score_table, prev_table, x, y, output_path):
    """
    Traces back through the DP table to construct the optimal alignment.
    Writes aligned sequences, traceback path, final score, and tables to output_path.
    """
    i, j = len(x), len(y)
    path = []
    path.append((i, j))
    aligned_x = []
    aligned_y = []

    final_score = score_table[i][j]

    while i > 0 or j > 0:
        prev_i, prev_j = prev_table[i][j]

        if prev_i == i - 1 and prev_j == j - 1:
            aligned_x.append(x[prev_i])
            aligned_y.append(y[prev_j])
        elif prev_i == i - 1 and prev_j == j:
            aligned_x.append(x[prev_i])
            aligned_y.append('-')
        else:
            aligned_x.append('-')
            aligned_y.append(y[prev_j])

        i, j = prev_i, prev_j
        path.append((i, j))

    path.reverse()
    aligned_x.reverse()
    aligned_y.reverse()

    with open(output_path, "w") as f:
        f.write("Aligned Sequence X:\n")
        f.write(''.join(aligned_x) + '\n\n')

        f.write("Aligned Sequence Y:\n")
        f.write(''.join(aligned_y) + '\n\n')

        f.write("Traceback Path:\n")
        f.write(' <- '.join([f"({i},{j})" for i, j in path]) + '\n\n')

        f.write(f"Final Alignment Score:\n{final_score}\n\n")

        f.write("Score Table:\n")
        for row in score_table:
            f.write('\t'.join(map(str, row)) + '\n')
        f.write('\n')

        f.write("Previous Table:\n")
        for row in prev_table:
            f.write('\t'.join([f"({a},{b})" for (a, b) in row]) + '\n')

    return path, final_score


# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    ensure_directories()

    input_file = os.path.join("input", "input1.txt")
    config_file = os.path.join("config", "config4.txt")
    output_file = os.path.join("output", "output.txt")

    x, y = read_two_lines(input_file)
    score = read_config_file(config_file)

    score_table, prev_table = fill_dp_table(x, y, score)
    path, final_score = traceback_alignment(score_table, prev_table, x, y, output_file)

    print("Traceback Path:", path)
    print("Final Alignment Score:", final_score)
