'''
Title: Pairwise Alignment
Author: Braveenan Sritharan
Date Created: 2025-04-15
'''

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
    Returns a dictionary with scores for character pairs,
    and alpha and beta gap penalties.
    """
    score = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == "alpha":
                score["alpha"] = int(parts[1])
            elif parts[0] == "beta":
                score["beta"] = int(parts[1])
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

# Fm(i, j) = max(
#     Fm(i-1, j-1) + S(x[i], y[j]),  # State M to M
#     Fi(i-1, j-1) + S(x[i], y[j]),  # State I to M
#     Fd(i-1, j-1) + S(x[i], y[j]),  # State D to M
# )

# Fi(i, j) = max(
#     Fm(i, j-1) + alpha,  # State M to I
#     Fi(i, j-1) + beta,  # State I to I
# )

# Fd(i, j) = max(
#     Fm(i-1, j) + alpha,  # State M to D
#     Fd(i-1, j) + beta,  # State D to D
# )

# -------------------------
# Step 2: Fill DP Table
# -------------------------

"""
Fills the dynamic programming tables (score_table and prev_table)
using recurrence relations for match, insert, and delete states.
Each table is represented in score_table[k][i][j] and prev_table[k][i][j] where:
    k = 0 represents match state (M)
    k = 1 represents insert state (I)
    k = 2 represents delete state (D)
Each cell stores the maximum alignment score up to that point and
tracks its predecessor to support traceback.
"""

def fill_dp_table(x, y, score):
    m, n = len(x), len(y)
    alpha = score["alpha"]
    beta = score["beta"]

    # Initialize score table with -inf and prev_table with None
    score_table = [[[float('-inf')] * (n + 1) for _ in range(m + 1)] for _ in range(3)]
    prev_table = [[[None] * (n + 1) for _ in range(m + 1)] for _ in range(3)]

    # Base initialization
    score_table[0][0][0] = 0

    for j in range(1, n + 1):
        insert_candidates = [
            (score_table[0][0][j - 1] + alpha, 0),
            (score_table[1][0][j - 1] + beta,  1)
        ]
        score_table[1][0][j], k = max(insert_candidates, key=lambda x: x[0])
        if score_table[1][0][j] > float('-inf'):
            prev_table[1][0][j] = (k, 0, j - 1)

    for i in range(1, m + 1):
        delete_candidates = [
            (score_table[0][i - 1][0] + alpha, 0),
            (score_table[2][i - 1][0] + beta,  2)
        ]
        score_table[2][i][0], k = max(delete_candidates, key=lambda x: x[0])
        if score_table[2][i][0] > float('-inf'):
            prev_table[2][i][0] = (k, i - 1, 0)

    # Fill tables
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = score[(x[i - 1], y[j - 1])]

            match_candidates = [
                (score_table[0][i - 1][j - 1] + s, 0),
                (score_table[1][i - 1][j - 1] + s, 1),
                (score_table[2][i - 1][j - 1] + s, 2)
            ]
            score_table[0][i][j], k = max(match_candidates, key=lambda x: x[0])
            if score_table[0][i][j] > float('-inf'):
                prev_table[0][i][j] = (k, i - 1, j - 1)

            insert_candidates = [
                (score_table[0][i][j - 1] + alpha, 0),
                (score_table[1][i][j - 1] + beta,  1)
            ]
            score_table[1][i][j], k = max(insert_candidates, key=lambda x: x[0])
            if score_table[1][i][j] > float('-inf'):
                prev_table[1][i][j] = (k, i, j - 1)

            delete_candidates = [
                (score_table[0][i - 1][j] + alpha, 0),
                (score_table[2][i - 1][j] + beta,  2)
            ]
            score_table[2][i][j], k = max(delete_candidates, key=lambda x: x[0])
            if score_table[2][i][j] > float('-inf'):
                prev_table[2][i][j] = (k, i - 1, j)

    return score_table, prev_table

# -------------------------
# Step 3: Traceback and Output
# -------------------------

"""
Traces back through the prev_table starting from the highest-scoring
cell at the bottom-right of the DP table. It compares the final cell
across all three state tables (match, insert, delete), selects the one 
with the maximum score, and initiates traceback from that state. It 
reconstructs the aligned sequences, the traceback path, and writes all 
results including alignment, scores, and path tables to an output file.
"""

def traceback_alignment(score_table, prev_table, x, y, output_path):
    i, j = len(x), len(y)
    aligned_x = []
    aligned_y = []

    final_candidates = [
        (score_table[0][i][j], 0),
        (score_table[1][i][j], 1),
        (score_table[2][i][j], 2)
    ]
    final_score, final_prev = max(final_candidates, key=lambda x: x[0])

    k = final_prev
    path = [(k, i, j)]

    # Traceback to reconstruct the alignment
    while i > 0 or j > 0:
        prev_k, prev_i, prev_j = prev_table[k][i][j]

        if prev_i == i - 1 and prev_j == j - 1:
            aligned_x.append(x[prev_i])
            aligned_y.append(y[prev_j])
        elif prev_i == i - 1 and prev_j == j:
            aligned_x.append(x[prev_i])
            aligned_y.append('-')
        else:
            aligned_x.append('-')
            aligned_y.append(y[prev_j])

        k, i, j = prev_k, prev_i, prev_j
        path.append((k, i, j))

    # Reverse results to get correct alignment
    path.reverse()
    aligned_x.reverse()
    aligned_y.reverse()

    # Write all results to a file
    with open(output_path, "w") as f:
        f.write("Aligned Sequence X:\n")
        f.write(''.join(aligned_x) + '\n\n')

        f.write("Aligned Sequence Y:\n")
        f.write(''.join(aligned_y) + '\n\n')

        f.write("Traceback Path:\n")
        f.write(' <- '.join([f"({k},{i},{j})" for k, i, j in path]) + '\n\n')

        f.write(f"Final Alignment Score:\n{final_score}\n\n")

        # Score Tables
        f.write("Match Score Table:\n")
        for row in score_table[0]:
            f.write('\t'.join(map(str, row)) + '\n')
        f.write('\n')

        f.write("Insert Score Table:\n")
        for row in score_table[1]:
            f.write('\t'.join(map(str, row)) + '\n')
        f.write('\n')

        f.write("Delete Score Table:\n")
        for row in score_table[2]:
            f.write('\t'.join(map(str, row)) + '\n')
        f.write('\n')

        # Previous Tables
        f.write("Match Previous Table:\n")
        for row in prev_table[0]:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append("None")
                else:
                    a, b, c = cell
                    formatted_row.append(f"({a},{b},{c})")
            f.write('\t'.join(formatted_row) + '\n')
        f.write('\n')

        f.write("Insert Previous Table:\n")
        for row in prev_table[1]:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append("None")
                else:
                    a, b, c = cell
                    formatted_row.append(f"({a},{b},{c})")
            f.write('\t'.join(formatted_row) + '\n')
        f.write('\n')

        f.write("Delete Previous Table:\n")
        for row in prev_table[2]:
            formatted_row = []
            for cell in row:
                if cell is None:
                    formatted_row.append("None")
                else:
                    a, b, c = cell
                    formatted_row.append(f"({a},{b},{c})")
            f.write('\t'.join(formatted_row) + '\n')
        f.write('\n')

    return path, final_score

# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    ensure_directories()

    # Initialize input and config file paths
    input_file = os.path.join("input", "input1.txt")
    config_file = os.path.join("config", "config4.txt")
    output_file = os.path.join("output", "output.txt")

    # Read sequences and score configuration
    x, y = read_two_lines(input_file)
    score = read_config_file(config_file)

    # Fill DP table
    score_table, prev_table = fill_dp_table(x, y, score)

    # Perform traceback to build aligned sequences
    path, final_score = traceback_alignment(score_table, prev_table, x, y, output_file)

    print("Traceback Path:", path)
    print("Final Alignment Score:", final_score)
