"""
Title: Pairwise Alignment
Author: Braveenan Sritharan
Date Created: 2025-04-12

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

# F(i, j) = max(Fm(i, j), Fi(i, j), Fd(i, j))


# -------------------------
# Step 2: Fill DP Table
# -------------------------

def fill_dp_table(x, y, score):
    """
    Fills the DP tables for affine gap penalties.
    Returns: 
    - m/i/d_score_tables,
    - m/i/d_prev_tables,
    - f_score_table (final max score table),
    - f_prev_table (to start traceback from max state).
    """

    m, n = len(x), len(y)
    alpha = score["alpha"]
    beta = score["beta"]

    # Score tables for Match, Insertion, Deletion
    m_score_table = [[float('-inf')] * (n + 1) for _ in range(m + 1)]
    i_score_table = [[float('-inf')] * (n + 1) for _ in range(m + 1)]
    d_score_table = [[float('-inf')] * (n + 1) for _ in range(m + 1)]

    # Previous tables for traceback in each state
    m_prev_table = [[None] * (n + 1) for _ in range(m + 1)]
    i_prev_table = [[None] * (n + 1) for _ in range(m + 1)]
    d_prev_table = [[None] * (n + 1) for _ in range(m + 1)]

    # Final score and backtrack table
    f_score_table = [[float('-inf')] * (n + 1) for _ in range(m + 1)]
    f_prev_table = [[None] * (n + 1) for _ in range(m + 1)]

    # Base initialization
    m_score_table[0][0] = 0
    f_score_table[0][0] = 0

    for i in range(1, m + 1):
        d_score_table[i][0] = alpha + (i - 1) * beta
        d_prev_table[i][0] = (2, i - 1, 0)

    for j in range(1, n + 1):
        i_score_table[0][j] = alpha + (j - 1) * beta
        i_prev_table[0][j] = (1, 0, j - 1)

    # Fill tables
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = score[(x[i - 1], y[j - 1])]

            # Match (M)
            match_options = [
                (m_score_table[i - 1][j - 1], 0),
                (i_score_table[i - 1][j - 1], 1),
                (d_score_table[i - 1][j - 1], 2),
            ]
            best_m_score, best_m_from = max((val + s, state) for val, state in match_options)
            m_score_table[i][j] = best_m_score
            m_prev_table[i][j] = (best_m_from, i - 1, j - 1)

            # Insertion (I)
            from_m = m_score_table[i][j - 1] + alpha
            from_i = i_score_table[i][j - 1] + beta
            if from_m >= from_i:
                i_score_table[i][j] = from_m
                i_prev_table[i][j] = (0, i, j - 1)
            else:
                i_score_table[i][j] = from_i
                i_prev_table[i][j] = (1, i, j - 1)

            # Deletion (D)
            from_m = m_score_table[i - 1][j] + alpha
            from_d = d_score_table[i - 1][j] + beta
            if from_m >= from_d:
                d_score_table[i][j] = from_m
                d_prev_table[i][j] = (0, i - 1, j)
            else:
                d_score_table[i][j] = from_d
                d_prev_table[i][j] = (2, i - 1, j)

            # Final table F(i, j)
            final_scores = [
                (m_score_table[i][j], 0),
                (i_score_table[i][j], 1),
                (d_score_table[i][j], 2),
            ]
            best_f_score, best_state = max(final_scores)
            f_score_table[i][j] = best_f_score
            f_prev_table[i][j] = (best_state, i, j)

    return (
        (m_score_table, i_score_table, d_score_table),
        (m_prev_table, i_prev_table, d_prev_table),
        f_score_table,
        f_prev_table
    )


# -------------------------
# Step 3: Traceback and Output
# -------------------------

def traceback_alignment_from_f_table(
    f_score_table, f_prev_table,
    m_score_table, i_score_table, d_score_table,
    m_prev_table, i_prev_table, d_prev_table,
    x, y, output_path
):
    """
    Traceback using f_score_table and f_prev_table and logs the full alignment.
    Also writes all DP and traceback tables (M, I, D, F) to the output log.
    """

    m, n = len(x), len(y)

    # Initialize from f_prev_table: (k, i, j)
    k, i, j = f_prev_table[m][n]
    path = [(i, j, k)]
    aligned_x = []
    aligned_y = []

    final_score = f_score_table[m][n]

    while i > 0 or j > 0:
        if k == 0:
            prev_k, prev_i, prev_j = m_prev_table[i][j]
            aligned_x.append(x[prev_i])
            aligned_y.append(y[prev_j])
        elif k == 1:
            prev_k, prev_i, prev_j = i_prev_table[i][j]
            aligned_x.append('-')
            aligned_y.append(y[prev_j])
        elif k == 2:
            prev_k, prev_i, prev_j = d_prev_table[i][j]
            aligned_x.append(x[prev_i])
            aligned_y.append('-')
        else:
            raise ValueError(f"Invalid state k: {k}")

        i, j, k = prev_i, prev_j, prev_k
        path.append((i, j, k))

    # Reverse sequences and path
    aligned_x.reverse()
    aligned_y.reverse()
    path.reverse()

    # Write to output log
    with open(output_path, "w") as f:
        f.write("Aligned Sequence X:\n")
        f.write(''.join(aligned_x) + '\n\n')

        f.write("Aligned Sequence Y:\n")
        f.write(''.join(aligned_y) + '\n\n')

        f.write("Traceback Path (i, j, k):\n")
        f.write(' <- '.join([f"({i},{j},{k})" for i, j, k in path]) + '\n\n')

        f.write(f"Final Alignment Score: {final_score}\n\n")

        def write_table(table, name):
            f.write(f"{name}:\n")
            for row in table:
                f.write('\t'.join(f"{val:.1f}" if isinstance(val, float) else str(val) for val in row) + '\n')
            f.write('\n')

        def write_prev_table(table, name):
            f.write(f"{name} (Previous Table):\n")
            for row in table:
                formatted_row = []
                for cell in row:
                    if cell is None:
                        formatted_row.append("None")
                    else:
                        formatted_row.append(f"({cell[0]},{cell[1]},{cell[2]})")
                f.write('\t'.join(formatted_row) + '\n')
            f.write('\n')

        # Score tables
        write_table(m_score_table, "Match Score Table (M)")
        write_table(i_score_table, "Insertion Score Table (I)")
        write_table(d_score_table, "Deletion Score Table (D)")
        write_table(f_score_table, "Final Score Table (F)")

        # Traceback tables
        write_prev_table(m_prev_table, "Match Traceback Table (M_prev)")
        write_prev_table(i_prev_table, "Insertion Traceback Table (I_prev)")
        write_prev_table(d_prev_table, "Deletion Traceback Table (D_prev)")
        write_prev_table(f_prev_table, "Final Traceback Table (F_prev)")

    return path, final_score



# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    ensure_directories()

    # Initialize input and config file paths
    input_file = os.path.join("input", "input1.txt")
    config_file = os.path.join("config", "config3.txt")
    output_file = os.path.join("output", "output.txt")

    # Read sequences and score configuration
    x, y = read_two_lines(input_file)
    score = read_config_file(config_file)

    # Fill DP tables (returns all necessary tables)
    (m_score_table, i_score_table, d_score_table), \
    (m_prev_table, i_prev_table, d_prev_table), \
    f_score_table, f_prev_table = fill_dp_table(x, y, score)

    # Perform traceback and write output
    path, final_score = traceback_alignment_from_f_table(
        f_score_table, f_prev_table,
        m_score_table, i_score_table, d_score_table,
        m_prev_table, i_prev_table, d_prev_table,
        x, y, output_file
    )

    print("Traceback Path:", path)
    print("Final Alignment Score:", final_score)
