"""Verify our engine reproduces the published exact values of Kalah(6, n).

Irving, Donkers & Uiterwijk (2000), "Solving Kalah" (ICGA Journal 23(3)),
solved the *empty-capture* variant of Kalah. Their Table 9 gives the exact game
value -- the final store margin (store1 - store2) under perfect play, with the
first player = South:

    Kalah(6,1) = +2     Kalah(6,2) = +10    Kalah(6,3) = +2
    Kalah(6,4) = +10    Kalah(6,5) = +12

We run our own exact C solver (mancala_rl.csolver: alpha-beta + transposition
table) on the opening of each size and check it matches. A match is ground
truth on our *own* code -- it confirms our rules are that solved variant, not
taken on faith. Larger n are dramatically harder (see the paper's complexity
tables), so this stops early if a solve exceeds the time budget.

    .venv\\Scripts\\python scripts/verify_solution.py --max-n 2
    .venv\\Scripts\\python scripts/verify_solution.py --max-n 4 --budget 600
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mancala_rl import csolver, engine

# Irving, Donkers & Uiterwijk (2000), Table 9 (first-player margin under perfect play).
PUBLISHED = {1: 2, 2: 10, 3: 2, 4: 10, 5: 12}


def opening(n):
    """Kalah(6, n) opening: n seeds in each of the 6 pits per side, stores 0."""
    return tuple([n] * 6 + [0] + [n] * 6 + [0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-n", type=int, default=2, help="largest Kalah(6,n) to attempt")
    p.add_argument("--budget", type=float, default=120.0,
                   help="stop before starting a solve if the previous one took longer (s)")
    args = p.parse_args()

    print(f"  {'game':<11}{'published':>10}{'ours':>7}{'nodes':>16}{'time':>10}   result",
          flush=True)
    last = 0.0
    for n in range(1, args.max_n + 1):
        if last > args.budget:
            print(f"  Kalah(6,{n}): skipped -- previous solve took {last:.0f}s "
                  f"(> budget {args.budget:.0f}s)", flush=True)
            break
        board = opening(n)
        t0 = time.perf_counter()
        v = csolver.solve_exact(engine.State(board, 1))
        nodes = csolver.solve_nodes()
        last = time.perf_counter() - t0
        pub = PUBLISHED.get(n)
        result = "OK" if (pub is not None and v == pub) else \
                 ("MISMATCH" if pub is not None else "(no published value)")
        pub_s = f"+{pub}" if pub is not None else "?"
        print(f"  Kalah(6,{n}){pub_s:>10}{v:>+7}{nodes:>16,}{last:>9.1f}s   {result}",
              flush=True)


if __name__ == "__main__":
    main()
