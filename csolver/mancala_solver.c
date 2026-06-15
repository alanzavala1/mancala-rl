/* mancala_solver.c
 *
 * Depth-limited alpha-beta search for our Kalah(6,4) engine, compiled to a DLL
 * and called from Python via ctypes. This is the "fast brain": the heavy game-
 * tree search lives here in C; everything else (evaluation harness, training,
 * the neural net) stays in Python.
 *
 * Board layout matches the Python engine exactly (14 ints):
 *     0..5  = A..F  (player 1 pits)     6  = store 1
 *     7..12 = G..L  (player 2 pits)     13 = store 2
 *
 * Player 1 maximizes the margin (store1 - store2); player 2 minimizes it.
 *
 * Note: this is NOT negamax. The extra-turn rule (land in your own store -> move
 * again) means players don't strictly alternate, so we keep explicit min/max
 * keyed on whose turn it actually is. The same subtlety as the Python solver.
 */

#include <string.h>

#define STORE1 6
#define STORE2 13
#define NEG_INF (-1000000)
#define POS_INF ( 1000000)

/* Counterclockwise successor of each board index (full ring, both stores). */
static const int NEXT[14] = {1, 2, 3, 4, 5, 6, 12, 13, 7, 8, 9, 10, 11, 0};

/* Pit directly opposite each pit; -1 for the two stores. */
static const int OPP[14] = {7, 8, 9, 10, 11, 12, -1, 0, 1, 2, 3, 4, 5, -1};

/* Apply action (pit 0..5) for `player` (1 or 2) to board `b`, writing the
 * result into `out`. Returns 1 if the game ended (and sets *margin to the final
 * store1-store2), otherwise returns 0 and sets *next_player. Faithful to the
 * Python engine's sow / capture / end-game sweep. */
static int apply_move(const int *b, int player, int action,
                      int *out, int *next_player, int *margin) {
    memcpy(out, b, sizeof(int) * 14);

    int idx = (player == 1) ? action : 7 + action;
    int own_store = (player == 1) ? STORE1 : STORE2;
    int opp_store = (player == 1) ? STORE2 : STORE1;
    int lo = (player == 1) ? 0 : 7;          /* own pits are [lo, lo+5] */

    int seeds = out[idx];
    out[idx] = 0;
    while (seeds > 0) {
        idx = NEXT[idx];
        if (idx == opp_store) continue;
        out[idx]++;
        seeds--;
    }

    int extra_turn = (idx == own_store);
    if (!extra_turn && idx >= lo && idx <= lo + 5 && out[idx] == 1) {
        int opp = OPP[idx];                       /* own pits always have a valid opposite */
        out[own_store] += out[idx] + out[opp];    /* empty-capture: opposite may be 0 */
        out[idx] = 0;
        out[opp] = 0;
    }

    int p1 = out[0] + out[1] + out[2] + out[3] + out[4] + out[5];
    int p2 = out[7] + out[8] + out[9] + out[10] + out[11] + out[12];
    if (p1 == 0 || p2 == 0) {
        if (p1 == 0) { out[STORE2] += p2; for (int i = 7; i < 13; i++) out[i] = 0; }
        else         { out[STORE1] += p1; for (int i = 0; i < 6; i++) out[i] = 0; }
        *margin = out[STORE1] - out[STORE2];
        return 1;
    }

    *next_player = extra_turn ? player : (player == 1 ? 2 : 1);
    return 0;
}

/* Heuristic for when we run out of lookahead: store differential (weighted),
 * plus a small bonus for seeds sitting on your own side (closer to banking).
 * Always from player 1's perspective. */
static int evaluate(const int *b) {
    int p1 = b[0] + b[1] + b[2] + b[3] + b[4] + b[5];
    int p2 = b[7] + b[8] + b[9] + b[10] + b[11] + b[12];
    return 4 * (b[STORE1] - b[STORE2]) + (p1 - p2);
}

static int legal(const int *b, int player, int *moves) {
    int n = 0, lo = (player == 1) ? 0 : 7;
    for (int a = 0; a < 6; a++)
        if (b[lo + a] > 0) moves[n++] = a;
    return n;
}

/* Depth-limited alpha-beta. Returns the position's value to `depth` plies,
 * from player 1's perspective. Exact when the game ends within the budget. */
static int alphabeta(const int *b, int player, int depth, int alpha, int beta) {
    int moves[6];
    int n = legal(b, player, moves);

    if (player == 1) {
        int best = NEG_INF;
        for (int i = 0; i < n; i++) {
            int child[14], np, margin, v;
            if (apply_move(b, player, moves[i], child, &np, &margin)) v = margin;
            else if (depth <= 1) v = evaluate(child);
            else v = alphabeta(child, np, depth - 1, alpha, beta);
            if (v > best) best = v;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
        return best;
    } else {
        int best = POS_INF;
        for (int i = 0; i < n; i++) {
            int child[14], np, margin, v;
            if (apply_move(b, player, moves[i], child, &np, &margin)) v = margin;
            else if (depth <= 1) v = evaluate(child);
            else v = alphabeta(child, np, depth - 1, alpha, beta);
            if (v < best) best = v;
            if (best < beta) beta = best;
            if (beta <= alpha) break;
        }
        return best;
    }
}

/* --- exported API ------------------------------------------------------- */

/* Fill out[6] with the value of each pit action for `player`, searched to
 * `depth` plies (player 1's perspective). Illegal actions get a sentinel the
 * caller will ignore. Each action is searched with a full window, so the
 * values are exact-to-depth and can be compared/ranked directly. */
__declspec(dllexport)
void move_values(const int *b, int player, int depth, int *out) {
    int lo = (player == 1) ? 0 : 7;
    for (int a = 0; a < 6; a++) {
        if (b[lo + a] == 0) { out[a] = NEG_INF; continue; }   /* illegal */
        int child[14], np, margin, v;
        if (apply_move(b, player, a, child, &np, &margin)) v = margin;
        else if (depth <= 1) v = evaluate(child);
        else v = alphabeta(child, np, depth - 1, NEG_INF, POS_INF);
        out[a] = v;
    }
}

/* Apply one move; write the resulting board into out[14]. Returns the next
 * player (1 or 2), or 0 if the game ended. Exists so Python can cross-check the
 * C rules against the trusted engine. */
__declspec(dllexport)
int apply_one(const int *b, int player, int action, int *out) {
    int np, margin;
    return apply_move(b, player, action, out, &np, &margin) ? 0 : np;
}
