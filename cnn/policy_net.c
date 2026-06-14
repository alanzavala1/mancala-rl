/* Pure-C inference for the trained policy network.
 *
 * The weights are baked in via policy_weights.h (generated from a checkpoint by
 * scripts/export_weights.py), so choosing a move is just a few matrix-vector
 * products + ReLU + an argmax -- microsecond scale, with no Python or PyTorch
 * at runtime. Called from Python via ctypes (mancala_rl/cpolicy.py).
 *
 * Board layout matches the engine: 0..5 player-1 pits, 6 store1,
 * 7..12 player-2 pits, 13 store2.
 */

#include <string.h>
#include <time.h>
#include "policy_weights.h"

#define MAX_DIM 1024

/* Perspective-canonical, divide-by-48 normalization -- identical to
 * features.encode: "my side, then opponent's side." */
static void encode(const int *board, int player, float *x) {
    if (player == 1) {
        for (int i = 0; i < 14; i++) x[i] = board[i] / 48.0f;
    } else {
        for (int i = 0; i < 7; i++) x[i]     = board[7 + i] / 48.0f;
        for (int i = 0; i < 7; i++) x[7 + i] = board[i]     / 48.0f;
    }
}

static void forward(const float *x, float *logits) {
    float a[MAX_DIM], b[MAX_DIM];
    const float *in = x;
    float *out = a;
    for (int L = 0; L < NUM_LAYERS; L++) {
        const Layer *ly = &LAYERS[L];
        for (int o = 0; o < ly->out; o++) {
            const float *row = ly->W + (size_t)o * ly->in;
            float s = ly->b[o];
            for (int i = 0; i < ly->in; i++) s += row[i] * in[i];
            if (L < NUM_LAYERS - 1 && s < 0.0f) s = 0.0f;   /* ReLU except last layer */
            out[o] = s;
        }
        in = out;
        out = (out == a) ? b : a;
    }
    memcpy(logits, in, sizeof(float) * OUT_DIM);
}

/* Best legal move (pit index 0..5) for the player to move. */
__declspec(dllexport)
int policy_best_move(const int *board, int player) {
    float x[IN_DIM], logits[OUT_DIM];
    encode(board, player, x);
    forward(x, logits);
    int base = (player == 1) ? 0 : 7;
    int best = -1;
    float best_v = -1e30f;
    for (int a = 0; a < 6; a++) {
        if (board[base + a] > 0 && logits[a] > best_v) { best_v = logits[a]; best = a; }
    }
    return best;
}

__declspec(dllexport)
void policy_logits(const int *board, int player, float *out) {
    float x[IN_DIM];
    encode(board, player, x);
    forward(x, out);
}

/* Pure-C latency in microseconds/move (no ctypes/Python overhead). */
__declspec(dllexport)
double policy_bench(const int *board, int player, int reps) {
    volatile int sink = 0;
    clock_t t0 = clock();
    for (int i = 0; i < reps; i++) sink += policy_best_move(board, player);
    clock_t t1 = clock();
    (void)sink;
    return (double)(t1 - t0) / CLOCKS_PER_SEC / reps * 1e6;
}
