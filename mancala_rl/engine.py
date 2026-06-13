"""Mancala game engine (capture variant).

Rules:
- Two players, six pits each, one store each, four seeds per pit at the start.
- Sow seeds counterclockwise one per pit; skip the opponent's store.
- If the last seed lands in your own store, you take another turn.
- If the last seed lands in one of your own pits that was empty, you capture
  that seed plus every seed in the pit directly opposite, into your store.
- The game ends when one player's six pits are all empty. The other player
  sweeps their remaining seeds into their store. Higher store wins; equal ties.

The sowing/capture logic is ported unchanged from the original project's
mancala.py. The difference is the interface: this module is side-effect free
(no input(), no printing in the API path) and exposes an immutable State so
bots and learning agents can call it programmatically.

Board layout and sowing order:

    A  B  C  D  E  F  [1] L  K  J  I  H  G  [2]  -> wraps back to A

  Player 1 owns pits A-F and store '1'; player 2 owns pits G-L and store '2'.
"""

from dataclasses import dataclass

P1_PITS = ('A', 'B', 'C', 'D', 'E', 'F')
P2_PITS = ('G', 'H', 'I', 'J', 'K', 'L')

# Index 6 is store 1, index 13 is store 2.
BOARD_ORDER = ('A', 'B', 'C', 'D', 'E', 'F', '1',
               'G', 'H', 'I', 'J', 'K', 'L', '2')

OPPOSITE_PIT = {
    'A': 'G', 'B': 'H', 'C': 'I', 'D': 'J', 'E': 'K', 'F': 'L',
    'G': 'A', 'H': 'B', 'I': 'C', 'J': 'D', 'K': 'E', 'L': 'F',
}

NEXT_PIT = {
    'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F', 'F': '1',
    '1': 'L', 'L': 'K', 'K': 'J', 'J': 'I', 'I': 'H', 'H': 'G',
    'G': '2', '2': 'A',
}

STARTING_SEEDS = 4


@dataclass(frozen=True)
class State:
    """An immutable game position.

    board: the 14 counts in BOARD_ORDER.
    current_player: 1 or 2, the player whose turn it is to act.
    """
    board: tuple
    current_player: int


def _to_dict(state):
    return dict(zip(BOARD_ORDER, state.board))


def _from_dict(d, player):
    return State(tuple(d[k] for k in BOARD_ORDER), player)


def pits_for(player):
    return P1_PITS if player == 1 else P2_PITS


def reset(first_player=1):
    """Return the standard opening position with `first_player` to act."""
    d = {p: STARTING_SEEDS for p in P1_PITS + P2_PITS}
    d['1'] = 0
    d['2'] = 0
    return _from_dict(d, first_player)


def legal_moves(state):
    """Return the legal actions as pit indices 0..5 for the current player.

    Action i refers to the i-th pit on the current player's side: A..F for
    player 1, G..L for player 2. A pit is legal only if it is non-empty.
    """
    pits = pits_for(state.current_player)
    d = _to_dict(state)
    return [i for i, p in enumerate(pits) if d[p] > 0]


def stores(state):
    """Return (player_1_store, player_2_store)."""
    d = _to_dict(state)
    return d['1'], d['2']


def _sow(d, player, pit):
    """Sow the seeds from `pit` in place. Return (last_pit, extra_turn).

    Mutates d. `player` is '1' or '2'; `pit` is a letter.
    """
    own_store = '1' if player == '1' else '2'
    opp_store = '2' if player == '1' else '1'
    own_pits = P1_PITS if player == '1' else P2_PITS

    seeds = d[pit]
    d[pit] = 0
    while seeds > 0:
        pit = NEXT_PIT[pit]
        if pit == opp_store:
            continue
        d[pit] += 1
        seeds -= 1

    if pit == own_store:
        return pit, True

    # Capture: last seed landed in one of our own pits that was empty.
    if pit in own_pits and d[pit] == 1:
        opposite = OPPOSITE_PIT[pit]
        if d[opposite] > 0:
            d[own_store] += d[pit] + d[opposite]
            d[pit] = 0
            d[opposite] = 0

    return pit, False


def _resolve_if_over(d):
    """If one side is empty, sweep and return the winner; else return None.

    Mutates d when the game is over. Winner is 1, 2, or 0 for a tie.
    """
    p1 = sum(d[p] for p in P1_PITS)
    p2 = sum(d[p] for p in P2_PITS)
    if p1 != 0 and p2 != 0:
        return None

    if p1 == 0:
        d['2'] += p2
        for p in P2_PITS:
            d[p] = 0
    else:
        d['1'] += p1
        for p in P1_PITS:
            d[p] = 0

    if d['1'] > d['2']:
        return 1
    if d['2'] > d['1']:
        return 2
    return 0


def step(state, action):
    """Apply `action` (a pit index 0..5) for the current player.

    Returns (next_state, reward, done, info).

    reward is sparse and from the perspective of the player who just moved:
    0 for every non-terminal move, then +1 win / -1 loss / 0 tie at the end.
    On an extra turn, next_state.current_player is unchanged. info carries
    'winner' (1/2/0/None), 'extra_turn' (bool), and 'move' (the pit letter).
    """
    legal = legal_moves(state)
    if action not in legal:
        raise ValueError(
            f"illegal action {action!r}; legal actions are {legal}"
        )

    player = state.current_player
    pit = pits_for(player)[action]

    d = _to_dict(state)
    _, extra_turn = _sow(d, str(player), pit)
    winner = _resolve_if_over(d)
    done = winner is not None

    if done:
        reward = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        next_player = player
    else:
        reward = 0.0
        next_player = player if extra_turn else (2 if player == 1 else 1)

    info = {'winner': winner, 'extra_turn': extra_turn, 'move': pit}
    return _from_dict(d, next_player), reward, done, info
