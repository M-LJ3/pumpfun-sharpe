"""Pump.fun Sharpe Sprint — clean paper bot.

Ranks by hourly Sharpe ratio, so this bot optimises for steady, low-variance
gains rather than one lucky moonshot: tiny 0.01 SOL stakes, only in coins
still priced near their creation price, quick take-profit, tight stop-loss,
short holds, at most 5 open positions, never a re-entry on the same coin.

Paper trading only: no real funds, no shorts, no leverage. Fully
deterministic: decisions are pure functions of the replay state — no clock,
no randomness, no network, no set iteration.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bot  # noqa: E402  (after the sys.path line: -P leaves the script directory off sys.path)

LAMPORTS_PER_SOL = makefun_bot.LAMPORTS_PER_SOL

STAKE_LAMPORTS = 10_000_000  # 0.01 SOL per coin: the minimum round trip
ENTRY_MAX_NUM = 107  # buy only while price <= 107% of creation price:
# the size cap needs 1 SOL of real reserves, which already moves price
# ~7%, so 107% is the tightest reachable filter — it buys coins that
# have not pumped yet instead of chasing ones that already did
ENTRY_MAX_DEN = 100
TAKE_PROFIT_BPS = 400  # sell when mark is 4% above cost per token
STOP_LOSS_BPS = 100000  # no stop-loss: coins retrace to creation price,
# so a tight stop just locks in the fee bleed; the hold cap exits instead
HOLD_SLOTS = 50000  # about 3 hours: let the one pump arrive, then leave
MAX_SLIPPAGE_BPS = 500
WAIT_RETRACE = False  # when True, keep waiting if the price is above the
# entry cap instead of skipping the coin for good: coins retrace to
# creation price, so a later dip can still be a buy
TRAIL_BPS = 0  # when >0, arm a trailing stop once the mark is 10% above
# cost: sell when the mark falls TRAIL_BPS below its peak (lets winners
# run past the fixed take-profit instead of capping them there)
TRAIL_ARM_BPS = 1000

#: mint -> (creation virtual sol reserves, creation virtual token reserves)
_creation = {}
#: mints already bought (never re-enter the same coin)
_bought = set()
#: mints already decided and skipped for good (pumped past the entry cap)
_skipped = set()
#: mints still waiting for the size cap to allow a 0.01 SOL buy
_waiting = set()
#: mint -> slot of our buy, for the hold cap
_buy_slot = {}
#: mint -> best mark/cost ratio in bps seen since the buy (trailing stop)
_peak_ratio = {}
#: every order this run that actually filled, in order:
#: (seq, side, mint, amount) — side is buy/sell, plus "close" for a
#: harness close on a CompleteEvent. Filled orders are what the
#: guardrails count: a sent order the market rejects opens nothing.
_filled_orders = []


def audit_sharpe_log(state):
    """Re-read our own filled orders and verify the guardrails held.

    Checks: at most 5 simultaneous open positions, every sell follows a
    buy of the same mint, no mint bought twice, every buy is exactly the
    minimum round-trip stake. Prints one summary line to stderr so the
    protocol stream on stdout stays clean. Returns True when all pass.
    """
    open_now = set()
    bought = set()
    max_open = 0
    reentered = 0
    bad_stake = 0
    sell_without_buy = 0
    round_trips = 0
    for _seq, side, mint, amount in _filled_orders:
        if side == "buy":
            if mint in bought:
                reentered += 1
            bought.add(mint)
            open_now.add(mint)
            if amount != STAKE_LAMPORTS:
                bad_stake += 1
        elif side == "sell":
            if mint in open_now:
                open_now.discard(mint)
                round_trips += 1
            else:
                sell_without_buy += 1
        else:  # harness close on a CompleteEvent
            open_now.discard(mint)
        if len(open_now) > max_open:
            max_open = len(open_now)
    ok = (max_open <= state.hello["caps"]["max_open_positions"]
          and reentered == 0 and bad_stake == 0 and sell_without_buy == 0)
    print(f"audit_sharpe_log: distinct_bought={len(bought)} "
          f"round_trips={round_trips} max_open={max_open} "
          f"reentered={reentered} bad_stake={bad_stake} "
          f"sell_without_buy={sell_without_buy} -> "
          f"{'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return ok


def on_finish(state, done):
    audit_sharpe_log(state)


def _cost_ratio_bps(state, mint):
    """Mark / cost-per-token in basis points, or None when unknowable."""
    holding = state.positions.get(mint)
    curve = state.curves.get(mint)
    if holding is None or curve is None:
        return None
    if holding.token_amount <= 0 or curve.virtual_token_reserves <= 0:
        return None
    # mark per token (lamports) = vquote / vtok
    # cost per token (lamports) = cost_lamports / token_amount
    # ratio_bps = mark / cost * 10000
    num = curve.virtual_quote_reserves * holding.token_amount
    den = holding.cost_lamports * curve.virtual_token_reserves
    if den <= 0:
        return None
    return num * 10000 // den


def _entry_ok(state, mint):
    """True when the coin is still near its creation price."""
    curve = state.curves.get(mint)
    created = _creation.get(mint)
    if curve is None or created is None:
        return False
    if curve.complete or curve.virtual_token_reserves <= 0:
        return False
    c_vsol, c_vtok = created
    # vsol / vtok <= c_vsol / c_vtok * 105 / 100
    left = curve.virtual_quote_reserves * c_vtok * ENTRY_MAX_DEN
    right = c_vsol * curve.virtual_token_reserves * ENTRY_MAX_NUM
    return left <= right


def on_batch(state, batch):
    caps = state.hello["caps"]
    tx = state.tx_fee_lamports()
    free = state.free_cash_lamports()
    slots = caps["max_open_positions"] - state.committed_position_count()
    orders = []

    # 0. record what actually filled, for the end-of-run self-check.
    for f in batch.get("fills", []):
        if f.get("status") == "filled":
            if f["side"] == "buy":
                _filled_orders.append((state.seq, "buy", f["mint"],
                                       int(f["quote_lamports"])))
            else:
                _filled_orders.append((state.seq, "sell", f["mint"],
                                       int(f["token_amount"])))
    for e in batch["events"]:
        if e["event"] == "CompleteEvent":
            _filled_orders.append((state.seq, "close", e["mint"], 0))

    # 1. exits: take small profits, cut small losses, never hold past the cap.
    pending_sell = {p.order["mint"] for p in state.pending.values()
                    if p.order["side"] == "sell"}
    for mint in list(state.positions.keys()):
        holding = state.positions.get(mint)
        if holding is None or holding.token_amount <= 0:
            continue
        if mint in pending_sell:
            continue  # a sell is already in flight for this coin
        curve = state.curves.get(mint)
        if curve is None or curve.complete:
            continue
        ratio = _cost_ratio_bps(state, mint)
        if ratio is None:
            continue
        if ratio > _peak_ratio.get(mint, 0):
            _peak_ratio[mint] = ratio
        age = state.slot - _buy_slot.get(mint, state.slot)
        trailing_hit = (TRAIL_BPS > 0
                        and _peak_ratio.get(mint, 0) >= 10000 + TRAIL_ARM_BPS
                        and ratio <= _peak_ratio[mint] - TRAIL_BPS)
        if (ratio >= 10000 + TAKE_PROFIT_BPS
                or ratio <= 10000 - STOP_LOSS_BPS
                or age > HOLD_SLOTS
                or trailing_hit):
            orders.append({"side": "sell", "mint": mint,
                           "token_amount": holding.token_amount,
                           "max_slippage_bps": MAX_SLIPPAGE_BPS})
            _peak_ratio.pop(mint, None)

    # 2. entries: note fresh coins, buy the ones still near creation price.
    for e in batch["events"]:
        mint = e["mint"]
        if e["event"] == "CreateEvent":
            _creation[mint] = (int(e["virtual_quote_reserves"]),
                               int(e["virtual_token_reserves"]))
            if mint not in _bought and mint not in _skipped:
                _waiting.add(mint)
        if mint not in _waiting:
            continue
        curve = state.curves.get(mint)
        if curve is None or curve.complete:
            _waiting.discard(mint)
            continue
        if state.max_buy_lamports(mint) < STAKE_LAMPORTS:
            continue  # reserves still too small; keep waiting, judge price later
        if not _entry_ok(state, mint):
            # already pumped past the entry cap by the time we could buy
            if WAIT_RETRACE:
                continue  # stay waiting: the coin may retrace to the cap
            _waiting.discard(mint)
            _skipped.add(mint)
            continue
        if len(orders) >= caps["max_orders_per_batch"] or slots <= 0:
            continue
        if free < STAKE_LAMPORTS + tx:
            continue
        orders.append({"side": "buy", "mint": mint,
                       "quote_lamports": STAKE_LAMPORTS,
                       "max_slippage_bps": MAX_SLIPPAGE_BPS})
        _waiting.discard(mint)
        _bought.add(mint)
        _buy_slot[mint] = state.slot
        free -= STAKE_LAMPORTS + tx
        slots -= 1
    return orders


if __name__ == "__main__":
    makefun_bot.run(on_batch, on_done=on_finish)
