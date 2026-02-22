import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyBboxPatch

import streamlit as st


# =============================
# Constants / Names
# =============================
WHITE = 1
BLACK = -1
BAR = 0
OFF = 25

WHITE_NAME = "Yonatan"
BLACK_NAME = "Michael"

def player_name(player: int) -> str:
    return WHITE_NAME if player == WHITE else BLACK_NAME


# =============================
# Renderer (modified: returns fig, doesn't plt.show)
# =============================
def draw_bg_felt(board, title="", dice=None, bar_white=0, bar_black=0, off_white=0, off_black=0):
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    wood_outer = "#3b1d1d"
    wood_inner = "#6b3b2a"
    felt = "#1f6f5b"
    rail = "#c9c9c9"
    rail_dark = "#9a9a9a"
    tri_dark = "#0f4f3f"
    tri_light = "#2b8a73"

    ax.add_patch(FancyBboxPatch((0.3, 0.3), 13.4, 9.4,
                               boxstyle="round,pad=0.02,rounding_size=0.25",
                               facecolor=wood_outer, edgecolor="#1b0b0b", linewidth=3))
    ax.add_patch(FancyBboxPatch((0.6, 0.6), 12.8, 8.8,
                               boxstyle="round,pad=0.02,rounding_size=0.22",
                               facecolor=wood_inner, edgecolor="#2a120f", linewidth=2))
    ax.add_patch(FancyBboxPatch((0.9, 0.9), 12.2, 8.2,
                               boxstyle="round,pad=0.02,rounding_size=0.18",
                               facecolor=felt, edgecolor="#0f3b30", linewidth=2))

    ax.add_patch(Rectangle((1.05, 1.1), 0.65, 7.8, facecolor="#2a2a2a", edgecolor=rail, linewidth=1.2, alpha=0.35))
    ax.add_patch(Rectangle((12.3, 1.1), 0.65, 7.8, facecolor="#2a2a2a", edgecolor=rail, linewidth=1.2, alpha=0.35))

    ax.add_patch(Rectangle((1.8, 1.05), 10.35, 7.9, fill=False, edgecolor=rail, linewidth=2))
    ax.add_patch(Rectangle((1.8, 5.0), 10.35, 0.1, facecolor=rail_dark, edgecolor="none", alpha=0.5))

    bar_x = 6.85
    ax.add_patch(Rectangle((bar_x, 1.05), 0.5, 7.9, facecolor="#bfbfbf", edgecolor=rail, linewidth=1.5))
    ax.add_patch(Rectangle((bar_x+0.05, 1.15), 0.4, 7.7, facecolor="#9e9e9e", edgecolor="none", alpha=0.35))

    left_start = 1.95
    point_w = 0.78
    gap = 0.55

    def x_for_slot(slot):
        if slot <= 5:
            return left_start + slot * point_w
        else:
            return left_start + (slot * point_w) + gap

    top_points = list(range(13, 25))
    bottom_points = list(range(12, 0, -1))

    def draw_triangle(x0, top, dark):
        color = tri_dark if dark else tri_light
        if top:
            pts = [(x0, 8.85), (x0+point_w, 8.85), (x0+point_w/2, 5.15)]
        else:
            pts = [(x0, 1.15), (x0+point_w, 1.15), (x0+point_w/2, 4.85)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=rail, linewidth=0.6, alpha=0.95))

    for i, _p in enumerate(top_points):
        x0 = x_for_slot(i)
        draw_triangle(x0, top=True, dark=(i % 2 == 0))
    for i, _p in enumerate(bottom_points):
        x0 = x_for_slot(i)
        draw_triangle(x0, top=False, dark=(i % 2 == 0))

    for i, p in enumerate(top_points):
        ax.text(x_for_slot(i)+point_w/2, 9.05, str(p), ha="center", va="bottom", fontsize=8, color="#e8e8e8")
    for i, p in enumerate(bottom_points):
        ax.text(x_for_slot(i)+point_w/2, 0.95, str(p), ha="center", va="top", fontsize=8, color="#e8e8e8")

    def draw_checker(x, y, is_white):
        if is_white:
            face = "#f4f4f4"
            edge = "#1a1a1a"
        else:
            face = "#1a1a1a"
            edge = "#f4f4f4"
        ax.add_patch(Circle((x, y), 0.23, facecolor=face, edgecolor=edge, linewidth=2))

    def point_to_xy(point, stack_index):
        if 13 <= point <= 24:
            slot = point - 13
            x0 = x_for_slot(slot)
            x = x0 + point_w/2
            y = 8.55 - stack_index * 0.5
        else:
            slot = 12 - point
            x0 = x_for_slot(slot)
            x = x0 + point_w/2
            y = 1.45 + stack_index * 0.5
        return x, y

    for p in range(1, 25):
        v = board[p]
        if v == 0:
            continue
        is_white = v > 0
        cnt = abs(v)
        for k in range(min(cnt, 8)):
            x, y = point_to_xy(p, k)
            draw_checker(x, y, is_white)
        if cnt > 8:
            x, y = point_to_xy(p, 8)
            ax.text(x, y, f"+{cnt-8}", ha="center", va="center",
                    fontsize=10, color="#ffffff" if not is_white else "#000000", fontweight="bold")

    def draw_bar_stack(count, is_white, top=True):
        for k in range(min(count, 6)):
            x = bar_x + 0.25
            y = (8.4 - k*0.5) if top else (1.6 + k*0.5)
            draw_checker(x, y, is_white)
        if count > 6:
            ax.text(bar_x+0.25, (8.4 - 6*0.5) if top else (1.6 + 6*0.5),
                    f"+{count-6}", ha="center", va="center",
                    fontsize=10, color="#ffffff" if not is_white else "#000000", fontweight="bold")

    if bar_white:
        draw_bar_stack(bar_white, True, top=False)
    if bar_black:
        draw_bar_stack(bar_black, False, top=True)

    ax.text(1.1, 9.0, f"OFF W: {off_white}", color="#f4f4f4", fontsize=10, fontweight="bold")
    ax.text(12.35, 1.0, f"OFF B: {off_black}", color="#f4f4f4", fontsize=10, fontweight="bold")

    if dice is not None:
        d1, d2 = dice
        def draw_die(x, y, val):
            ax.add_patch(FancyBboxPatch((x, y), 0.6, 0.6,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor="#eaeaea", edgecolor="#222", linewidth=1.5))
            ax.text(x+0.3, y+0.3, str(val), ha="center", va="center", fontsize=12, fontweight="bold", color="#222")
        draw_die(7.65, 5.25, d1)
        draw_die(8.35, 5.25, d2)

    if title:
        ax.text(7, 9.75, title, ha="center", va="center", fontsize=13, fontweight="bold", color="#f4f4f4")

    fig.tight_layout(pad=0.2)
    return fig


# =============================
# Game Logic (your code, cleaned from IPython)
# =============================
@dataclass
class State:
    board: List[int]
    bar_w: int = 0
    bar_b: int = 0
    off_w: int = 0
    off_b: int = 0
    turn: int = WHITE

def initial_state() -> State:
    board = [0]*25
    board[1]  = +2
    board[12] = +5
    board[17] = +3
    board[19] = +5

    board[24] = -2
    board[13] = -5
    board[8]  = -3
    board[6]  = -5
    return State(board=board, turn=WHITE)

def roll_dice() -> List[int]:
    a = random.randint(1,6)
    b = random.randint(1,6)
    return [a,a,a,a] if a==b else [a,b]

def opponent(player:int)->int:
    return BLACK if player==WHITE else WHITE

def bar_count(s:State, player:int)->int:
    return s.bar_w if player==WHITE else s.bar_b

def set_bar(s:State, player:int, val:int)->None:
    if player==WHITE: s.bar_w = val
    else: s.bar_b = val

def off_count(s:State, player:int)->int:
    return s.off_w if player==WHITE else s.off_b

def add_off(s:State, player:int, k:int=1)->None:
    if player==WHITE: s.off_w += k
    else: s.off_b += k

def is_blocked(s:State, player:int, point:int)->bool:
    v = s.board[point]
    return (v * opponent(player)) >= 2

def is_blot_of_opponent(s:State, player:int, point:int)->bool:
    v = s.board[point]
    return (v * opponent(player)) == 1

def direction(player:int)->int:
    return 1 if player==WHITE else -1

def entry_point(player:int, die:int)->int:
    return die if player==WHITE else (25 - die)

def all_in_home(s:State, player:int)->bool:
    if bar_count(s, player) > 0:
        return False
    if player == WHITE:
        for p in range(1,19):
            if s.board[p] > 0: return False
        return True
    else:
        for p in range(7,25):
            if s.board[p] < 0: return False
        return True

def highest_occupied_home_point(s:State, player:int)->Optional[int]:
    if player == WHITE:
        for p in range(24,18,-1):
            if s.board[p] > 0:
                return p
    else:
        for p in range(1,7):
            if s.board[p] < 0:
                return p
    return None

def any_behind_in_home(s:State, player:int, from_point:int)->bool:
    if player == WHITE:
        for p in range(19, from_point):
            if s.board[p] > 0:
                return True
        return False
    else:
        for p in range(from_point+1, 7):
            if s.board[p] < 0:
                return True
        return False

Move = Tuple[int,int]

def legal_single_moves(s:State, player:int, die:int) -> List[Move]:
    moves = []
    dirr = direction(player)

    if bar_count(s, player) > 0:
        tp = entry_point(player, die)
        if not is_blocked(s, player, tp):
            moves.append((BAR, tp))
        return moves

    for fp in range(1,25):
        v = s.board[fp]
        if v * player <= 0:
            continue

        tp = fp + dirr*die
        if 1 <= tp <= 24:
            if not is_blocked(s, player, tp):
                moves.append((fp, tp))
        else:
            if all_in_home(s, player):
                if (player == WHITE and tp == 25) or (player == BLACK and tp == 0):
                    moves.append((fp, OFF))
                else:
                    if player == WHITE:
                        if fp == highest_occupied_home_point(s, WHITE) and not any_behind_in_home(s, WHITE, fp):
                            moves.append((fp, OFF))
                    else:
                        if fp == highest_occupied_home_point(s, BLACK) and not any_behind_in_home(s, BLACK, fp):
                            moves.append((fp, OFF))
    return moves

def apply_move(s:State, player:int, move:Move) -> None:
    fp, tp = move

    if fp == BAR:
        set_bar(s, player, bar_count(s, player)-1)
    else:
        s.board[fp] -= player

    if tp == OFF:
        add_off(s, player, 1)
        return

    if is_blot_of_opponent(s, player, tp):
        s.board[tp] -= opponent(player)
        set_bar(s, opponent(player), bar_count(s, opponent(player)) + 1)

    s.board[tp] += player

def terminal(s:State)->bool:
    return s.off_w >= 15 or s.off_b >= 15

def winner(s:State)->Optional[int]:
    if s.off_w >= 15: return WHITE
    if s.off_b >= 15: return BLACK
    return None

def generate_sequences(s:State, player:int, dice:List[int]) -> List[List[Move]]:
    orders = [dice]
    if len(dice)==2 and dice[0]!=dice[1]:
        orders = [dice, [dice[1], dice[0]]]

    all_seqs = []
    for ord_dice in orders:
        seqs = [[]]
        temp_states = [deepcopy(s)]
        for die in ord_dice:
            new_seqs = []
            new_states = []
            for base_seq, base_state in zip(seqs, temp_states):
                legal = legal_single_moves(base_state, player, die)
                if not legal:
                    new_seqs.append(base_seq)
                    new_states.append(base_state)
                else:
                    for mv in legal:
                        ns = deepcopy(base_state)
                        apply_move(ns, player, mv)
                        new_seqs.append(base_seq + [mv])
                        new_states.append(ns)
            seqs, temp_states = new_seqs, new_states
        all_seqs.extend(seqs)

    max_len = max((len(x) for x in all_seqs), default=0)
    best = [x for x in all_seqs if len(x)==max_len]

    uniq = []
    seen = set()
    for seq in best:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            uniq.append(seq)
    return uniq

def pip_count(s:State, player:int)->int:
    total = 0
    if player == WHITE:
        for p in range(1,25):
            if s.board[p] > 0:
                total += s.board[p]*(25-p)
        total += s.bar_w * 25
    else:
        for p in range(1,25):
            if s.board[p] < 0:
                total += (-s.board[p])*p
        total += s.bar_b * 25
    return total

def count_blots(s:State, player:int)->int:
    bl = 0
    for p in range(1,25):
        v = s.board[p]*player
        if v == 1:
            bl += 1
    return bl

def made_points(s:State, player:int)->int:
    mp = 0
    for p in range(1,25):
        v = s.board[p]*player
        if v >= 2:
            mp += 1
    return mp

def anchors_in_opponent_home(s:State, player:int)->int:
    anc = 0
    rng = range(1,7) if player == WHITE else range(19,25)
    for p in rng:
        if s.board[p]*player >= 2:
            anc += 1
    return anc

def contact_phase(s:State)->bool:
    white_points = [p for p in range(1,25) if s.board[p] > 0]
    black_points = [p for p in range(1,25) if s.board[p] < 0]
    if not white_points or not black_points:
        return False
    min_w = min(white_points)
    max_b = max(black_points)
    return min_w <= max_b

def evaluate(s:State, player:int, style:float)->float:
    opp = opponent(player)

    my_pip  = pip_count(s, player)
    op_pip  = pip_count(s, opp)
    my_bl   = count_blots(s, player)
    op_bl   = count_blots(s, opp)
    my_mp   = made_points(s, player)
    my_anc  = anchors_in_opponent_home(s, player)

    my_bar  = bar_count(s, player)
    op_bar  = bar_count(s, opp)

    my_off  = off_count(s, player)
    op_off  = off_count(s, opp)

    phase_contact = contact_phase(s)
    w_pip   = 1.0
    w_off   = 12.0
    w_bar   = 10.0
    w_hit   = 9.0
    w_blots = 6.0
    w_mp    = 2.5
    w_anc   = 3.0

    if not phase_contact:
        w_hit *= 0.2
        w_blots *= 0.4
        w_mp *= 0.7
        w_anc *= 0.4
        w_pip *= 1.6
        w_off *= 14.0

    off_mult = style
    def_mult = 1.0 - style

    score = 0.0
    score += w_off * (my_off - op_off)
    score += (-w_pip) * (my_pip - op_pip)
    score += (-w_bar) * (my_bar) + (w_hit) * (op_bar)

    score += (-w_blots) * (def_mult*1.2 + off_mult*0.4) * my_bl
    score += (w_blots)  * (off_mult*0.8 + def_mult*0.3) * op_bl

    score += (w_mp)  * (off_mult*1.0 + def_mult*0.6) * my_mp
    score += (w_anc) * (def_mult*1.0 + off_mult*0.4) * my_anc

    return score

def choose_style(s:State, player:int)->float:
    opp = opponent(player)
    diff = pip_count(s, player) - pip_count(s, opp)  # positive = behind
    phase_contact = contact_phase(s)

    if diff > 15:
        style = 0.8
    elif diff < -15:
        style = 0.25
    else:
        style = 0.5

    if phase_contact and bar_count(s, opp) > 0:
        style = max(style, 0.75)

    if bar_count(s, player) > 0:
        style = min(style, 0.45)

    return max(0.0, min(1.0, style))

def ai_pick_sequence(s:State, player:int, dice:List[int]) -> List[Move]:
    seqs = generate_sequences(s, player, dice)
    if not seqs:
        return []
    style = choose_style(s, player)

    best_seq = None
    best_val = -1e18
    for seq in seqs:
        ns = deepcopy(s)
        for mv in seq:
            apply_move(ns, player, mv)
        val = evaluate(ns, player, style)
        if val > best_val:
            best_val = val
            best_seq = seq

    return best_seq if best_seq is not None else []

def apply_sequence(s:State, player:int, seq:List[Move])->None:
    for mv in seq:
        apply_move(s, player, mv)

def pretty_move(mv:Move)->str:
    fp,tp = mv
    a = "BAR" if fp==BAR else str(fp)
    b = "OFF" if tp==OFF else str(tp)
    return f"{a}->{b}"


# =============================
# Streamlit App
# =============================
st.set_page_config(page_title="Backgammon AI vs AI", layout="wide")
st.title("Backgammon AI vs AI")
st.caption("Web demo for portfolio — White: Yonatan | Black: Michael")

if "state" not in st.session_state:
    st.session_state.state = initial_state()
if "turn_idx" not in st.session_state:
    st.session_state.turn_idx = 1
if "last_dice" not in st.session_state:
    st.session_state.last_dice = (1, 1)
if "last_title" not in st.session_state:
    st.session_state.last_title = "Ready"

colA, colB, colC = st.columns([1,1,2])

with colA:
    if st.button("Reset Game"):
        st.session_state.state = initial_state()
        st.session_state.turn_idx = 1
        st.session_state.last_title = "Ready"
        st.session_state.last_dice = (1, 1)
        st.rerun()

with colB:
    speed_turn = st.slider("Pause between turns (sec)", 0.0, 3.0, 1.2, 0.1)
    speed_move = st.slider("Pause between moves (sec)", 0.0, 2.0, 0.6, 0.1)

with colC:
    show_each_move = st.checkbox("Show each move (slower, clearer)", value=True)
    max_turns = st.number_input("Max turns", min_value=10, max_value=1000, value=250, step=10)

left, right = st.columns([2,1])
board_area = left.empty()
log_area = right.empty()

def render_board(title: str, dice_show: tuple):
    s = st.session_state.state
    fig = draw_bg_felt(
        s.board,
        title=title,
        dice=dice_show,
        bar_white=s.bar_w, bar_black=s.bar_b,
        off_white=s.off_w, off_black=s.off_b
    )
    board_area.pyplot(fig, clear_figure=True)
    plt.close(fig)

def step_one_turn():
    s = st.session_state.state
    if terminal(s) or st.session_state.turn_idx > max_turns:
        w = winner(s)
        st.session_state.last_title = f"Game Over | Winner: {player_name(w) if w else 'None (turn limit)'}"
        return

    player = s.turn
    dice_list = roll_dice()
    dice_show = (dice_list[0], dice_list[1]) if len(dice_list) == 2 else (dice_list[0], dice_list[0])

    style = choose_style(s, player)
    mode = "OFFENSE" if style >= 0.6 else ("DEFENSE" if style <= 0.35 else "BALANCED")

    title_base = f"Turn {st.session_state.turn_idx} | Now Playing: {player_name(player)} | Rolled: {dice_show} | {mode}"
    seq = ai_pick_sequence(s, player, dice_list)

    if show_each_move and seq:
        st.session_state.last_title = title_base + " | thinking..."
        st.session_state.last_dice = dice_show
        render_board(st.session_state.last_title, st.session_state.last_dice)
        time.sleep(speed_move)

        for mv in seq:
            apply_move(s, player, mv)
            st.session_state.last_title = title_base + f" | Move: {pretty_move(mv)}"
            st.session_state.last_dice = dice_show
            render_board(st.session_state.last_title, st.session_state.last_dice)
            time.sleep(speed_move)

        time.sleep(speed_turn)
    else:
        apply_sequence(s, player, seq)
        moves_txt = ", ".join(pretty_move(m) for m in seq) if seq else "NO MOVES"
        st.session_state.last_title = title_base + f" | {moves_txt}"
        st.session_state.last_dice = dice_show
        render_board(st.session_state.last_title, st.session_state.last_dice)
        time.sleep(speed_turn)

    s.turn = opponent(s.turn)
    st.session_state.turn_idx += 1

run_col1, run_col2 = st.columns([1,1])

with run_col1:
    if st.button("Run 1 Turn"):
        step_one_turn()
        st.rerun()

with run_col2:
    auto_run = st.button("Auto Run (limited)")

# Always render current state
render_board(st.session_state.last_title, st.session_state.last_dice)

# Simple log panel
s = st.session_state.state
with log_area:
    st.subheader("Status")
    st.write(f"Turn: {st.session_state.turn_idx}")
    st.write(f"On bar — White: {s.bar_w} | Black: {s.bar_b}")
    st.write(f"Off — White: {s.off_w} | Black: {s.off_b}")
    if terminal(s):
        w = winner(s)
        st.success(f"Winner: {player_name(w) if w else 'None'}")

# Auto-run in this session (kept short to avoid freezing)
if auto_run:
    steps = 25  # keep it small per click
    for _ in range(steps):
        if terminal(st.session_state.state) or st.session_state.turn_idx > max_turns:
            break
        step_one_turn()
        render_board(st.session_state.last_title, st.session_state.last_dice)
    st.rerun()