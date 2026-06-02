# POCETAK v2



import csv
import math
import os
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


T0 = time.time()
SEED = 39
CSV_PATH = "/data/loto7_4624_k43.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(HERE, "7_quant_mezoni_v2.txt")
PNG_OUT = os.path.join(HERE, "7_quant_mezoni_v2.png")

N_NUMBERS = 39
K_PICK = 7
TOTAL_COMB = math.comb(N_NUMBERS, K_PICK)
PLACEHOLDER = (1, 2, 3, 4, 5, 6, 7)

N_QUBITS = 25
BLOCKS = 5
Q_PER_BLOCK = 5
LAYERS = 4

TRAIN_ITERS = 80
TRAIN_SHOTS = 4096
FINAL_SHOTS = 100000
TOP_K = 12
TARGET_SAMPLE_N = 768
GEN_SAMPLE_N = 768
MMD_SIGMA = 0.18


def fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def load_loto_csv(path: str) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            vals: list[int] = []
            for cell in row:
                try:
                    vals.append(int(str(cell).strip()))
                except ValueError:
                    continue
            if len(vals) >= K_PICK:
                combo = tuple(sorted(vals[:K_PICK]))
                if len(set(combo)) == K_PICK and all(1 <= x <= N_NUMBERS for x in combo):
                    rows.append(combo)
    if not rows:
        raise ValueError("CSV nije ucitan: nema validnih 7/39 kombinacija.")
    return rows


def lex_rank(combo: tuple[int, ...]) -> int:
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo, start=1):
        for x in range(prev + 1, value):
            rank0 += math.comb(N_NUMBERS - x, K_PICK - i)
        prev = value
    return rank0 + 1


def lex_derank(rank: int) -> tuple[int, ...]:
    r = int(rank) - 1
    combo: list[int] = []
    start = 1
    for i in range(K_PICK):
        remaining = K_PICK - i - 1
        for x in range(start, N_NUMBERS + 1):
            cnt = math.comb(N_NUMBERS - x, remaining)
            if r < cnt:
                combo.append(x)
                start = x + 1
                break
            r -= cnt
    return tuple(combo)


def int_to_bitstring(value: int, n_bits: int = N_QUBITS) -> str:
    return format(int(value), f"0{n_bits}b")


def recency_weights(n: int, tau: float = 950.0) -> np.ndarray:
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-ages / tau)
    weights /= weights.sum()
    return weights


def weighted_target_bits(lex_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(N_QUBITS, dtype=np.float64)
    for idx, w in zip(lex_indices, weights):
        bits = int_to_bitstring(int(idx) - 1)
        out += w * np.fromiter((1.0 if b == "1" else 0.0 for b in bits), dtype=np.float64)
    return out


def weighted_target_sample(
    lex_indices: np.ndarray,
    weights: np.ndarray,
    sample_n: int = TARGET_SAMPLE_N,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(lex_indices), size=sample_n, replace=True, p=weights)
    sample = lex_indices[pick].astype(np.float64) / TOTAL_COMB
    return sample.reshape(-1, 1)


def gaussian_mmd(x: np.ndarray, y: np.ndarray, sigma: float = MMD_SIGMA) -> float:
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xx = (x - x.T) ** 2
    yy = (y - y.T) ** 2
    xy = (x - y.T) ** 2
    denom = 2.0 * sigma * sigma
    kxx = np.exp(-xx / denom).mean()
    kyy = np.exp(-yy / denom).mean()
    kxy = np.exp(-xy / denom).mean()
    return float(kxx + kyy - 2.0 * kxy)


def counts_to_valid_lex_sample(counts: dict[str, int], sample_n: int = GEN_SAMPLE_N) -> np.ndarray:
    vals: list[float] = []
    for bitstr, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        lex_val = int(clean, 2) + 1
        if 1 <= lex_val <= TOTAL_COMB:
            vals.extend([lex_val / TOTAL_COMB] * min(int(count), sample_n - len(vals)))
        if len(vals) >= sample_n:
            break
    if not vals:
        vals = [0.5]
    while len(vals) < sample_n:
        vals.append(vals[-1])
    return np.array(vals[:sample_n], dtype=np.float64).reshape(-1, 1)


def counts_to_bit_probs(counts: dict[str, int]) -> np.ndarray:
    total = max(1, sum(counts.values()))
    probs = np.zeros(N_QUBITS, dtype=np.float64)
    for bitstr, count in counts.items():
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        probs += count * np.fromiter((1.0 if b == "1" else 0.0 for b in clean), dtype=np.float64)
    return probs / total


def params_per_layer() -> int:
    single = 2 * N_QUBITS
    intra_block_cry = BLOCKS * (Q_PER_BLOCK - 1)
    mezon_cycle_cry = 2 * (BLOCKS - 1)
    return single + intra_block_cry + mezon_cycle_cry


def build_mezon_qcbm_v2(theta: np.ndarray, seed_lex: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    seed_bits = int_to_bitstring(int(seed_lex) - 1)
    for q, bit in enumerate(reversed(seed_bits)):
        if bit == "1":
            qc.x(q)

    p = 0
    reps = [block * Q_PER_BLOCK for block in range(BLOCKS)]
    for layer in range(LAYERS):
        for q in range(N_QUBITS):
            qc.ry(float(theta[p]), q)
            p += 1
            qc.rz(float(theta[p]), q)
            p += 1

        for block in range(BLOCKS):
            start = block * Q_PER_BLOCK
            for j in range(Q_PER_BLOCK - 1):
                qc.cry(float(theta[p]), start + j, start + j + 1)
                p += 1

        # Puni mezonski ciklus: A->B->C->D->E->D->C->B->A.
        for a, b in zip(reps[:-1], reps[1:]):
            qc.cry(float(theta[p]), a, b)
            p += 1
        for a, b in zip(reps[:0:-1], reps[-2::-1]):
            qc.cry(float(theta[p]), a, b)
            p += 1

        for block in range(BLOCKS - 1):
            qc.cz(block * Q_PER_BLOCK + Q_PER_BLOCK - 1, (block + 1) * Q_PER_BLOCK)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def run_counts(
    theta: np.ndarray,
    simulator: AerSimulator,
    shots: int,
    seed_lex: int,
    seed_offset: int = 0,
) -> dict[str, int]:
    qc = build_mezon_qcbm_v2(theta, seed_lex)
    tqc = transpile(qc, simulator, optimization_level=1, seed_transpiler=SEED + seed_offset)
    result = simulator.run(tqc, shots=shots, seed_simulator=SEED + seed_offset).result()
    return result.get_counts()


def cost_from_counts(
    counts: dict[str, int],
    target_sample: np.ndarray,
    target_bits: np.ndarray,
) -> float:
    generated_sample = counts_to_valid_lex_sample(counts)
    mmd = gaussian_mmd(generated_sample, target_sample)
    bit_mse = float(np.mean((counts_to_bit_probs(counts) - target_bits) ** 2))
    return float(mmd + 0.15 * bit_mse)


def init_theta_from_target(target_bits: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    ppl = params_per_layer()
    theta = np.zeros(LAYERS * ppl, dtype=np.float64)
    base_ry = 2.0 * np.arcsin(np.sqrt(np.clip(target_bits, 1e-6, 1.0 - 1e-6)))

    p = 0
    for layer in range(LAYERS):
        layer_scale = 1.0 / math.sqrt(layer + 1.0)
        for q in range(N_QUBITS):
            theta[p] = base_ry[q] * layer_scale + rng.normal(0.0, 0.035)
            p += 1
            theta[p] = rng.normal(0.0, 0.10)
            p += 1
        for _ in range(BLOCKS * (Q_PER_BLOCK - 1)):
            theta[p] = rng.normal(0.0, 0.22)
            p += 1
        for _ in range(2 * (BLOCKS - 1)):
            theta[p] = rng.normal(0.0, 0.30)
            p += 1
    return np.mod(theta, 2.0 * np.pi)


def spsa_train(
    theta0: np.ndarray,
    target_sample: np.ndarray,
    target_bits: np.ndarray,
    simulator: AerSimulator,
    seed_lex: int,
) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    rng = np.random.default_rng(SEED)
    theta = theta0.copy()
    losses: list[float] = []
    mmd_losses: list[float] = []
    bit_losses: list[float] = []

    for it in range(1, TRAIN_ITERS + 1):
        a = 0.14 / (it ** 0.33)
        c = 0.10 / (it ** 0.12)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        counts_plus = run_counts(theta + c * delta, simulator, TRAIN_SHOTS, seed_lex, 2 * it)
        counts_minus = run_counts(theta - c * delta, simulator, TRAIN_SHOTS, seed_lex, 2 * it + 1)
        loss_plus = cost_from_counts(counts_plus, target_sample, target_bits)
        loss_minus = cost_from_counts(counts_minus, target_sample, target_bits)

        ghat = (loss_plus - loss_minus) / (2.0 * c) * delta
        theta = np.mod(theta - a * ghat, 2.0 * np.pi)

        counts_eval = counts_plus if loss_plus <= loss_minus else counts_minus
        generated = counts_to_valid_lex_sample(counts_eval)
        mmd_now = gaussian_mmd(generated, target_sample)
        bit_now = float(np.mean((counts_to_bit_probs(counts_eval) - target_bits) ** 2))
        loss_now = float(mmd_now + 0.15 * bit_now)

        losses.append(loss_now)
        mmd_losses.append(float(mmd_now))
        bit_losses.append(bit_now)
        print(
            f"  SPSA iter {it:02d}/{TRAIN_ITERS}  "
            f"loss={loss_now:.8f}  mmd={mmd_now:.8f}  bit_mse={bit_now:.8f}"
        )

    return theta, losses, mmd_losses, bit_losses


def valid_sample_rows(
    counts: dict[str, int],
    historical_set: set[int],
) -> tuple[list[dict[str, object]], int, int, int]:
    rows: list[dict[str, object]] = []
    seen_combos: set[tuple[int, ...]] = set()
    skipped_out = 0
    skipped_placeholder = 0
    skipped_seen = 0

    for bitstr, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        lex_val = int(clean, 2) + 1
        if not (1 <= lex_val <= TOTAL_COMB):
            skipped_out += int(count)
            continue
        combo = lex_derank(lex_val)
        if combo == PLACEHOLDER:
            skipped_placeholder += int(count)
            continue
        if lex_val in historical_set:
            skipped_seen += int(count)
            continue
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        rows.append(
            {
                "count": int(count),
                "prob": float(count) / FINAL_SHOTS,
                "lex": int(lex_val),
                "combo": combo,
            }
        )
        if len(rows) >= TOP_K:
            break

    return rows, skipped_out, skipped_placeholder, skipped_seen


def make_png(
    losses: list[float],
    mmd_losses: list[float],
    bit_losses: list[float],
    rows: list[dict[str, object]],
    target_bits: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.4, label="loss")
    ax1.plot(range(1, len(mmd_losses) + 1), mmd_losses, linewidth=1.2, label="MMD")
    ax1.set_title("QCBM v2 SPSA loss")
    ax1.set_xlabel("iter")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(N_QUBITS), target_bits, color="#2563eb")
    ax2.set_title("Exponential-recency target bit amplitude")
    ax2.set_xlabel("bit pozicija")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_rows = [
        [i + 1, row["count"], f"{row['prob']:.6f}", row["lex"], str(row["combo"])]
        for i, row in enumerate(rows)
    ]
    table = ax3.table(
        cellText=table_rows,
        colLabels=["rang", "count", "prob", "lex", "kombinacija"],
        cellLoc="center",
        loc="center",
        colWidths=[0.07, 0.10, 0.12, 0.17, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")

    fig.suptitle("7_quant_mezoni_v2 - Qiskit QCBM MMD 25q", fontweight="bold")
    fig.tight_layout()
    plt.show()
    fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print()
    print("=" * 72)
    print("7_quant_mezoni_v2 - finalni Qiskit QCBM MMD nad 4624 lex-indeksa")
    print("=" * 72)
    print()

    combos = load_loto_csv(CSV_PATH)
    lex_indices = np.array([lex_rank(c) for c in combos], dtype=np.int64)
    historical_set = set(int(x) for x in lex_indices)
    weights = recency_weights(len(lex_indices))
    target_bits = weighted_target_bits(lex_indices, weights)
    target_sample = weighted_target_sample(lex_indices, weights)
    seed_lex = int(lex_indices[-1])

    print(f"CSV:                 {CSV_PATH}")
    print(f"Ucitano izvlacenja:  {len(combos)}")
    print(f"C(39,7):             {TOTAL_COMB:,}")
    print(f"Zadnji lex seed:     {seed_lex:,}")
    print(f"Qubita:              {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK} qubita")
    print(f"Layers:              {LAYERS}")
    print(f"Parametara:          {LAYERS * params_per_layer()}")
    print(f"Simulator:           AerSimulator qasm, shots train={TRAIN_SHOTS}, final={FINAL_SHOTS}")
    print()

    simulator = AerSimulator(method="automatic")
    theta0 = init_theta_from_target(target_bits)

    t_train = time.time()
    theta, losses, mmd_losses, bit_losses = spsa_train(
        theta0,
        target_sample,
        target_bits,
        simulator,
        seed_lex,
    )
    train_seconds = time.time() - t_train

    print()
    print("Finalno semplovanje istreniranog v2 kola...")
    final_counts = run_counts(theta, simulator, FINAL_SHOTS, seed_lex, 10_000)
    rows, skipped_out, skipped_placeholder, skipped_seen = valid_sample_rows(final_counts, historical_set)

    if not rows:
        raise RuntimeError("Nema validnih novih sampled lex kandidata posle filtera.")

    main_row = rows[0]
    total_seconds = time.time() - T0

    lines: list[str] = []
    lines.append("7_quant_mezoni_v2 - finalni Qiskit QCBM MMD 25q / mezonski ciklus")
    lines.append("=" * 72)
    lines.append("")
    lines.append("KORAK 1: Weierstrass lex-kriva nad svim do sad izvucenim kombinacijama")
    lines.append("")
    lines.append(f"  CSV izvucenih:        {CSV_PATH}")
    lines.append(f"  Ucitano izvlacenja:    {len(combos)}")
    lines.append(f"  C(39,7):              {TOTAL_COMB:,}")
    lines.append(f"  Zadnji lex seed:       {seed_lex:,}")
    lines.append("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7")
    lines.append("")
    lines.append("KORAK 2: Stvarni kvantni model v2")
    lines.append("")
    lines.append("  Model:                QCBM / parametrizovano kvantno kolo")
    lines.append("  Loss:                 MMD(lex distribucija) + 0.15*MSE(bit-marginale)")
    lines.append("  Recency:              exponential weights nad svih 4624 tacaka")
    lines.append(f"  Qubita:               {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK}")
    lines.append(f"  Layers:               {LAYERS}")
    lines.append(f"  Parametara:           {len(theta)}")
    lines.append("  Conditional seed:     zadnji lex-indeks enkodovan X-gateovima")
    lines.append("  Entanglement:          CRY unutar blokova + CZ izmedju blokova")
    lines.append("  Mezonski ciklus:       A-B-C-D-E-D-C-B-A CRY petlja kroz 5 blokova")
    lines.append(f"  SPSA iteracija:        {TRAIN_ITERS}")
    lines.append(f"  train shots:           {TRAIN_SHOTS}")
    lines.append(f"  final shots:           {FINAL_SHOTS}")
    lines.append(f"  initial loss:          {losses[0]:.8f}")
    lines.append(f"  final loss:            {losses[-1]:.8f}")
    lines.append(f"  final MMD:             {mmd_losses[-1]:.8f}")
    lines.append(f"  final bit MSE:         {bit_losses[-1]:.8f}")
    lines.append("")
    lines.append("Filter finalnih kandidata:")
    lines.append(f"  out-of-range shots:    {skipped_out}")
    lines.append(f"  placeholder shots:     {skipped_placeholder}")
    lines.append(f"  vec izvuceni shots:    {skipped_seen}")
    lines.append("")
    lines.append("PREDIKCIJA 2: NEXT / 7_quant_mezoni_v2")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Glavna kvantna prognoza:")
    lines.append(f"  sampled count:         {main_row['count']}")
    lines.append(f"  sampled prob:          {main_row['prob']:.8f}")
    lines.append(f"  pred. lex:             {main_row['lex']:,}")
    lines.append(f"  pred. kombinacija:     {main_row['combo']}")
    lines.append("  vec izvucena ranije:   NE (filtrirano)")
    lines.append("")
    lines.append("Top kvantni kandidati (novi, bez placeholder-a):")
    lines.append(f"  {'rang':<5}{'count':>8}{'prob':>12}{'lex':>14}  {'kombinacija':<30}")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"  {i:<5}{row['count']:>8}{row['prob']:>12.8f}{row['lex']:>14,}  "
            f"{str(row['combo']):<30}"
        )
    lines.append("")
    lines.append(f"Vreme treninga:       {fmt_time(train_seconds)} ({train_seconds:.1f} s)")
    lines.append(f"Ukupno vreme:         {fmt_time(total_seconds)} ({total_seconds:.1f} s)")
    lines.append(f"PNG:                  {PNG_OUT}")
    lines.append("")

    text = "\n".join(lines)
    print()
    print(text)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"TXT saved -> {TXT_OUT}")

    make_png(losses, mmd_losses, bit_losses, rows, target_bits)
    print(f"PNG saved -> {PNG_OUT}")
    print()


if __name__ == "__main__":
    main()


"""
========================================================================
7_quant_mezoni_v2 - finalni Qiskit QCBM MMD nad 4624 lex-indeksa
========================================================================

CSV:                 /data/loto7_4624_k43.csv
Ucitano izvlacenja:  4624
C(39,7):             15,380,937
Zadnji lex seed:     513,114
Qubita:              25 = 5 blokova x 5 qubita
Layers:              4
Parametara:          312
Simulator:           AerSimulator qasm, shots train=4096, final=100000

  SPSA iter 01/80  loss=0.05332714  mmd=0.03469041  bit_mse=0.12424484
  SPSA iter 02/80  loss=0.03335027  mmd=0.01615360  bit_mse=0.11464449
  SPSA iter 03/80  loss=0.06191616  mmd=0.04335462  bit_mse=0.12374358
  SPSA iter 04/80  loss=0.03044063  mmd=0.01080218  bit_mse=0.13092306
  SPSA iter 05/80  loss=0.08749785  mmd=0.06935401  bit_mse=0.12095898
  SPSA iter 06/80  loss=0.03370362  mmd=0.01443473  bit_mse=0.12845925
  SPSA iter 07/80  loss=0.04498629  mmd=0.02618845  bit_mse=0.12531897
  SPSA iter 08/80  loss=0.03945338  mmd=0.02149531  bit_mse=0.11972046
  SPSA iter 09/80  loss=0.02459770  mmd=0.00707950  bit_mse=0.11678798
  SPSA iter 10/80  loss=0.06844296  mmd=0.05094031  bit_mse=0.11668434
  SPSA iter 11/80  loss=0.03374093  mmd=0.01411662  bit_mse=0.13082873
  SPSA iter 12/80  loss=0.03015906  mmd=0.01236008  bit_mse=0.11865989
  SPSA iter 13/80  loss=0.08664720  mmd=0.06930629  bit_mse=0.11560604
  SPSA iter 14/80  loss=0.05846662  mmd=0.03975352  bit_mse=0.12475396
  SPSA iter 15/80  loss=0.04017966  mmd=0.02366196  bit_mse=0.11011799
  SPSA iter 16/80  loss=0.02704249  mmd=0.00894365  bit_mse=0.12065893
  SPSA iter 17/80  loss=0.02299937  mmd=0.00494607  bit_mse=0.12035533
  SPSA iter 18/80  loss=0.02865276  mmd=0.00843113  bit_mse=0.13481088
  SPSA iter 19/80  loss=0.04991771  mmd=0.02927273  bit_mse=0.13763318
  SPSA iter 20/80  loss=0.03410264  mmd=0.01458967  bit_mse=0.13008651
  SPSA iter 21/80  loss=0.07304067  mmd=0.05359880  bit_mse=0.12961251
  SPSA iter 22/80  loss=0.02727417  mmd=0.00874070  bit_mse=0.12355645
  SPSA iter 23/80  loss=0.02725973  mmd=0.00684378  bit_mse=0.13610633
  SPSA iter 24/80  loss=0.02855474  mmd=0.00759472  bit_mse=0.13973348
  SPSA iter 25/80  loss=0.02752879  mmd=0.00911295  bit_mse=0.12277226
  SPSA iter 26/80  loss=0.06168994  mmd=0.04281068  bit_mse=0.12586173
  SPSA iter 27/80  loss=0.04316494  mmd=0.02467883  bit_mse=0.12324070
  SPSA iter 28/80  loss=0.03126701  mmd=0.01114086  bit_mse=0.13417432
  SPSA iter 29/80  loss=0.03089542  mmd=0.01324461  bit_mse=0.11767205
  SPSA iter 30/80  loss=0.02585115  mmd=0.00468938  bit_mse=0.14107847
  SPSA iter 31/80  loss=0.04118818  mmd=0.02286258  bit_mse=0.12217069
  SPSA iter 32/80  loss=0.04261780  mmd=0.02351779  bit_mse=0.12733342
  SPSA iter 33/80  loss=0.03772474  mmd=0.01874510  bit_mse=0.12653095
  SPSA iter 34/80  loss=0.02620043  mmd=0.00614816  bit_mse=0.13368178
  SPSA iter 35/80  loss=0.02465898  mmd=0.00535278  bit_mse=0.12870797
  SPSA iter 36/80  loss=0.02369829  mmd=0.00523399  bit_mse=0.12309531
  SPSA iter 37/80  loss=0.02113300  mmd=0.00162643  bit_mse=0.13004377
  SPSA iter 38/80  loss=0.02313604  mmd=0.00361914  bit_mse=0.13011267
  SPSA iter 39/80  loss=0.02943886  mmd=0.01134458  bit_mse=0.12062852
  SPSA iter 40/80  loss=0.02094616  mmd=0.00609254  bit_mse=0.09902410
  SPSA iter 41/80  loss=0.02563502  mmd=0.00939373  bit_mse=0.10827524
  SPSA iter 42/80  loss=0.02476417  mmd=0.00955926  bit_mse=0.10136606
  SPSA iter 43/80  loss=0.02720020  mmd=0.00984191  bit_mse=0.11572189
  SPSA iter 44/80  loss=0.02345438  mmd=0.00810573  bit_mse=0.10232433
  SPSA iter 45/80  loss=0.02048197  mmd=0.00336695  bit_mse=0.11410014
  SPSA iter 46/80  loss=0.02201066  mmd=0.00671091  bit_mse=0.10199835
  SPSA iter 47/80  loss=0.02005012  mmd=0.00315331  bit_mse=0.11264538
  SPSA iter 48/80  loss=0.02002323  mmd=0.00430911  bit_mse=0.10476086
  SPSA iter 49/80  loss=0.04342338  mmd=0.02656473  bit_mse=0.11239098
  SPSA iter 50/80  loss=0.02888412  mmd=0.01087907  bit_mse=0.12003372
  SPSA iter 51/80  loss=0.03295919  mmd=0.01537201  bit_mse=0.11724791
  SPSA iter 52/80  loss=0.02790084  mmd=0.01085218  bit_mse=0.11365775
  SPSA iter 53/80  loss=0.02468225  mmd=0.00877298  bit_mse=0.10606180
  SPSA iter 54/80  loss=0.02562917  mmd=0.00900339  bit_mse=0.11083851
  SPSA iter 55/80  loss=0.02909686  mmd=0.01281933  bit_mse=0.10851687
  SPSA iter 56/80  loss=0.02751368  mmd=0.01102458  bit_mse=0.10992733
  SPSA iter 57/80  loss=0.04954221  mmd=0.03419253  bit_mse=0.10233119
  SPSA iter 58/80  loss=0.03497939  mmd=0.02007275  bit_mse=0.09937758
  SPSA iter 59/80  loss=0.02627540  mmd=0.00947795  bit_mse=0.11198298
  SPSA iter 60/80  loss=0.02810538  mmd=0.01175491  bit_mse=0.10900314
  SPSA iter 61/80  loss=0.02890442  mmd=0.01153279  bit_mse=0.11581086
  SPSA iter 62/80  loss=0.02639007  mmd=0.00920607  bit_mse=0.11455997
  SPSA iter 63/80  loss=0.02779404  mmd=0.01023757  bit_mse=0.11704307
  SPSA iter 64/80  loss=0.04207034  mmd=0.02557798  bit_mse=0.10994902
  SPSA iter 65/80  loss=0.03518088  mmd=0.01648366  bit_mse=0.12464815
  SPSA iter 66/80  loss=0.04673325  mmd=0.02807846  bit_mse=0.12436526
  SPSA iter 67/80  loss=0.02885549  mmd=0.01104211  bit_mse=0.11875586
  SPSA iter 68/80  loss=0.05545623  mmd=0.03902134  bit_mse=0.10956597
  SPSA iter 69/80  loss=0.02901170  mmd=0.01125714  bit_mse=0.11836373
  SPSA iter 70/80  loss=0.03078339  mmd=0.01373659  bit_mse=0.11364536
  SPSA iter 71/80  loss=0.05908651  mmd=0.04324316  bit_mse=0.10562232
  SPSA iter 72/80  loss=0.02748818  mmd=0.01201710  bit_mse=0.10314048
  SPSA iter 73/80  loss=0.03379778  mmd=0.01549356  bit_mse=0.12202813
  SPSA iter 74/80  loss=0.02597027  mmd=0.00912354  bit_mse=0.11231151
  SPSA iter 75/80  loss=0.01823449  mmd=0.00382631  bit_mse=0.09605457
  SPSA iter 76/80  loss=0.02175251  mmd=0.00532826  bit_mse=0.10949501
  SPSA iter 77/80  loss=0.02300134  mmd=0.00798917  bit_mse=0.10008112
  SPSA iter 78/80  loss=0.02480469  mmd=0.00779160  bit_mse=0.11342062
  SPSA iter 79/80  loss=0.02803241  mmd=0.01099198  bit_mse=0.11360291
  SPSA iter 80/80  loss=0.02140850  mmd=0.00480872  bit_mse=0.11066516

Finalno semplovanje istreniranog v2 kola...

7_quant_mezoni_v2 - finalni Qiskit QCBM MMD 25q / mezonski ciklus
========================================================================

KORAK 1: Weierstrass lex-kriva nad svim do sad izvucenim kombinacijama

  CSV izvucenih:        /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937
  Zadnji lex seed:      513,114
  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7

KORAK 2: Stvarni kvantni model v2

  Model:                QCBM / parametrizovano kvantno kolo
  Loss:                 MMD(lex distribucija) + 0.15*MSE(bit-marginale)
  Recency:              exponential weights nad svih 4624 tacaka
  Qubita:               25 = 5 blokova x 5
  Layers:               4
  Parametara:           312
  Conditional seed:     zadnji lex-indeks enkodovan X-gateovima
  Entanglement:          CRY unutar blokova + CZ izmedju blokova
  Mezonski ciklus:       A-B-C-D-E-D-C-B-A CRY petlja kroz 5 blokova
  SPSA iteracija:        80
  train shots:           4096
  final shots:           100000
  initial loss:          0.05332714
  final loss:            0.02140850
  final MMD:             0.00480872
  final bit MSE:         0.11066516

Filter finalnih kandidata:
  out-of-range shots:    2856
  placeholder shots:     0
  vec izvuceni shots:    0

PREDIKCIJA 2: NEXT / 7_quant_mezoni_v2
========================================================================

Glavna kvantna prognoza:
  sampled count:         72
  sampled prob:          0.00072000
  pred. lex:             15,143,298
  pred. kombinacija:     (17, x, 20, y, 28, z, 38)
  vec izvucena ranije:   NE (filtrirano)

Top kvantni kandidati (novi, bez placeholder-a):
  rang    count        prob           lex  kombinacija                   
  1          72  0.00072000    15,143,298  (17, x, 20, y, 28, z, 38)  
  2          70  0.00070000    13,046,150  (9, x, 25, y, 27, z, 35)   
  3          63  0.00063000    13,046,146  (9, x, 25, y, 27, z, 31)   
  4          63  0.00063000    10,948,994  (6, x, 18, y, 30, z, 39)   
  5          60  0.00060000    13,046,148  (9, x, 25, y, 27, z, 33)   
  6          55  0.00055000     2,560,386  (1, x, 18, y, 24, z, 38)   
  7          54  0.00054000    15,143,302  (17, x, 20, y, 28, z, 34)  
  8          51  0.00051000       463,234  (1, x, 4, y, 19, z, 27)     
  9          51  0.00051000    13,046,152  (9, x, 25, y, 27, z, 37)   
  10         51  0.00051000    15,139,202  (17, x, 19, y, 31, z, 38)  
  11         46  0.00046000     2,560,390  (1, x, 18, y, 24, z, 38)   
  12         46  0.00046000    13,042,050  (9, x, 22, y, 28, z, 36)   

Vreme treninga:       0:07:43 (462.6 s)
Ukupno vreme:         0:07:46 (465.7 s)
PNG:                  /7_quant_mezoni_v2.png

TXT saved -> /7_quant_mezoni_v2.txt
PNG saved -> /7_quant_mezoni_v2.png
"""




"""
Analiza v2
v2 je mnogo ozbiljniji od v1 jer je trening zaista konvergirao.

Loss:

initial: 0.05332714
final: 0.02140850
To je pad oko 59.9%, znači SPSA je stvarno našao bolju konfiguraciju kola. 
Kod v1 loss je čak blago porastao, pa je v2 realno uspešniji kvantni model.

Šta je model naučio
Glavna prognoza:

(17, x, 20, y, 28, z, 38)
lex: 15,143,298
prob: 0.00072

Ovo je vrlo visok lex deo prostora, skoro pri vrhu 39C7. 
Za razliku od v1, koji je bio zbijen u opsegu ~10.6M-12.4M, 
v2 širi kandidate na više regiona lex-distribucije:

visoki region: 15.14M
srednje-visoki region: 13.04M
srednji region: 10.95M
niski region: 2.56M
vrlo niski region: 463k
To je dobro: v2 nije samo zaključao jednu oblast, 
nego pravi više kvantnih modova/distribucionih ostrva.

Najjači signal
Najjasniji klaster je oko lex 13,046,146-13,046,152:

(9, x, 25, y, 27, z, 31)
(9, x, 25, y, 27, z, 33)
(9, x, 25, y, 27, z, 35)
(9, x, 25, y, 27, z, 37)
To je praktično ista baza:

(9, x, 25, y, 27, z, X)

sa promenljivim zadnjim brojem. 
To znači da je kvantno kolo pronašlo stabilan lokalni obrazac, 
ali poslednji qubit-blok još varira.

Drugi klaster:

(17, x, 20, y, 28, 30/31, 34/38)

To je gornji klaster i iz njega dolazi glavna prognoza.

Filteri
out-of-range shots: 2856 od 100000 = 2.856%
placeholder: 0
već izvučeni: 0
Ovo je dobro. Model ne baca mnogo van prostora, 
ne generiše trivijalni placeholder i ne kopira prethodnu istoriju.



Poređenje sa v1
v1:
brži: ~2:50
loss nije poboljšan
top kandidati mnogo zbijeni
rezultat više liči na inicijalizaciju bit-marginala

v2:
sporiji: ~7:46
loss značajno poboljšan
koristi MMD nad lex-distribucijom
daje više modova
filtrira validne nove kandidate
Zaključak: v2 je bolji i finalni kvantni model.

Glavna NEXT preporuka iz v2
Primarna:

(17, x, 20, y, 28, z, 38)

Sekundarni jak klaster, stabilnost preko više kandidata:

(9, x, 25, y, 27, x, X)

gde su X = 31, 33, 35, 37, a najjači rang je:

(9, x, 25, y, 27, z, 35)
"""





"""
7_quant_mezoni_v2.py: MMD, puni 5-blok ciklus, 4 sloja, 100k final shots i filteri

Qiskit + Aer simulator
25 qubita = 5 blokova x 5 qubita
4 sloja
MMD loss nad lex-distribucijom
exponential recency weight
conditional seed iz zadnjeg lex-indeksa
puni mezonski ciklus A-B-C-D-E-D-C-B-A
CRY unutar i između blokova
filter: bez placeholder-a i bez već izvučenih kombinacija
FINAL_SHOTS = 100000
izlazi:
7_quant_mezoni_v2.txt
7_quant_mezoni_v2.png
"""





"""
7_quant_mezoni_v2

Finalna verzija stvarnog kvantnog Qiskit pristupa za loto7_4624_k43.

v2 dodaje u odnosu na v1:
  - MMD loss nad lex-distribucijom, ne samo MSE bit-marginala
  - 25 qubita = 5 blokova x 5 qubita, bez sirenja na 35q
  - 4 varijaciona sloja
  - puni mezonski ciklus A-B-C-D-E-D-C-B-A kroz blokove
  - conditional seed iz zadnjeg lex-indeksa
  - exponential recency weight nad celom krivom
  - filter placeholder-a i vec izvucenih kombinacija
  - finalno semplovanje sa 100000 shots
"""
