import csv
import math
import sys
from collections import defaultdict, deque

import numpy as np

# SDK cs_de.c と同じ定数
C = 299792458.0
DELTA_F = 1_000_000.0
CHANNEL_INDEX_OFFSET = 2
CS_DE_NUM_CHANNELS = 75
TONE_QUALITY_HIGH = 0

# my_ras_initiator/prj.conf は CONFIG_BT_CS_DE_512_NFFT=y
DEFAULT_NFFT = 512


def _combine_iq_like_sdk(il, ql, ir, qr):
    """SDKのcs_de_combined_iq_calculate()と同じ複素積."""
    il = np.float32(il)
    ql = np.float32(ql)
    ir = np.float32(ir)
    qr = np.float32(qr)

    i_comb = np.float32(il * ir - ql * qr)
    q_comb = np.float32(il * qr + ir * ql)
    return np.complex64(complex(i_comb, q_comb))


def _calculate_ifft_mag(iq_bins):
    """SDKのcalculate_ifft_mag()相当."""
    return np.abs(np.fft.ifft(iq_bins).astype(np.complex64)).astype(np.float32)


def _calculate_ifft_find_left_null(peak_index, mag):
    left_null_index = int(peak_index)
    nfft = len(mag)

    while True:
        next_left_null_index = nfft - 1 if left_null_index == 0 else left_null_index - 1

        if (
            (
                mag[left_null_index] * 2 > mag[peak_index]
                or mag[left_null_index] > np.float32(1.10) * mag[next_left_null_index]
            )
            and mag[left_null_index] * 10 > mag[peak_index]
            and next_left_null_index != peak_index
        ):
            left_null_index = next_left_null_index
        else:
            break

    return left_null_index


def _calculate_distance_to_left_null(peak_index, left_null_index, nfft):
    if left_null_index > peak_index:
        return nfft + peak_index - left_null_index
    return peak_index - left_null_index


def _calculate_left_null_compensation_of_peak(peak_index, mag):
    nfft = len(mag)
    normal_peak_to_null = (nfft + CS_DE_NUM_CHANNELS - 1) // CS_DE_NUM_CHANNELS
    left_null_index = _calculate_ifft_find_left_null(peak_index, mag)
    peak_to_null_distance = _calculate_distance_to_left_null(
        peak_index, left_null_index, nfft
    )

    compensated_peak_index = int(peak_index)

    if peak_to_null_distance > normal_peak_to_null:
        if left_null_index > peak_index:
            candidate = left_null_index + normal_peak_to_null - nfft
            compensated_peak_index = candidate if candidate > 0 else peak_index
        else:
            compensated_peak_index = left_null_index + normal_peak_to_null

    return int(compensated_peak_index), int(left_null_index), int(peak_to_null_distance)


def _find_ifft_peak_index(mag):
    """SDKのfind_ifft_peak_index()相当."""
    nfft = len(mag)
    max_peak_index = int(np.argmax(mag))
    max_peak_value = mag[max_peak_index]

    nw = nfft - 2
    nw_next = nfft - 1
    first_rise_found = False
    short_path_found = False
    shortest_path_index = max_peak_index

    while nw != max_peak_index and not short_path_found:
        if mag[nw_next] < mag[nw]:
            if np.float32(2.5) * mag[nw] > max_peak_value and first_rise_found:
                shortest_path_index = nw
                short_path_found = True
        else:
            first_rise_found = True

        nw = nw_next
        nw_next = (nw_next + 1) % nfft

    peak_index = int(shortest_path_index)
    left_null_index = None
    peak_to_null_distance = None

    if peak_index < nfft - 2:
        peak_index, left_null_index, peak_to_null_distance = (
            _calculate_left_null_compensation_of_peak(peak_index, mag)
        )

    return {
        "peak_index": peak_index,
        "max_peak_index": max_peak_index,
        "shortest_path_index": int(shortest_path_index),
        "left_null_index": left_null_index,
        "peak_to_null_distance": peak_to_null_distance,
    }


def _peak_index_to_distance(peak_index, mag):
    nfft = len(mag)
    prompt = mag[peak_index]
    early = mag[peak_index - 1] if peak_index != 0 else mag[nfft - 1]
    late = mag[peak_index + 1] if peak_index != nfft - 1 else mag[0]

    if prompt >= early and prompt >= late:
        denom = 4 * prompt - 2 * (early + late)
        t_hat = (late - early) / denom if denom != 0 else math.nan
    else:
        t_hat = 0.0

    distance = ((peak_index + t_hat) * C) / (2.0 * nfft * DELTA_F)

    if peak_index >= nfft - 2 or distance < 0.0:
        distance = math.nan

    return float(distance), float(t_hat)


def _tone_values(tone):
    if isinstance(tone, dict):
        return (
            int(tone["channel"]),
            float(tone["local_i"]),
            float(tone["local_q"]),
            float(tone["peer_i"]),
            float(tone["peer_q"]),
        )

    if len(tone) != 5:
        raise ValueError("toneは(channel, local_i, local_q, peer_i, peer_q)にしてください")

    ch, il, ql, ir, qr = tone
    return int(ch), float(il), float(ql), float(ir), float(qr)


def _average_tones_by_channel(tones):
    averages = {}
    counts = defaultdict(int)
    ignored_channels = []

    for tone in tones:
        ch, il, ql, ir, qr = _tone_values(tone)
        n = ch - CHANNEL_INDEX_OFFSET

        if n < 0 or n >= CS_DE_NUM_CHANNELS:
            ignored_channels.append(ch)
            continue

        counts[n] += 1
        count = counts[n]

        if count == 1:
            averages[n] = [il, ql, ir, qr]
        else:
            old = averages[n]
            new = [il, ql, ir, qr]
            averages[n] = [
                (new_value / count) + ((1.0 - 1.0 / count) * old_value)
                for old_value, new_value in zip(old, new)
            ]

    return averages, counts, ignored_channels


def calc_ifft_distance_sdk(tones, nfft=DEFAULT_NFFT):
    """
    SDKのcs_de_ifft()に合わせたIFFT距離推定.

    tones:
        [(channel, local_i, local_q, peer_i, peer_q), ...]
    """
    iq_bins = np.zeros(nfft, dtype=np.complex64)
    averages, counts, ignored_channels = _average_tones_by_channel(tones)

    for n, (il, ql, ir, qr) in averages.items():
        iq_bins[n] = _combine_iq_like_sdk(il, ql, ir, qr)

    mag = _calculate_ifft_mag(iq_bins)
    peak_info = _find_ifft_peak_index(mag)
    distance, t_hat = _peak_index_to_distance(peak_info["peak_index"], mag)

    return {
        "distance": distance,
        "k_peak": peak_info["peak_index"],
        "t_hat": t_hat,
        "mag": mag,
        "used_channels": sorted(n + CHANNEL_INDEX_OFFSET for n in averages),
        "used_tones": int(sum(counts.values())),
        "ignored_channels": ignored_channels,
        **peak_info,
    }


def calc_ifft_distance(tones, nfft=DEFAULT_NFFT):
    """従来呼び出し用.戻り値は以前と同じ4個."""
    result = calc_ifft_distance_sdk(tones, nfft)
    return result["distance"], result["k_peak"], result["t_hat"], result["mag"]


def _safe_int(value):
    return int(str(value).strip(), 0)


def _row_from_parts(headers, parts):
    if len(parts) < 8:
        return None

    if headers and len(parts) >= len(headers):
        row = {name: parts[index] for index, name in enumerate(headers)}
    elif len(parts) >= 9:
        row = {
            "ranging_counter": parts[0],
            "mode": parts[1],
            "channel": parts[2],
            "side": parts[3],
            "antenna_path": parts[4],
            "quality_indicator": parts[5],
            "pct_le_hex": parts[6],
            "i": parts[7],
            "q": parts[8],
        }
    else:
        row = {
            "ranging_counter": parts[0],
            "mode": parts[1],
            "channel": parts[2],
            "side": parts[3],
            "antenna_path": parts[4],
            "quality_indicator": "",
            "pct_le_hex": parts[5],
            "i": parts[6],
            "q": parts[7],
        }

    try:
        quality_text = str(row.get("quality_indicator", "")).strip()
        quality = _safe_int(quality_text) if quality_text != "" else None
        if quality is not None and quality not in (0, 1, 2, 3):
            return None

        return {
            "ranging_counter": _safe_int(row["ranging_counter"]),
            "channel": _safe_int(row["channel"]),
            "side": str(row["side"]).strip(),
            "antenna_path": _safe_int(row.get("antenna_path", 0)),
            "quality_indicator": quality,
            "i": _safe_int(row["i"]),
            "q": _safe_int(row["q"]),
        }
    except (KeyError, ValueError):
        return None


def load_raw_tone_csv(path, counter=None, antenna_path=0):
    """
    raw_tone.csvからSDK計算に入るtoneだけを作る.
    quality_indicator列がある場合は,local/peerの両方が0のtoneだけ使う.
    """
    by_channel = defaultdict(lambda: {"local": deque(), "peer": deque()})
    headers = None
    total_rows = 0
    quality_rows = 0
    low_quality_rows = 0
    quality_counts = defaultdict(int)
    skipped_quality = 0
    skipped_incomplete = 0

    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        for parts in csv.reader(f):
            if not parts:
                continue

            if parts[0] == "ranging_counter":
                headers = parts
                continue

            row = _row_from_parts(headers, parts)
            if row is None:
                continue

            if counter is not None and row["ranging_counter"] != counter:
                continue

            if row["antenna_path"] != antenna_path:
                continue

            if row["side"] not in ("local", "peer"):
                continue

            total_rows += 1

            if row["quality_indicator"] is not None:
                quality_rows += 1
                quality_counts[row["quality_indicator"]] += 1

                if row["quality_indicator"] != TONE_QUALITY_HIGH:
                    low_quality_rows += 1

            by_channel[row["channel"]][row["side"]].append(row)

    tones = []

    for channel in sorted(by_channel):
        local_rows = by_channel[channel]["local"]
        peer_rows = by_channel[channel]["peer"]
        pair_count = min(len(local_rows), len(peer_rows))

        if pair_count == 0:
            skipped_incomplete += len(local_rows) + len(peer_rows)
            continue

        skipped_incomplete += abs(len(local_rows) - len(peer_rows))

        for _ in range(pair_count):
            local = local_rows.popleft()
            peer = peer_rows.popleft()

            local_quality = local["quality_indicator"]
            peer_quality = peer["quality_indicator"]

            if (
                local_quality is not None
                and peer_quality is not None
                and (
                    local_quality != TONE_QUALITY_HIGH
                    or peer_quality != TONE_QUALITY_HIGH
                )
            ):
                skipped_quality += 1
                continue

            tones.append(
                (
                    channel,
                    local["i"],
                    local["q"],
                    peer["i"],
                    peer["q"],
                )
            )

    return tones, {
        "total_rows": total_rows,
        "quality_rows": quality_rows,
        "low_quality_rows": low_quality_rows,
        "quality_counts": dict(sorted(quality_counts.items())),
        "skipped_quality": skipped_quality,
        "skipped_incomplete": skipped_incomplete,
    }


def _read_nfft(default=DEFAULT_NFFT):
    nfft_input = input(f"NFFTを入力してください,空ならSDK設定の{default}: ")
    return default if nfft_input.strip() == "" else int(nfft_input)


def _read_manual_tones():
    print("手入力します.")
    print("入力形式: channel,local_i,local_q,peer_i,peer_q")
    print("例: 2,152,59,153,-75")
    print("入力終了: 何も入力せずEnter")
    print()

    tones = []

    while True:
        line = input("tone入力: ")

        if line.strip() == "":
            break

        try:
            ch, il, ql, ir, qr = map(int, line.split(","))
            tones.append((ch, il, ql, ir, qr))
        except ValueError:
            print("入力形式が違います,例: 2,152,59,153,-75")

    return tones


def _print_result(result, nfft, meta=None):
    print()
    print("===== SDK IFFT計算結果 =====")
    print(f"NFFT: {nfft}")
    print(f"使用tone数: {result['used_tones']}")
    print(f"使用channel数: {len(result['used_channels'])}")
    print(f"最大ピークindex: {result['max_peak_index']}")
    print(f"近距離ピークindex: {result['shortest_path_index']}")
    print(f"補正後k_peak: {result['k_peak']}")
    print(f"t_hat: {result['t_hat']:.6f}")
    print(f"IFFT距離: {result['distance']:.6f} m")

    if result["left_null_index"] is not None:
        print(f"left_null_index: {result['left_null_index']}")
        print(f"peak_to_null_distance: {result['peak_to_null_distance']}")

    if result["ignored_channels"]:
        print(f"範囲外channel: {result['ignored_channels']}")

    if meta:
        print()
        print("===== CSV読み込み情報 =====")
        print(f"読み込み行数: {meta['total_rows']}")
        print(f"quality_indicatorありの行数: {meta['quality_rows']}")
        print(f"低品質tone行数: {meta['low_quality_rows']}")
        print(f"qualityで除外したペア数: {meta['skipped_quality']}")
        print(f"local/peer不足で除外した行数: {meta['skipped_incomplete']}")

        if meta["quality_rows"] == 0:
            print("注意: quality_indicatorが無いので,全toneをHIGH扱いで計算しました.")
        else:
            print("quality_indicator内訳:")
            for quality, count in meta["quality_counts"].items():
                label = "HIGH" if quality == TONE_QUALITY_HIGH else "LOW_QUALITY"
                print(f"  {quality}({label}): {count}")

    print()
    print("===== ピーク上位5個 =====")
    mag = result["mag"]
    top_indices = np.argsort(mag)[-5:][::-1]

    for idx in top_indices:
        d = idx * C / (2.0 * nfft * DELTA_F)
        print(f"index={idx}, magnitude={mag[idx]:.6f}, 距離={d:.6f} m")


def main():
    print("SDKと同じ考え方でIFFT距離推定を行います.")
    print("raw_tone.csvを指定すると,quality_indicator=0のtoneだけ使います.")
    print()

    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        counter = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        nfft = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_NFFT
        tones, meta = load_raw_tone_csv(csv_path, counter)
    else:
        csv_path = input("raw_tone.csvのパスを入力,手入力なら空Enter: ").strip().strip('"')

        if csv_path:
            counter_input = input("ranging_counterを入力,全件なら空Enter: ")
            counter = None if counter_input.strip() == "" else int(counter_input)
            nfft = _read_nfft()
            tones, meta = load_raw_tone_csv(csv_path, counter)
        else:
            nfft = _read_nfft()
            tones = _read_manual_tones()
            meta = None

    if len(tones) == 0:
        print("toneがありません.")
        return

    result = calc_ifft_distance_sdk(tones, nfft)
    _print_result(result, nfft, meta)


if __name__ == "__main__":
    main()
