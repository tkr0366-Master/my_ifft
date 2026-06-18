import csv

import numpy as np

C = 299792458.0
DELTA_F = 1_000_000.0


def load_raw_tone_csv(path, counter=None, antenna_path=0):
    """
    raw_tone.csvからtone配列を作る.
    basic版なのでquality_indicatorでは除外せず,local/peerが揃ったtoneを使う.
    """
    rows_by_channel = {}
    headers = None
    total_rows = 0
    skipped_rows = 0

    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        for parts in csv.reader(f):
            if not parts:
                continue

            if parts[0] == "ranging_counter":
                headers = parts
                continue

            try:
                if headers and len(parts) >= len(headers):
                    row = {name: parts[index] for index, name in enumerate(headers)}
                elif len(parts) >= 9:
                    row = {
                        "ranging_counter": parts[0],
                        "channel": parts[2],
                        "side": parts[3],
                        "antenna_path": parts[4],
                        "i": parts[7],
                        "q": parts[8],
                    }
                elif len(parts) >= 8:
                    row = {
                        "ranging_counter": parts[0],
                        "channel": parts[2],
                        "side": parts[3],
                        "antenna_path": parts[4],
                        "i": parts[6],
                        "q": parts[7],
                    }
                else:
                    skipped_rows += 1
                    continue

                row_counter = int(row["ranging_counter"])
                ch = int(row["channel"])
                side = row["side"].strip()
                ap = int(row.get("antenna_path", 0))
                i_value = int(row["i"])
                q_value = int(row["q"])
            except (KeyError, ValueError):
                skipped_rows += 1
                continue

            if counter is not None and row_counter != counter:
                continue

            if ap != antenna_path:
                continue

            if side not in ("local", "peer"):
                skipped_rows += 1
                continue

            total_rows += 1

            if ch not in rows_by_channel:
                rows_by_channel[ch] = {"local": [], "peer": []}

            rows_by_channel[ch][side].append((i_value, q_value))

    tones = []
    skipped_incomplete = 0

    for ch in sorted(rows_by_channel):
        local_rows = rows_by_channel[ch]["local"]
        peer_rows = rows_by_channel[ch]["peer"]
        pair_count = min(len(local_rows), len(peer_rows))

        if pair_count == 0:
            skipped_incomplete += len(local_rows) + len(peer_rows)
            continue

        skipped_incomplete += abs(len(local_rows) - len(peer_rows))

        for index in range(pair_count):
            il, ql = local_rows[index]
            ir, qr = peer_rows[index]
            tones.append((ch, il, ql, ir, qr))

    return tones, {
        "total_rows": total_rows,
        "skipped_rows": skipped_rows,
        "skipped_incomplete": skipped_incomplete,
    }


def calc_ifft_distance(tones, nfft=512):
    """
    基本的なIFFT距離推定

    tones:
        [(channel, local_i, local_q, peer_i, peer_q), ...]

    処理:
        1．X[channel - 2] = local × peer
        2．N点IFFT
        3．IFFT magnitudeを計算
        4．最大ピークindexを探す
        5．indexを距離に変換
    """

    # IFFT入力配列
    x = np.zeros(nfft, dtype=np.complex128)

    for ch, il, ql, ir, qr in tones:
        n = ch - 2

        if n < 0 or n >= 75:
            print(f"channel {ch} は範囲外なので無視します")
            continue

        local = complex(il, ql)
        peer = complex(ir, qr)

        # X[n] = local × peer
        x[n] = local * peer

    # IFFT
    y = np.fft.ifft(x)

    # IFFT magnitude
    mag = np.abs(y)

    # 最大ピークindex
    k_peak = int(np.argmax(mag))

    # 1 indexあたりの距離
    bin_distance = C / (2.0 * nfft * DELTA_F)

    # 補間なし距離
    distance_no_interp = k_peak * bin_distance

    # 3点補間
    t_hat = 0.0

    if 0 < k_peak < nfft - 1:
        early = mag[k_peak - 1]
        prompt = mag[k_peak]
        late = mag[k_peak + 1]

        denominator = 4.0 * prompt - 2.0 * (early + late)

        if prompt >= early and prompt >= late and abs(denominator) > 1e-12:
            t_hat = (late - early) / denominator

    # 補間あり距離
    distance_interp = (k_peak + t_hat) * bin_distance

    return {
        "k_peak": k_peak,
        "t_hat": t_hat,
        "bin_distance": bin_distance,
        "distance_no_interp": distance_no_interp,
        "distance_interp": distance_interp,
        "max_magnitude": mag[k_peak],
        "mag": mag,
    }


def main():
    print("基本的なIFFT距離推定を行います")
    print("raw_tone.csvを読むか,手入力できます")
    print()

    csv_path = input("raw_tone.csvのパスを入力,手入力なら空Enter: ").strip().strip('"')

    nfft_text = input("NFFTを入力してください．何も入力しない場合は512：").strip()

    if nfft_text == "":
        nfft = 512
    else:
        nfft = int(nfft_text)

    if csv_path:
        counter_text = input("ranging_counterを入力,全件なら空Enter: ").strip()
        counter = None if counter_text == "" else int(counter_text)
        tones, meta = load_raw_tone_csv(csv_path, counter)
    else:
        print("入力形式：channel,local_i,local_q,peer_i,peer_q")
        print("例：2,152,59,153,-75")
        print("入力終了：何も入力せずEnter")
        print()

        tones = []
        meta = None

        while True:
            line = input("tone入力：").strip()

            if line == "":
                break

            try:
                ch, il, ql, ir, qr = map(int, line.split(","))
                tones.append((ch, il, ql, ir, qr))
            except ValueError:
                print("入力形式が違います．例：2,152,59,153,-75")

    if len(tones) == 0:
        print("toneが入力されていません")
        return

    result = calc_ifft_distance(tones, nfft)

    print()
    print("===== IFFT計算結果 =====")
    print(f"使用tone数: {len(tones)}")
    print(f"NFFT: {nfft}")
    print(f"1 indexあたりの距離: {result['bin_distance']:.9f} m")
    print()
    print(f"最大ピークindex: {result['k_peak']}")
    print(f"最大ピーク距離: {result['distance_no_interp']:.6f} m")
    print(f"最大ピークmagnitude: {result['max_magnitude']:.6f}")
    print()
    print(f"IFFT距離(最大ピーク距離): {result['distance_no_interp']:.6f} m")

    if meta is not None:
        print()
        print("===== CSV読み込み情報 =====")
        print(f"読み込み行数：{meta['total_rows']}")
        print(f"スキップ行数：{meta['skipped_rows']}")
        print(f"local/peer不足で除外した行数：{meta['skipped_incomplete']}")

    print()
    print("===== ピーク上位5個 =====")

    mag = result["mag"]
    top_indices = np.argsort(mag)[-5:][::-1]

    for k in top_indices:
        distance = k * result["bin_distance"]
        print(f"index={k}, magnitude={mag[k]:.6f}, distance={distance:.6f} m")


if __name__ == "__main__":
    main()
