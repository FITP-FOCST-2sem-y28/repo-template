# signal_filtering_n105.py

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


N = 105
DT = 1e-4
FS = 1 / DT
DF = FS / N

MESSAGE = "OK"
START_BIN = 5
SIGNAL_AMPLITUDE = 1.0

OUTDIR = Path("results")
SEED = 42


def text_to_bits(text):
    return "".join(f"{ord(ch):08b}" for ch in text)


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def amplitude_spectrum(x):
    x_fft = np.fft.rfft(x)
    amp = 2 * np.abs(x_fft) / len(x)
    amp[0] = np.abs(x_fft[0]) / len(x)
    return x_fft, amp


def create_encoded_signal(bits):
    n = np.arange(N)
    max_bin = N // 2

    if START_BIN + len(bits) - 1 > max_bin:
        raise ValueError("Сообщение слишком длинное для N=105. Используйте более короткий текст.")

    code_bins = np.arange(START_BIN, START_BIN + len(bits))
    signal = np.zeros(N)
    active_bins = []

    for bit, k in zip(bits, code_bins):
        if bit == "1":
            phase = 0.31 * k
            signal += SIGNAL_AMPLITUDE * np.cos(2 * np.pi * k * n / N + phase)
            active_bins.append(k)

    return signal, code_bins, np.array(active_bins, dtype=int)


def create_controlled_noise(active_bins):
    rng = np.random.default_rng(SEED)
    noise_fft = np.zeros(N // 2 + 1, dtype=complex)

    max_noise_amplitude = SIGNAL_AMPLITUDE / 2

    for k in range(1, N // 2 + 1):
        if k in active_bins:
            continue

        amp = rng.uniform(0, max_noise_amplitude)
        phase = rng.uniform(0, 2 * np.pi)

        noise_fft[k] = (N * amp / 2) * np.exp(1j * phase)

    noise = np.fft.irfft(noise_fft, n=N)
    return noise


def detect_useful_bins(x):
    x_fft, amp = amplitude_spectrum(x)

    bins = np.arange(1, len(amp))
    values = amp[1:]

    order = np.argsort(values)[::-1]
    sorted_values = values[order]

    split_index = None

    for i in range(len(sorted_values) - 1):
        if sorted_values[i + 1] == 0:
            continue

        ratio = sorted_values[i] / sorted_values[i + 1]

        if ratio >= 2:
            split_index = i
            break

    if split_index is None:
        ratios = sorted_values[:-1] / np.maximum(sorted_values[1:], 1e-15)
        split_index = int(np.argmax(ratios))

    threshold = np.sqrt(sorted_values[split_index] * sorted_values[split_index + 1])
    detected_bins = bins[values > threshold]

    return detected_bins, threshold, x_fft, amp


def filter_signal(x_fft, detected_bins):
    filtered_fft = np.zeros_like(x_fft)
    filtered_fft[detected_bins] = x_fft[detected_bins]
    return np.fft.irfft(filtered_fft, n=N)


def save_results(clean, noise, noisy, filtered, code_bins, detected_bins, bits, decoded_bits, amp, threshold):
    OUTDIR.mkdir(exist_ok=True)

    clean.astype(np.float64).tofile(OUTDIR / "clean_signal.bin")
    noisy.astype(np.float64).tofile(OUTDIR / "noisy_signal.bin")
    filtered.astype(np.float64).tofile(OUTDIR / "filtered_signal.bin")

    freqs = np.fft.rfftfreq(N, DT)

    with open(OUTDIR / "frequencies.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "frequency_hz", "encoded_bit", "detected"])

        detected_set = set(detected_bins)

        for bit, k in zip(bits, code_bins):
            writer.writerow([
                k,
                freqs[k],
                bit,
                int(k in detected_set)
            ])

    t = np.arange(N) * DT

    plt.figure(figsize=(10, 5))
    plt.stem(freqs[1:], amp[1:], basefmt=" ")
    plt.axhline(threshold, color="red", linestyle="--", label="автоматический порог")
    plt.scatter(
        freqs[detected_bins],
        amp[detected_bins],
        color="red",
        zorder=3,
        label="найденные информационные гармоники"
    )
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "spectrum.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t * 1000, noisy, "o-", alpha=0.55, label="зашумленный сигнал")
    plt.plot(t * 1000, clean, linewidth=2, label="исходный сигнал")
    plt.plot(t * 1000, filtered, "--", linewidth=2, label="восстановленный сигнал")
    plt.xlabel("Время, мс")
    plt.ylabel("Амплитуда")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "time_signals.png", dpi=200)
    plt.close()

    original_bits = np.array([int(b) for b in bits])
    recovered_bits = np.array([int(b) for b in decoded_bits])

    plt.figure(figsize=(10, 3))
    plt.imshow(
        np.vstack([original_bits, recovered_bits]),
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1
    )
    plt.yticks([0, 1], ["исходные", "восстановленные"])
    plt.xticks(np.arange(len(bits)), np.arange(1, len(bits) + 1))
    plt.xlabel("Номер бита")
    plt.title("Исходные и восстановленные биты")
    plt.tight_layout()
    plt.savefig(OUTDIR / "decoded_bits.png", dpi=200)
    plt.close()


def main():
    bits = text_to_bits(MESSAGE)

    clean, code_bins, active_bins = create_encoded_signal(bits)
    noise = create_controlled_noise(active_bins)
    noisy = clean + noise

    detected_bins, threshold, noisy_fft, amp = detect_useful_bins(noisy)
    filtered = filter_signal(noisy_fft, detected_bins)

    detected_set = set(detected_bins)
    decoded_bits = "".join("1" if k in detected_set else "0" for k in code_bins)
    decoded_message = bits_to_text(decoded_bits)

    save_results(
        clean=clean,
        noise=noise,
        noisy=noisy,
        filtered=filtered,
        code_bins=code_bins,
        detected_bins=detected_bins,
        bits=bits,
        decoded_bits=decoded_bits,
        amp=amp,
        threshold=threshold
    )

    print(f"N = {N}")
    print(f"dt = {DT} s")
    print(f"Fs = {FS:.2f} Hz")
    print(f"df = {DF:.4f} Hz")
    print()
    print(f"Исходное сообщение:        {MESSAGE}")
    print(f"Исходные биты:             {bits}")
    print(f"Восстановленные биты:      {decoded_bits}")
    print(f"Восстановленное сообщение: {decoded_message}")
    print()
    print("Найденные информационные частоты:")

    freqs = np.fft.rfftfreq(N, DT)

    for k in detected_bins:
        print(f"k = {k:2d}, f = {freqs[k]:8.3f} Гц, амплитуда = {amp[k]:.4f}")

    print()
    print("Файлы сохранены в папку:", OUTDIR)


if __name__ == "__main__":
    main()
