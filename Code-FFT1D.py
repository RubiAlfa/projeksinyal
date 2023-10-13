import numpy as np
import matplotlib.pyplot as plt

# Fungsi FFT buatan
def fft1d(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Membuat data sinyal contoh
N = 64
t = np.linspace(0, 2 * np.pi, N)
x = np.sin(5 * t) + 0.5 * np.sin(12 * t)

# Menghitung FFT dengan fungsi buatan
X = fft1d(x)

# Menghitung FFT dengan NumPy
X_np = np.fft.fft(x)

# Plot sinyal asli
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title('Sinyal Asli')

# Plot magnitude spektrum FFT buatan
plt.subplot(3, 2, 3)
plt.plot(np.abs(X))
plt.title('Magnitude FFT Buatan')

# Plot fase spektrum FFT buatan
plt.subplot(3, 2, 4)
plt.plot(np.angle(X))
plt.title('Fase FFT Buatan')

# Plot magnitude spektrum FFT NumPy
plt.subplot(3, 2, 5)
plt.plot(np.abs(X_np))
plt.title('Magnitude FFT NumPy')

# Plot fase spektrum FFT NumPy
plt.subplot(3, 2, 6)
plt.plot(np.angle(X_np))
plt.title('Fase FFT NumPy')

plt.tight_layout()
plt.show()
