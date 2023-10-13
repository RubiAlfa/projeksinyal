import numpy as np
import matplotlib.pyplot as plt

# Fungsi DFT dua dimensi buatan
def dft2d(x):
    M, N = x.shape
    X = np.zeros((M, N), dtype=np.complex128)

    for u in range(M):
        for v in range(N):
            for m in range(M):
                for n in range(N):
                    X[u, v] += x[m, n] * np.exp(-2j * np.pi * ((u * m) / M + (v * n) / N))

    return X

# Membuat data gambar contoh (sumber: Wikipedia - Test Images)
image = np.zeros((128, 128), dtype=np.float32)
image[32:96, 32:96] = 1.0
image[64:96, 64:96] = 0.0

# Menghitung DFT dua dimensi dengan fungsi buatan
spectrum = dft2d(image)

# Menghitung DFT dua dimensi dengan NumPy
spectrum_np = np.fft.fft2(image)

# Plot gambar asli
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Gambar Asli')

# Plot magnitude spektrum DFT buatan
plt.subplot(1, 3, 2)
plt.imshow(np.abs(spectrum), cmap='gray')
plt.title('Magnitude DFT Buatan')

# Plot magnitude spektrum DFT NumPy
plt.subplot(1, 3, 3)
plt.imshow(np.abs(spectrum_np), cmap='gray')
plt.title('Magnitude DFT NumPy')

plt.tight_layout()
plt.show()
