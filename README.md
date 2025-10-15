# Proyecto- Poligrafo 

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt

# ================== FUNCIÓN DE ECG SINTÉTICO ==================
def ecgsyn(fs=500, N=5000, hr_mean=60):
    hr = hr_mean / 60.0
    dt = 1/fs
    t = np.arange(N) * dt

    theta_p, theta_q, theta_r, theta_s, theta_t = -np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2
    a_p, a_q, a_r, a_s, a_t = 0.25, -0.1, 1.0, -0.25, 0.35
    b_p, b_q, b_r, b_s, b_t = 0.25, 0.1, 0.1, 0.1, 0.4

    theta = np.zeros(N)
    z = np.zeros(N)
    omega = 2 * np.pi * hr

    for i in range(1, N):
        theta[i] = np.mod(theta[i-1] + omega * dt, 2*np.pi)
        z[i] = (
            a_p * np.exp(-((theta[i]-theta_p)**2)/(2*b_p**2)) +
            a_q * np.exp(-((theta[i]-theta_q)**2)/(2*b_q**2)) +
            a_r * np.exp(-((theta[i]-theta_r)**2)/(2*b_r**2)) +
            a_s * np.exp(-((theta[i]-theta_s)**2)/(2*b_s**2)) +
            a_t * np.exp(-((theta[i]-theta_t)**2)/(2*b_t**2))
        )
    return t, z


# ================== GENERACIÓN DE SEÑALES ==================
fs = 1000
t_simulacion = 120
t = np.linspace(0, t_simulacion, fs*t_simulacion)

_, ecg = ecgsyn(fs, N=t_simulacion*fs, hr_mean=65)
ecg = ecg + 0.08 * np.random.randn(len(t))

alterador = 0.3*np.sin(2*np.pi*0.02*t)
a_base = 0.5 + 0.2*np.sin(2*np.pi*(1.335+alterador)*t) + 0.12*np.sin(2*np.pi*(1.335+alterador)*3*t+200)
ruido = 0.08 * np.random.randn(len(t))
ruido_suave = np.convolve(ruido, np.ones(10)/10, mode='same')
resp = a_base + ruido_suave

sweat = np.cumsum(0.001 * np.random.randn(len(t))) + 0.3


# ================== FILTRADO ==================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def filtrar_senal(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

ecg_filtrada = filtrar_senal(ecg, 0.5, 40, fs)
resp_filtrada = filtrar_senal(resp, 0.1, 1, fs)

# ================== GRAFICAR FILTRADO (solo ECG) ==================
plt.figure(figsize=(10,4))
plt.plot(t[0:5000], ecg[0:5000], label="ECG crudo", alpha=0.6)
plt.plot(t[0:5000], ecg_filtrada[0:5000], label="ECG filtrado", color='r')
plt.legend(); plt.title("ECG antes y después del filtrado")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()


# ================== FFT (solo ECG) ==================
def fft_magnitude(x, fs):
    N = len(x)
    f = np.fft.rfftfreq(N, 1/fs)
    X = np.abs(np.fft.rfft(x * np.hanning(N))) / N
    return f, X

f_raw, X_raw = fft_magnitude(ecg, fs)
f_filt, X_filt = fft_magnitude(ecg_filtrada, fs)
plt.figure(figsize=(10,4))
plt.semilogy(f_raw, X_raw, label="ECG crudo")
plt.semilogy(f_filt, X_filt, label="ECG filtrado")
plt.xlim(0, 60)
plt.title("FFT antes y después del filtrado (ECG)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()


# ================== ANÁLISIS POR VENTANAS ==================
ventana = fs * 10
solape = fs * 5
inicio = 0
indices = []
media_ecg, var_ecg, hr_ecg = [], [], []
media_resp = []
media_sweat, var_sweat = [], []

while inicio + ventana <= len(t):
    fin = inicio + ventana
    seg_ecg = ecg_filtrada[inicio:fin]
    seg_resp = resp_filtrada[inicio:fin]
    seg_sweat = sweat[inicio:fin]
    indices.append(t[inicio:fin].mean())

    # Estadísticos
    media_ecg.append(np.mean(seg_ecg))
    var_ecg.append(np.var(seg_ecg))
    media_resp.append(np.mean(seg_resp))
    media_sweat.append(np.mean(seg_sweat))
    var_sweat.append(np.var(seg_sweat))

    # Picos ECG
    prom = max(0.2*np.ptp(seg_ecg), 0.05)
    picos, _ = signal.find_peaks(seg_ecg, distance=fs/2.5, prominence=prom)
    hr = (len(picos) / (len(seg_ecg)/fs)) * 60 if len(picos)>1 else np.nan
    hr_ecg.append(hr)

    inicio += ventana - solape


# ================== FFT POR VENTANAS (solo ECG) ==================
def fft_por_ventanas(senal, fs, ventana, solape):
    inicio = 0
    espec = []
    tiempos = []
    while inicio + ventana <= len(senal):
        seg = senal[inicio:inicio+ventana]
        F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg)
        f = np.fft.rfftfreq(len(seg), 1/fs)
        espec.append(F)
        tiempos.append(seg.mean()) # Should be t[inicio:inicio+ventana].mean()
        inicio += ventana - solape
    return tiempos, np.array(espec), np.array(f)


# ================== GRAFICAR RESULTADOS POR VENTANAS ==================
# Convertir listas a arrays de numpy para graficar más fácil
indices = np.array(indices)
media_ecg = np.array(media_ecg)
var_ecg = np.array(var_ecg)
hr_ecg = np.array(hr_ecg)
media_resp = np.array(media_resp)
media_sweat = np.array(media_sweat)
var_sweat = np.array(var_sweat)


plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(indices, media_ecg, label='Media ECG', color='r')
plt.plot(indices, var_ecg, label='Varianza ECG', color='orange')
plt.ylabel("Amplitud/Varianza")
plt.legend()
plt.title("Estadísticos ECG por Ventanas")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(indices, hr_ecg, label='Frecuencia Cardíaca Estimada', color='purple')
plt.ylabel("BPM")
plt.legend()
plt.title("Frecuencia Cardíaca por Ventanas")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(indices, media_sweat, label='Media Sudoración', color='g')
plt.plot(indices, var_sweat, label='Varianza Sudoración', color='brown')
plt.ylabel("Amplitud/Varianza")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.title("Estadísticos Sudoración por Ventanas")
plt.grid(True)

plt.tight_layout()
plt.show()


```
## **Filtrado**

<img width="989" height="390" alt="download" src="https://github.com/user-attachments/assets/70a9e226-8b84-4d59-a5a6-d547fa90810c" />

## **Estadisticos por ventanas**
<img width="1189" height="790" alt="download" src="https://github.com/user-attachments/assets/76056942-351d-444d-8ce2-2fc40cd8925c" />


```
# Señal respiratoria (onda sinusoidal lenta)
alterador = 0.3*np.sin(2*np.pi*0.02*t)
a_base = 0.5 + 0.2*np.sin(2*np.pi*(1.335+alterador)*t) + 0.12*np.sin(2*np.pi*(1.335+alterador)*3*t+200)

# Ruido blanco
ruido = 0.08 * np.random.randn(len(t))

# --- Suavizar el ruido (baja su frecuencia) ---
N = 10  # tamaño de ventana, cuanto mayor => más suave
ruido_suave = np.convolve(ruido, np.ones(N)/N, mode='same')

a = a_base + ruido_suave

resp = a


# Señal de sudoración (respuesta galvánica lenta)
sweat = np.cumsum(0.001 * np.random.randn(len(t))) + 0.3
plt.subplot(3,1,2)
plt.plot(t[0:10000], resp[0:10000], color='b')
plt.title("Señal Respiratoria")
plt.subplot(3,1,3)
plt.plot(t, sweat, color='g')
plt.title("Señal de Sudoración (GSR)")
plt.xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()
```
## **Señal respiratoria y de sudoracion**
<img width="630" height="336" alt="download" src="https://github.com/user-attachments/assets/0e1efaf5-4879-4219-8fd5-5fa84c342fdc" />

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================================
#        FILTRO PASA BANDA BUTTERWORTH - FUNCIÓN GENERAL
# ==========================================================
def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
 
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def aplicar_filtro(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4):
  
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal


# ==========================================================
#        DEMOSTRACIÓN DE USO (EJEMPLO DE APLICACIÓN)
# ==========================================================
if __name__ == "__main__":
    # Parámetros de simulación
    fs = 1000  # Frecuencia de muestreo (Hz)
    duration = 5  # Duración de la señal (segundos)
    t = np.linspace(0, duration, fs * duration)

    # Señal con varias frecuencias + ruido
    signal_original = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
    ruido = 0.2 * np.random.randn(len(t))
    signal_ruidosa = signal_original + ruido

    # Aplicación del filtro pasa banda (1–40 Hz)
    signal_filtrada = aplicar_filtro(signal_ruidosa, lowcut=1, highcut=40, fs=fs, order=4)

    # ==========================================================
    #        VISUALIZACIÓN DE RESULTADOS
    # ==========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal_ruidosa, label="Señal con ruido", alpha=0.6)
    plt.plot(t, signal_filtrada, label="Señal filtrada (1–40 Hz)", color='r')
    plt.title("Filtro Pasa Banda Butterworth Aplicado a Señal Ruidosa")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

```
## **Filtro pasa banda**
<img width="989" height="490" alt="download" src="https://github.com/user-attachments/assets/da3e3c49-99b0-450b-bae3-2f6875d675ec" />

