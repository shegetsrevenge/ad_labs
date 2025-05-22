import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

initial_amplitude = 1.0
initial_frequency = 1.0
initial_phase = 0.0
initial_noise_mean = 0.0
initial_noise_covariance = 0.1
initial_show_noise = True
initial_show_filtered = True

t = np.linspace(0, 2 * np.pi, 1000)

def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise, keep_noise=None):
    clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if keep_noise is None:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=t.shape)
    else:
        noise = keep_noise
    result = clean + noise if show_noise else clean
    return clean, result, noise

def filter_signal(signal, cutoff=3, fs=1000, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.45)
line_clean, = ax.plot(t, np.zeros_like(t), label='Чиста гармоніка', linestyle='--')
line_noisy, = ax.plot(t, np.zeros_like(t), label='З шумом', alpha=0.6)
line_filtered, = ax.plot(t, np.zeros_like(t), label='Фільтрована', linestyle='-.')

ax.legend(loc='upper right')
ax.set_title("Гармоніка з шумом та фільтрацією")
ax.set_xlabel("Час")
ax.set_ylabel("Амплітуда")

amp_ax = plt.axes([0.25, 0.35, 0.65, 0.03])
freq_ax = plt.axes([0.25, 0.30, 0.65, 0.03])
phase_ax = plt.axes([0.25, 0.25, 0.65, 0.03])
mean_ax = plt.axes([0.25, 0.20, 0.65, 0.03])
cov_ax = plt.axes([0.25, 0.15, 0.65, 0.03])

amp_slider = Slider(amp_ax, 'Амплітуда', 0.1, 5.0, valinit=initial_amplitude)
freq_slider = Slider(freq_ax, 'Частота', 0.1, 5.0, valinit=initial_frequency)
phase_slider = Slider(phase_ax, 'Фаза', 0.0, 2*np.pi, valinit=initial_phase)
mean_slider = Slider(mean_ax, 'Сер. шуму', -1.0, 1.0, valinit=initial_noise_mean)
cov_slider = Slider(cov_ax, 'Дисперсія шуму', 0.0, 1.0, valinit=initial_noise_covariance)

reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset')

checkbox_ax = plt.axes([0.025, 0.5, 0.2, 0.15])
checkbox = CheckButtons(checkbox_ax, ['Показати шум', 'Показати фільтр'], [initial_show_noise, initial_show_filtered])

last_noise = None

def update(val=None, changed=''):
    global last_noise
    amp = amp_slider.val
    freq = freq_slider.val
    phase = phase_slider.val
    mean = mean_slider.val
    cov = cov_slider.val
    show_noise = checkbox.get_status()[0]
    show_filtered = checkbox.get_status()[1]

    if changed in ['mean', 'cov']:
        clean, noisy, last_noise = harmonic_with_noise(amp, freq, phase, mean, cov, show_noise)
    else:
        clean, noisy, _ = harmonic_with_noise(amp, freq, phase, mean, cov, show_noise, keep_noise=last_noise)

    filtered = filter_signal(noisy)

    line_clean.set_ydata(clean)
    line_noisy.set_ydata(noisy if show_noise else np.full_like(t, np.nan))
    line_filtered.set_ydata(filtered if show_filtered else np.full_like(t, np.nan))

    fig.canvas.draw_idle()

amp_slider.on_changed(lambda val: update(changed='amp'))
freq_slider.on_changed(lambda val: update(changed='freq'))
phase_slider.on_changed(lambda val: update(changed='phase'))
mean_slider.on_changed(lambda val: update(changed='mean'))
cov_slider.on_changed(lambda val: update(changed='cov'))
checkbox.on_clicked(lambda label: update())

def reset(event):
    amp_slider.reset()
    freq_slider.reset()
    phase_slider.reset()
    mean_slider.reset()
    cov_slider.reset()
    checkbox.set_active(0)
    checkbox.set_active(1)
    update()

reset_button.on_clicked(reset)

update()

plt.show()
