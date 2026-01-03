import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Askelmittaus + GPS", layout="wide")
st.title("Askelmittaus + GPS-analyysi")

# --- Lataa data (pidetään mahdollisimman samana) ---
acc_df = pd.read_csv('./Data/projekti_kiihtyvyys.csv')

# =========================
# Kiihtyvyyskuvaajat
# =========================
st.header("Kiihtyvyys (y)")

plt.figure(figsize=(12, 8))

#plt.subplot(3, 1, 1)
#plt.plot(acc_df['Time (s)'], acc_df['Linear Acceleration x (m/s^2)'])
#plt.ylabel('Acceleration x')

plt.subplot(3, 1, 2)
plt.plot(acc_df['Time (s)'], acc_df['Linear Acceleration y (m/s^2)'])
plt.ylabel('Acceleration y')

#plt.subplot(3, 1, 3)
#plt.plot(acc_df['Time (s)'], acc_df['Linear Acceleration z (m/s^2)'])
#plt.ylabel('Acceleration z')

plt.xlabel('Aika [s]')
plt.suptitle('Askelmittaus')
st.pyplot(plt.gcf())
plt.close()

# =========================
# Menetelmä 1: Suodatus
# =========================
st.header("Askelmäärä (Menetelmä 1: Suodatus)")

def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

data = acc_df['Linear Acceleration y (m/s^2)']
T_tot = acc_df['Time (s)'].max()
n = len(acc_df['Time (s)'])
fs = n / T_tot
nyq = fs / 2
order = 3
cutoff = 1 / 0.4

data_filt = butter_lowpass_filter(data, cutoff, nyq, order)

plt.figure(figsize=(12, 4))
plt.plot(acc_df['Time (s)'], data, label='data', alpha=0.4, linewidth=0.7)
plt.plot(acc_df['Time (s)'], data_filt, label='suodatettu data', linewidth=2)
plt.axis([0, 5, -6, 6])
plt.grid()
plt.legend()
st.pyplot(plt.gcf())
plt.close()

jaksot = 0
for i in range(n - 1):
    if data_filt[i] * data_filt[i + 1] < 0:
        jaksot = jaksot + 1 / 2

st.metric("Askelten määrä (suodatus)", f"{jaksot:.0f}")

# =========================
# Menetelmä 2: Fourier
# =========================
st.header("Askelmäärä (Menetelmä 2: Fourier-analyysi)")

signal = acc_df['Linear Acceleration y (m/s^2)']
t = acc_df['Time (s)']
N = len(signal)
dt = np.max(t) / N

fourier = np.fft.fft(signal, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N / 2))

plt.figure(figsize=(15, 6))
plt.plot(freq[L], psd[L].real)
plt.xlabel('Taajuus [Hz]')
plt.ylabel('Teho')
plt.axis([0, 10, 0, 6000])
st.pyplot(plt.gcf())
plt.close()

f_max = freq[L][psd[L] == np.max(psd[L])][0]
T = 1 / f_max
steps = f_max * np.max(t)

#st.write(f"Dominoiva askeltaajuus: **{f_max:.3f} Hz**")
#st.write(f"Jaksonaika (askelaika): **{T:.3f} s**")
st.metric("Askelmäärä (Fourier)", f"{np.round(steps):.0f}")

# =========================
# GPS: Reitti kartalla
# =========================
st.header("GPS-reitti kartalla")

gps_df = pd.read_csv('./Data/projekti_sijainti.csv')
gps_df.columns = gps_df.columns.str.strip()

# samat suodatukset
gps_df = gps_df[gps_df['Horizontal Accuracy (m)'] < 10]
gps_df = gps_df[gps_df['Vertical Accuracy (m)'] < 15]
gps_df = gps_df[gps_df['Velocity (m/s)'] < 8]
gps_df = gps_df[gps_df['Time (s)'] > 3]
gps_df = gps_df.dropna(subset=['Latitude (°)', 'Longitude (°)']).reset_index(drop=True)

lat1 = gps_df['Latitude (°)'].mean()
long1 = gps_df['Longitude (°)'].mean()

my_map = folium.Map(location=[lat1, long1], zoom_start=15)
folium.PolyLine(
    gps_df[['Latitude (°)', 'Longitude (°)']].values.tolist(),
    color='red',
    weight=3
).add_to(my_map)

st_folium(my_map, height=500)

# =========================
# Matka (Haversine)
# =========================
st.header("Kuljettu matka (Haversine)")

gps = pd.read_csv('./Data/projekti_sijainti.csv')
gps.columns = gps.columns.str.strip()
gps = gps.dropna(subset=['Latitude (°)', 'Longitude (°)', 'Time (s)']).sort_values('Time (s)').reset_index(drop=True)

gps = gps[gps['Horizontal Accuracy (m)'] < 10]
gps = gps[gps['Vertical Accuracy (m)'] < 15]
gps = gps[gps['Velocity (m/s)'] < 8]
gps = gps[gps['Time (s)'] > 3].reset_index(drop=True)

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

gps['dist_km'] = 0.0
if len(gps) > 1:
    gps.loc[1:, 'dist_km'] = haversine_km(
        gps['Longitude (°)'].values[:-1],
        gps['Latitude (°)'].values[:-1],
        gps['Longitude (°)'].values[1:],
        gps['Latitude (°)'].values[1:]
    )

gps['total_km'] = gps['dist_km'].cumsum()
total_distance_km = float(gps['total_km'].iloc[-1]) if len(gps) else 0.0
st.metric("Kuljettu matka (km)", f"{total_distance_km:.3f}")

plt.figure(figsize=(12, 5))
plt.plot(gps['Time (s)'], gps['total_km'])
plt.xlabel('Aika (s)')
plt.ylabel('Kokonaismatka (km)')
plt.grid()
st.pyplot(plt.gcf())
plt.close()

# =========================
# Askelpituus
# =========================
st.header("Askelpituus")

if jaksot > 0 and steps > 0:
    steplength1 = (total_distance_km * 1000) / jaksot
    steplength2 = (total_distance_km * 1000) / steps
    st.write(f"Askelpituus suodatetusta askelmäärästä: **{steplength1:.2f} m**")
    st.write(f"Askelpituus Fourier-analyysin askelmäärästä: **{steplength2:.2f} m**")
else:
    st.warning("Askelpituutta ei voi laskea (askeleita 0).")

# =========================
# Nopeus + keskinopeus
# =========================
st.header("Nopeus ja keskinopeus (GPS)")

t_max = gps_df['Time (s)'].max() if len(gps_df) else 0

plt.figure(figsize=(14, 6))
plt.plot(gps_df['Time (s)'], gps_df['Velocity (m/s)'], label="Nopeus")
plt.title('Nopeus')
plt.xlabel('Aika [s]')
plt.ylabel('Nopeus [m/s]')
plt.grid()
plt.axis([0, t_max, 0, 4])
plt.legend()
st.pyplot(plt.gcf())
plt.close()

avg_speed = gps_df['Velocity (m/s)'].mean() if len(gps_df) else 0
st.metric("Keskinopeus", f"{avg_speed:.2f} m/s")

