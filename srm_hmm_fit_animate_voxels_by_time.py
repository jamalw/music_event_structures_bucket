import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import brainiak.eventseg.event
from scipy import stats
from brainiak.funcalign.srm import SRM
from scipy.io import wavfile
from matplotlib.lines import Line2D

# ------------------------
# User settings
# ------------------------
song_name = 'Change_of_the_Guard'
numFeatures = 40
hrf = 0

datadir  = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
songdir  = '/jukebox/norman/jamalw/MES/data/songs/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

outdir = 'temp_plot_data'
os.makedirs(outdir, exist_ok=True)

wav_path = os.path.join(songdir, f'{song_name}.wav')

# What to render:
RENDER_EVENT_TEMPLATE_SIM = True
RENDER_TR_TR_SIM          = False
RENDER_FEATURES_BY_TIME   = False

# ------------------------
# Plot styling (bigger text)
# ------------------------
TITLE_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE  = 16

HUMAN_LW = 4.5
HMM_LW   = 4.5
OVERLAP_OFFSET_TR = 0.18  # only applied to HMM lines when overlap occurs

# Legend marker styling (boxes)
LEGEND_BOX_MARKER = 's'
LEGEND_BOX_SIZE = 10
LEGEND_HMM_EDGE = 'black'
LEGEND_HMM_FACE = 'white'
LEGEND_HMM_EDGE_LW = 2.0
LEGEND_HUMAN_FACE = 'black'
LEGEND_HUMAN_EDGE = 'black'
LEGEND_HUMAN_EDGE_LW = 2.0

# User-provided F1 to show in legend
F1_TO_DISPLAY = 1.0

# ------------------------
# Song bookkeeping
# ------------------------
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = [
    'Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole',
    'Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique',
    'Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things',
    'The_Bird','Early_Summer'
]
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973,2198,2377,2511])

songs2 = [
    'St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard',
    'Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia',
    'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle',
    'My_Favorite_Things', 'Blue_Monk','All_Blues'
]
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

song_number1 = songs1.index(song_name)
song_number2 = songs2.index(song_name)

# ------------------------
# Duration from WAV (seconds)
# ------------------------
sr_audio, audio = wavfile.read(wav_path)
song_duration_s = len(audio) / float(sr_audio)

# ------------------------
# Load human bounds (TR units)
# ------------------------
human_bounds = np.load(os.path.join(ann_dirs, song_name, f'{song_name}_beh_seg.npy')) + hrf
human_bounds = np.append(0, np.append(human_bounds, durs1[song_number1]))  # includes 0 and end

# ------------------------
# Load neural data
# ------------------------
train = np.nan_to_num(stats.zscore(
    np.load(os.path.join(datadir, 'fdr_01_bil_mPFC_split_merge_run1_n25.npy')),
    axis=1, ddof=1
))
test = np.nan_to_num(stats.zscore(
    np.load(os.path.join(datadir, 'fdr_01_bil_mPFC_split_merge_run2_n25.npy')),
    axis=1, ddof=1
))

train_list = [train[:, :, i] for i in range(train.shape[2])]
test_list  = [test[:, :, i]  for i in range(test.shape[2])]

# ------------------------
# SRM: fit run1 -> transform run2
# ------------------------
print('SRM: fit run1 -> transform run2')
srm = SRM(n_iter=10, features=numFeatures)
srm.fit(train_list)
shared_run2 = srm.transform(test_list)
avg_run2 = sum(shared_run2) / len(shared_run2)  # (features x TR)
avg_run2_crop = avg_run2[:, song_bounds2[song_number2]:song_bounds2[song_number2+1]]

# ------------------------
# SRM: fit run2 -> transform run1
# ------------------------
print('SRM: fit run2 -> transform run1')
srm = SRM(n_iter=10, features=numFeatures)
srm.fit(test_list)
shared_run1 = srm.transform(train_list)
avg_run1 = sum(shared_run1) / len(shared_run1)  # (features x TR)
avg_run1_crop = avg_run1[:, song_bounds1[song_number1]:song_bounds1[song_number1+1]]

# Combine shared responses
avg_response_combo = (avg_run2_crop + avg_run1_crop) / 2.0
nTR = avg_response_combo.shape[1]

print(f'nTR={nTR}, human_end={human_bounds[-1]}, audio_duration={song_duration_s:.2f}s')

# ------------------------
# Event segmentation
# ------------------------
K = len(human_bounds) - 1
ev = brainiak.eventseg.event.EventSegment(K)
ev.fit(avg_response_combo.T)

bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
bounds_aug = np.concatenate(([0], bounds, [nTR]))  # includes 0 and nTR for convenience

# ------------------------
# Helper: pace animation to song
# ------------------------
fps = nTR / song_duration_s
interval_ms = 1000.0 / fps  # ms per frame

def mux_audio(video_mp4, out_mp4):
    subprocess.check_call([
        'ffmpeg', '-y',
        '-i', video_mp4,
        '-i', wav_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        out_mp4
    ])

def render_matrix_video(matrix, title, ylab, out_stub):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(matrix, aspect='auto', interpolation='nearest', origin='lower')

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('TRs', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel(ylab, fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    # ---- Legend handles (boxes + F1 text) ----
    human_handle = Line2D(
        [0], [0],
        linestyle='None',
        marker=LEGEND_BOX_MARKER, markersize=LEGEND_BOX_SIZE,
        markerfacecolor=LEGEND_HUMAN_FACE,
        markeredgecolor=LEGEND_HUMAN_EDGE,
        markeredgewidth=LEGEND_HUMAN_EDGE_LW,
        label='Human bounds'
    )

    hmm_handle = Line2D(
        [0], [0],
        linestyle='None',
        marker=LEGEND_BOX_MARKER, markersize=LEGEND_BOX_SIZE,
        markerfacecolor=LEGEND_HMM_FACE,
        markeredgecolor=LEGEND_HMM_EDGE,
        markeredgewidth=LEGEND_HMM_EDGE_LW,
        label='HMM bounds'
    )

    f1_handle = Line2D(
        [0], [0],
        linestyle='None',
        marker=None,
        color='none',
        label=f'F1 = {F1_TO_DISPLAY:.2f}'
    )

    # Overlap handling:
    overlap_set = set(np.asarray(human_bounds, dtype=int).tolist())

    # Do not draw vertical lines at 0 (start)
    human_bounds_no0 = [b for b in human_bounds if int(b) != 0]
    hmm_bounds_no0   = [b for b in bounds_aug   if int(b) != 0]

    # Human lines (solid)
    for b in human_bounds_no0:
        ax.axvline(b, color='black', linewidth=HUMAN_LW, alpha=0.95, linestyle='-',
                   zorder=5)

    # HMM lines (dashed, offset if perfectly overlapping human)
    for b in hmm_bounds_no0:
        b_plot = b + (OVERLAP_OFFSET_TR if int(b) in overlap_set else 0.0)
        ax.axvline(b_plot, color='white', linewidth=HMM_LW, alpha=0.95, linestyle='--',
                   zorder=6)

    # Animated red cursor line (not in legend)
    cursor = ax.axvline(0, color='red', linewidth=3, zorder=7)

    ax.set_xlim(0, nTR - 1)

    ax.legend(
        handles=[human_handle, hmm_handle, f1_handle],
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=14,
        handlelength=1.2,
        handletextpad=0.8
    )

    plt.tight_layout()

    def update(frame):
        cursor.set_xdata([frame, frame])
        return (cursor,)

    anim = animation.FuncAnimation(
        fig, update,
        frames=range(nTR),
        interval=interval_ms,
        blit=True,
        repeat=False
    )

    silent_mp4 = os.path.join(outdir, f'{out_stub}_silent.mp4')
    with_audio = os.path.join(outdir, f'{out_stub}_with_music.mp4')

    print(f'Saving {silent_mp4} at fps={fps:.3f} (interval={interval_ms:.1f} ms)')
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(silent_mp4, writer=writer)

    print('Muxing audio...')
    mux_audio(silent_mp4, with_audio)

    print(f'Done: {with_audio}')
    plt.close(fig)

# ------------------------
# 1) Flattened + blocky: event-template similarity (K x TR)
# ------------------------
if RENDER_EVENT_TEMPLATE_SIM:
    state_seq = np.argmax(ev.segments_[0], axis=1)  # length nTR, values 0..K-1

    event_means = np.zeros((K, numFeatures))
    for k in range(K):
        idx = np.where(state_seq == k)[0]
        event_means[k, :] = 0 if len(idx) == 0 else avg_response_combo[:, idx].mean(axis=1)

    # cosine similarity: (K x nTR)
    X = avg_response_combo.T
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    En = event_means / (np.linalg.norm(event_means, axis=1, keepdims=True) + 1e-8)
    sim = (Xn @ En.T).T

    render_matrix_video(
        sim,
        title=f"Human and HMM Fit to mPFC ({song_name.replace('_', ' ')})",
        ylab='Event (k)',
        out_stub=f'{song_name}_K{numFeatures}_event_template_sim'
    )

# ------------------------
# 2) Optional: classic TR x TR similarity
# ------------------------
if RENDER_TR_TR_SIM:
    trtr = np.corrcoef(avg_response_combo.T)
    render_matrix_video(
        trtr,
        title=f"{song_name.replace('_', ' ')}: TR x TR similarity",
        ylab='TR',
        out_stub=f'{song_name}_K{numFeatures}_TRxTR'
    )

# ------------------------
# 3) Optional: features x time
# ------------------------
if RENDER_FEATURES_BY_TIME:
    render_matrix_video(
        avg_response_combo,
        title=f"{song_name.replace('_', ' ')}: SRM shared response",
        ylab='SRM feature',
        out_stub=f'{song_name}_K{numFeatures}_features_by_time'
    )

