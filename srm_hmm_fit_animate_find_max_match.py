import numpy as np
from scipy import stats
import brainiak.eventseg.event
from brainiak.funcalign.srm import SRM

# --------------------
# Paths / constants
# --------------------
datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
songdir = '/jukebox/norman/jamalw/MES/data/songs/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

hrf = 0
numFeatures = 10

# run 1 durations
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = [
    'Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole',
    'Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique',
    'Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things',
    'The_Bird','Early_Summer'
]

# run 1 times (TR indices)
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973,2198,2377,2511])

songs2 = [
    'St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard',
    'Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia',
    'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle',
    'My_Favorite_Things', 'Blue_Monk','All_Blues'
]

# run 2 times (TR indices)
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])


# --------------------
# Matching metric
# --------------------
def boundary_match_score(human_bounds, model_bounds, tol_tr=3):
    """
    Compare internal boundaries only (drops 0 and end).
    Returns: f1, precision, recall, mean_abs_err, median_abs_err
    """
    hb = np.asarray(human_bounds, dtype=int)
    mb = np.asarray(model_bounds, dtype=int)

    # drop endpoints
    hb_i = hb[1:-1] if hb.size >= 3 else np.array([], dtype=int)
    mb_i = mb[1:-1] if mb.size >= 3 else np.array([], dtype=int)

    if hb_i.size == 0 and mb_i.size == 0:
        return 1.0, 1.0, 1.0, 0.0, 0.0
    if hb_i.size == 0 or mb_i.size == 0:
        return 0.0, 0.0, 0.0, np.inf, np.inf

    d_h_to_m = np.min(np.abs(hb_i[:, None] - mb_i[None, :]), axis=1)
    d_m_to_h = np.min(np.abs(mb_i[:, None] - hb_i[None, :]), axis=1)

    recall = np.mean(d_h_to_m <= tol_tr)
    precision = np.mean(d_m_to_h <= tol_tr)

    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))

    mean_abs_err = float(np.mean(d_h_to_m))
    median_abs_err = float(np.median(d_h_to_m))
    return float(f1), float(precision), float(recall), mean_abs_err, median_abs_err


# --------------------
# Load neural data
# --------------------
train = np.nan_to_num(stats.zscore(
    np.load(datadir + 'fdr_01_bil_mPFC_split_merge_run1_n25.npy'),
    axis=1, ddof=1
))
test = np.nan_to_num(stats.zscore(
    np.load(datadir + 'fdr_01_bil_mPFC_split_merge_run2_n25.npy'),
    axis=1, ddof=1
))

train_list = [train[:, :, i] for i in range(train.shape[2])]
test_list  = [test[:, :, i]  for i in range(test.shape[2])]


# --------------------
# Fit SRM once each way
# --------------------
print('Building/Training SRM (fit on run1, transform run2)')
srm_12 = SRM(n_iter=10, features=numFeatures)
srm_12.fit(train_list)
shared_run2 = srm_12.transform(test_list)
avg_response_run2 = sum(shared_run2) / len(shared_run2)

print('Building/Training SRM (fit on run2, transform run1)')
srm_21 = SRM(n_iter=10, features=numFeatures)
srm_21.fit(test_list)
shared_run1 = srm_21.transform(train_list)
avg_response_run1 = sum(shared_run1) / len(shared_run1)


# --------------------
# Loop over songs and score
# --------------------
tol_tr = 3
results = []

for song_name in songs1:
    song_number1 = songs1.index(song_name)
    song_number2 = songs2.index(song_name)

    # human bounds: include 0 and end (duration)
    human_bounds = np.load(f"{ann_dirs}{song_name}/{song_name}_beh_seg.npy") + hrf
    human_bounds = np.append(0, np.append(human_bounds, durs1[song_number1]))

    # crop each run to this song
    run2_start, run2_end = song_bounds2[song_number2], song_bounds2[song_number2 + 1]
    run1_start, run1_end = song_bounds1[song_number1], song_bounds1[song_number1 + 1]

    avg_response2_crop = avg_response_run2[:, run2_start:run2_end]
    avg_response1_crop = avg_response_run1[:, run1_start:run1_end]

    avg_response_combo = (avg_response2_crop + avg_response1_crop) / 2.0
    nTR = avg_response_combo.shape[1]

    K = len(human_bounds) - 1
    if K <= 1 or nTR <= 2:
        results.append((song_name, np.nan, np.nan, np.nan, np.nan, np.nan, K, nTR))
        continue

    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(avg_response_combo.T)

    bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

    # INCLUDE final boundary:
    bounds_w_zero_end = np.concatenate(([0], bounds, [nTR]))

    f1, prec, rec, mae, medae = boundary_match_score(human_bounds, bounds_w_zero_end, tol_tr=tol_tr)
    results.append((song_name, f1, prec, rec, mae, medae, K, nTR))

results_sorted = sorted(
    results,
    key=lambda x: (-np.nan_to_num(x[1], nan=-1.0), np.nan_to_num(x[4], nan=np.inf))
)

print("\n=== Boundary match results (higher F1 better) ===")
print(f"(tolerance = ±{tol_tr} TRs; endpoints included in arrays but ignored in scoring)\n")
for (song_name, f1, prec, rec, mae, medae, K, nTR) in results_sorted:
    print(f"{song_name:22s}  F1={f1:6.3f}  P={prec:6.3f}  R={rec:6.3f}  "
          f"MAE={mae:6.2f}  MedAE={medae:6.2f}  K={K:2d}  nTR={nTR:4d}")

best = results_sorted[0]
print(f"\nBest by F1: {best[0]} (F1={best[1]:.3f}, MAE={best[4]:.2f})")

