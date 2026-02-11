import numpy as np
from scipy import stats
import brainiak.eventseg.event
from brainiak.funcalign.srm import SRM

# --------------------
# Paths / constants
# --------------------
datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

hrf = 0
tol_tr = 3  # boundary matching tolerance in TRs

# run 1 durations
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = [
    'Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole',
    'Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique',
    'Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things',
    'The_Bird','Early_Summer'
]

# run 1 bounds (TR indices)
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973,2198,2377,2511])

songs2 = [
    'St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard',
    'Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia',
    'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle',
    'My_Favorite_Things', 'Blue_Monk','All_Blues'
]

# run 2 bounds (TR indices)
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])


# --------------------
# Matching metric
# --------------------
def boundary_match_score(human_bounds, model_bounds, tol_tr=3):
    """
    Compare internal boundaries only (drops 0 and end) and computes:
    F1 (within tolerance), precision, recall, mean abs error, median abs error.
    Distances are in TRs.
    """
    hb = np.asarray(human_bounds, dtype=int)
    mb = np.asarray(model_bounds, dtype=int)

    hb_i = hb[1:-1] if hb.size >= 3 else np.array([], dtype=int)
    mb_i = mb[1:-1] if mb.size >= 3 else np.array([], dtype=int)

    if hb_i.size == 0 and mb_i.size == 0:
        return 1.0, 1.0, 1.0, 0.0, 0.0
    if hb_i.size == 0 or mb_i.size == 0:
        return 0.0, 0.0, 0.0, np.inf, np.inf

    d_h_to_m = np.min(np.abs(hb_i[:, None] - mb_i[None, :]), axis=1)
    d_m_to_h = np.min(np.abs(mb_i[:, None] - hb_i[None, :]), axis=1)

    recall = float(np.mean(d_h_to_m <= tol_tr))
    precision = float(np.mean(d_m_to_h <= tol_tr))
    f1 = 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))

    mae = float(np.mean(d_h_to_m))
    medae = float(np.median(d_h_to_m))
    return f1, precision, recall, mae, medae


# --------------------
# Load neural data once
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
# Sweep numFeatures
# --------------------
feature_grid = list(range(5, 51, 5))  # 5..50 step 5

all_results = {}  # numFeatures -> list of per-song tuples

for numFeatures in feature_grid:
    print(f"\n=== numFeatures={numFeatures} ===")

    # Fit SRM once each way
    print("  SRM: fit run1 -> transform run2")
    srm_12 = SRM(n_iter=10, features=numFeatures)
    srm_12.fit(train_list)
    shared_run2 = srm_12.transform(test_list)
    avg_response_run2 = sum(shared_run2) / len(shared_run2)

    print("  SRM: fit run2 -> transform run1")
    srm_21 = SRM(n_iter=10, features=numFeatures)
    srm_21.fit(test_list)
    shared_run1 = srm_21.transform(train_list)
    avg_response_run1 = sum(shared_run1) / len(shared_run1)

    results_nf = []

    for song_name in songs1:
        song_number1 = songs1.index(song_name)
        song_number2 = songs2.index(song_name)

        # human bounds for this song (include 0 and end)
        human_bounds = np.load(f"{ann_dirs}{song_name}/{song_name}_beh_seg.npy") + hrf
        human_bounds = np.append(0, np.append(human_bounds, durs1[song_number1]))

        # crop shared responses
        run2_start, run2_end = song_bounds2[song_number2], song_bounds2[song_number2 + 1]
        run1_start, run1_end = song_bounds1[song_number1], song_bounds1[song_number1 + 1]

        avg_response2_crop = avg_response_run2[:, run2_start:run2_end]
        avg_response1_crop = avg_response_run1[:, run1_start:run1_end]

        avg_response_combo = (avg_response2_crop + avg_response1_crop) / 2.0
        nTR = avg_response_combo.shape[1]

        K = len(human_bounds) - 1
        if K <= 1 or nTR <= 2:
            results_nf.append((song_name, np.nan, np.nan, np.nan, np.nan, np.nan, K, nTR))
            continue

        ev = brainiak.eventseg.event.EventSegment(K)
        ev.fit(avg_response_combo.T)

        bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
        bounds_w_zero_end = np.concatenate(([0], bounds, [nTR]))

        f1, prec, rec, mae, medae = boundary_match_score(human_bounds, bounds_w_zero_end, tol_tr=tol_tr)
        results_nf.append((song_name, f1, prec, rec, mae, medae, K, nTR))

    # store and print best song for this numFeatures
    all_results[numFeatures] = results_nf
    results_sorted = sorted(
        results_nf,
        key=lambda x: (-np.nan_to_num(x[1], nan=-1.0), np.nan_to_num(x[4], nan=np.inf))
    )
    best = results_sorted[0]
    mean_f1 = float(np.nanmean([r[1] for r in results_nf]))
    print(f"  Best song: {best[0]}  (F1={best[1]:.3f}, MAE={best[4]:.2f}, K={best[6]}, nTR={best[7]})")
    print(f"  Mean F1 across songs: {mean_f1:.3f}")

# --------------------
# Summaries across numFeatures
# --------------------
summary_rows = []
for nf in feature_grid:
    res = all_results[nf]
    best_f1 = float(np.nanmax([r[1] for r in res]))
    mean_f1 = float(np.nanmean([r[1] for r in res]))
    # best song at this nf
    best_song = sorted(
        res,
        key=lambda x: (-np.nan_to_num(x[1], nan=-1.0), np.nan_to_num(x[4], nan=np.inf))
    )[0][0]
    summary_rows.append((nf, best_song, best_f1, mean_f1))

print("\n=== Summary by numFeatures ===")
print("numFeat  best_song                 bestF1   meanF1")
for nf, best_song, best_f1, mean_f1 in summary_rows:
    print(f"{nf:6d}  {best_song:22s}  {best_f1:6.3f}  {mean_f1:6.3f}")

# Compare to numFeatures=10
nf10 = next((row for row in summary_rows if row[0] == 10), None)
if nf10 is not None:
    nf10_bestF1 = nf10[2]
    nf10_meanF1 = nf10[3]

    better_best = [row for row in summary_rows if row[2] > nf10_bestF1 + 1e-12]
    better_mean = [row for row in summary_rows if row[3] > nf10_meanF1 + 1e-12]

    print(f"\n=== Comparison to numFeatures=10 ===")
    print(f"numFeatures=10: bestF1={nf10_bestF1:.3f}, meanF1={nf10_meanF1:.3f}")

    if better_best:
        top = sorted(better_best, key=lambda x: -x[2])[0]
        print(f"Better *best-song* F1 found at numFeatures={top[0]} (bestF1={top[2]:.3f}, best_song={top[1]})")
    else:
        print("No numFeatures beat 10 on *best-song* F1.")

    if better_mean:
        topm = sorted(better_mean, key=lambda x: -x[3])[0]
        print(f"Better *mean* F1 found at numFeatures={topm[0]} (meanF1={topm[3]:.3f}, best_song={topm[1]})")
    else:
        print("No numFeatures beat 10 on *mean* F1.")

# Overall best (nf, song) combo
overall_best = None
for nf in feature_grid:
    for (song_name, f1, prec, rec, mae, medae, K, nTR) in all_results[nf]:
        if np.isnan(f1):
            continue
        key = (f1, -mae)  # maximize f1, then minimize mae
        if overall_best is None or key > (overall_best["f1"], -overall_best["mae"]):
            overall_best = dict(
                numFeatures=nf, song=song_name, f1=f1, precision=prec, recall=rec,
                mae=mae, medae=medae, K=K, nTR=nTR
            )

if overall_best is not None:
    print("\n=== Overall best combo ===")
    print(f"numFeatures={overall_best['numFeatures']}  song={overall_best['song']}")
    print(f"F1={overall_best['f1']:.3f}  P={overall_best['precision']:.3f}  "
          f"R={overall_best['recall']:.3f}  MAE={overall_best['mae']:.2f}  "
          f"MedAE={overall_best['medae']:.2f}  K={overall_best['K']}  nTR={overall_best['nTR']}")

