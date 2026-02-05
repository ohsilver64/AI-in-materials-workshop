import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.model_selection import KFold


import selfies as sf

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv, GINEConv, NNConv

from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning

import warnings

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


warnings.filterwarnings(
    "ignore",
    message=".*'squared' is deprecated.*",
    category=FutureWarning
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


#1 load and clean csv all


def load_and_clean_csv_all(
    csv_path,
    smiles_col="smiles",
    targets=("homo", "lumo", "gap"),
    drop_duplicates=True,
    verbose=True,
):
    """
    Load CSV and perform minimal global cleaning (scheme-2 ALL step).

    This function DOES NOT split data.
    It only:
      - loads CSV
      - keeps [smiles + targets]
      - drops NaNs
      - strips SMILES
      - optionally removes duplicate SMILES
      - coerces targets to float

    Parameters
    ----------
    csv_path : str
        Path to CSV file.

    smiles_col : str
        Column name for SMILES.

    targets : tuple/list
        Target column names (e.g. ("homo","lumo","gap")).

    drop_duplicates : bool
        Whether to drop duplicate SMILES (recommended).

    Returns
    -------
    df_clean : pd.DataFrame
        Clean dataframe with columns [smiles_col + targets].
    """

    # ---- load
    df = pd.read_csv(csv_path)

    # ---- column check
    needed = [smiles_col] + list(targets)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(df.columns)}")

    # ---- keep only needed columns
    df = df[needed].copy()

    # ---- drop NaNs
    df = df.dropna(subset=needed).reset_index(drop=True)

    # ---- clean SMILES
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""].reset_index(drop=True)

    # ---- optional: drop duplicate SMILES
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
        if verbose:
            print(f"Dropped {before - len(df)} duplicate SMILES")

    # ---- coerce targets to float
    for t in targets:
        df[t] = pd.to_numeric(df[t], errors="coerce")

    # ---- final NaN safety
    df = df.dropna(subset=list(targets)).reset_index(drop=True)

    if verbose:
        print("=== load_and_clean_csv_all ===")
        print(f"CSV: {csv_path}")
        print(f"SMILES column: {smiles_col}")
        print(f"Targets: {list(targets)}")
        print(f"Final molecule count: {len(df)}")

    return df

#2 
def rdkit_mol_valid_mask_all(df_clean, smiles_col="smiles", verbose=True):
    """
    Convert SMILES to RDKit Mol for ALL rows, build valid mask,
    and return RDKit-valid subset.

    Inputs:
        df_clean   : DataFrame (already cleaned: smiles + targets, no NaN)
        smiles_col: column name for SMILES

    Returns:
        mols_all   : pd.Series of RDKit Mol (invalid -> None), length = len(df_clean)
        valid_mask: np.ndarray(bool), same length
        df_all_v  : DataFrame, only RDKit-valid rows (reset index)
    """

    # ---- SMILES -> Mol (keep alignment with df_clean)
    def _smiles_to_mol(s):
        if not isinstance(s, str) or s.strip() == "":
            return None
        return Chem.MolFromSmiles(s)

    mols_all = df_clean[smiles_col].apply(_smiles_to_mol)

    # ---- valid mask
    valid_mask = mols_all.notna().to_numpy()

    n_total = len(mols_all)
    n_valid = int(valid_mask.sum())

    if verbose:
        print(f"RDKit valid molecules: {n_valid} / {n_total}")
        if n_valid < n_total:
            print(f"Dropped {n_total - n_valid} invalid SMILES")

    # ---- keep only valid rows (for downstream features)
    df_all_v = df_clean.loc[valid_mask].reset_index(drop=True)
    mols_all_v = mols_all.loc[valid_mask].reset_index(drop=True)

    return mols_all, valid_mask, df_all_v, mols_all_v

#3
def build_rdkit_feature_blocks_all(
    mols_all_v,
    DESC_FUNCS,
    build_morgan_var=True,
    verbose=True
):
    """
    Build FULL RDKit feature matrices ONCE (scheme-2 'ALL' step).

    Inputs:
        mols_all_v : pd.Series of RDKit Mol, already RDKit-valid (no None)
        DESC_FUNCS : list of (name, callable) for continuous descriptors
        build_morgan_var : whether to also build Morgan(radius=3, fpSize=2048)
        verbose : print shape report

    Returns:
        X_morgan_base_all : (N, 1024)
        X_morgan_var_all  : (N, 2048) if build_morgan_var else None
        X_maccs_all       : (N, 167)
        X_desc_raw_all    : (N, D) where D=len(DESC_FUNCS), may contain NaN
        desc_names        : list[str] length D
    """
    # ---- safety: ensure Series
    if not isinstance(mols_all_v, pd.Series):
        mols_all_v = pd.Series(mols_all_v)

    N = len(mols_all_v)
    if N == 0:
        raise ValueError("mols_all_v is empty.")

    # must be RDKit-valid
    if mols_all_v.isna().any():
        raise ValueError("mols_all_v contains NaN/None. Please filter to RDKit-valid mols first.")

    # ---- descriptor names
    desc_names = [n for n, _ in DESC_FUNCS]
    D = len(desc_names)

   
    # 1) Morgan base (1024)
   
    gen_base = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    fps_base = mols_all_v.apply(lambda m: gen_base.GetFingerprintAsNumPy(m))
    if fps_base.isna().any():
        raise ValueError("Morgan(base) produced NaN rows (unexpected if RDKit-valid).")

    X_morgan_base_all = np.stack(fps_base.to_numpy())
    # shape check
    if X_morgan_base_all.shape != (N, 1024):
        raise ValueError(f"Morgan(base) shape mismatch: got {X_morgan_base_all.shape}, expected {(N,1024)}")

    # 2) Morgan var (2048) optional
    X_morgan_var_all = None
    if build_morgan_var:
        gen_var = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        fps_var = mols_all_v.apply(lambda m: gen_var.GetFingerprintAsNumPy(m))
        if fps_var.isna().any():
            raise ValueError("Morgan(var) produced NaN rows (unexpected if RDKit-valid).")
        X_morgan_var_all = np.stack(fps_var.to_numpy())
        if X_morgan_var_all.shape != (N, 2048):
            raise ValueError(f"Morgan(var) shape mismatch: got {X_morgan_var_all.shape}, expected {(N,2048)}")

    # 3) MACCS (167)
    maccs_fp = mols_all_v.apply(lambda m: MACCSkeys.GenMACCSKeys(m))
    if maccs_fp.isna().any():
        raise ValueError("MACCS produced NaN rows (unexpected if RDKit-valid).")

    X_maccs_all = np.vstack([np.asarray(fp) for fp in maccs_fp.to_numpy()])
    if X_maccs_all.shape != (N, 167):
        raise ValueError(f"MACCS shape mismatch: got {X_maccs_all.shape}, expected {(N,167)}")

    # 4) Descriptors raw (N×D), allow NaN but keep alignment
    rows = []
    bad_rows = 0

    for m in mols_all_v:
        vals = []
        ok = True
        for _, fn in DESC_FUNCS:
            v = fn(m)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                ok = False
                break
            vals.append(float(v))

        if ok:
            rows.append(vals)
        else:
            bad_rows += 1
            rows.append([np.nan] * D)  # keep alignment

    X_desc_raw_all = np.asarray(rows, dtype=float)
    if X_desc_raw_all.shape != (N, D):
        raise ValueError(f"Desc(raw) shape mismatch: got {X_desc_raw_all.shape}, expected {(N,D)}")

    # final report
    if verbose:
        print("FULL RDKit feature blocks (scheme-2 ALL step):")
        print("  N molecules :", N)
        print("  Morgan base :", X_morgan_base_all.shape)
        print("  Morgan var  :", None if X_morgan_var_all is None else X_morgan_var_all.shape)
        print("  MACCS       :", X_maccs_all.shape)
        print("  Desc(raw)   :", X_desc_raw_all.shape)
        if bad_rows > 0:
            print(f"  [WARN] Descriptor invalid rows set to NaN: {bad_rows}")
        print("  desc_names  :", len(desc_names))

    return X_morgan_base_all, X_morgan_var_all, X_maccs_all, X_desc_raw_all, desc_names

#4
def make_train_test_indices(N, seed=42, test_size=0.2, stratify=None):
    """
    Scheme-2 split (index-based).

    Inputs:
        N (int): number of samples in the full dataset (after cleaning + RDKit-valid filtering)
        seed (int): random seed for reproducibility
        test_size (float or int): fraction (0-1) or absolute number of test samples
        stratify (array-like or None): optional labels for stratified split (length N)

    Returns:
        idx_tr (np.ndarray): 1D int array of train indices
        idx_te (np.ndarray): 1D int array of test indices
    """
    # ---- validate N
    if not isinstance(N, (int, np.integer)):
        raise TypeError(f"N must be an int, got {type(N)}")
    if N <= 1:
        raise ValueError(f"N must be > 1, got N={N}")

    # ---- validate test_size
    if isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
        if n_test <= 0 or n_test >= N:
            raise ValueError(f"Integer test_size must be in [1, N-1], got {test_size} for N={N}")
    else:
        # float
        ts = float(test_size)
        if not (0.0 < ts < 1.0):
            raise ValueError(f"Float test_size must be in (0,1), got {test_size}")
        n_test = int(round(N * ts))
        # enforce at least 1 and at most N-1
        n_test = max(1, min(N - 1, n_test))

    # ---- build dummy X just for index splitting
    idx = np.arange(N, dtype=int)

    # ---- stratify handling
    strat = None
    if stratify is not None:
        strat = np.asarray(stratify)
        if strat.shape[0] != N:
            raise ValueError(f"stratify must have length N={N}, got {strat.shape[0]}")
        # sklearn requires stratify not be all-unique when some classes too small; we don't enforce here,
        # let train_test_split raise a helpful error if stratification is impossible.

    # ---- split (returns arrays of indices)
    idx_tr, idx_te = train_test_split(
        idx,
        test_size=n_test if isinstance(test_size, (int, np.integer)) else test_size,
        random_state=seed,
        stratify=strat
    )

    # ---- enforce numpy int arrays, sorted for stability
    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_te = np.asarray(idx_te, dtype=int)

    idx_tr.sort()
    idx_te.sort()

    # ---- sanity checks: disjoint + cover
    if np.intersect1d(idx_tr, idx_te).size != 0:
        raise RuntimeError("Train/test indices overlap (should never happen).")
    if idx_tr.size + idx_te.size != N:
        raise RuntimeError("Train/test indices do not cover all samples (should never happen).")

    return idx_tr, idx_te

#5

def slice_targets_by_idx(df_all_v, targets=("homo", "lumo", "gap"), idx_tr=None, idx_te=None, as_array=False):
    """
    Scheme-2: slice targets by index arrays.

    Inputs:
        df_all_v (pd.DataFrame): RDKit-valid cleaned dataframe (length N)
        targets (list/tuple): target column names
        idx_tr (np.ndarray): train indices (int)
        idx_te (np.ndarray): test indices (int)
        as_array (bool): if True return numpy arrays, else return DataFrames

    Returns:
        Ytr, Yte: DataFrame (default) or ndarray (if as_array=True)
    """
    if idx_tr is None or idx_te is None:
        raise ValueError("idx_tr and idx_te must be provided.")

    if not isinstance(df_all_v, pd.DataFrame):
        raise TypeError(f"df_all_v must be a pandas DataFrame, got {type(df_all_v)}")

    targets = list(targets)
    missing = [t for t in targets if t not in df_all_v.columns]
    if missing:
        raise KeyError(f"Targets not found in df_all_v: {missing}. Available: {list(df_all_v.columns)}")

    N = len(df_all_v)
    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_te = np.asarray(idx_te, dtype=int)

    if idx_tr.ndim != 1 or idx_te.ndim != 1:
        raise ValueError("idx_tr and idx_te must be 1D arrays.")

    if idx_tr.size == 0 or idx_te.size == 0:
        raise ValueError("idx_tr and idx_te must be non-empty.")

    if idx_tr.min() < 0 or idx_te.min() < 0 or idx_tr.max() >= N or idx_te.max() >= N:
        raise IndexError(f"Indices out of range for df_all_v with N={N}.")

    # slice by integer positions
    Ytr = df_all_v.iloc[idx_tr][targets].reset_index(drop=True)
    Yte = df_all_v.iloc[idx_te][targets].reset_index(drop=True)

    if as_array:
        return Ytr.to_numpy(dtype=float), Yte.to_numpy(dtype=float)
    return Ytr, Yte

#6
def fit_binary_feature_filter_on_train(X_all, idx_tr, zero_var=True, max_zero_frac=0.99):
    """
    Scheme-2: fit a column keep-mask on TRAIN ONLY for (binary / sparse-like) features.

    Inputs:
        X_all: full feature matrix for all molecules, shape (N, P).
               Can be np.ndarray or scipy sparse matrix.
        idx_tr: 1D int array of train indices (positions into rows of X_all).
        zero_var: if True, drop columns with variance == 0 on train.
        max_zero_frac: if not None, drop columns where fraction of zeros on train > max_zero_frac.

    Returns:
        keep_mask: np.ndarray (bool) of shape (P,) indicating columns to keep.
    """
    if idx_tr is None:
        raise ValueError("idx_tr must be provided.")
    idx_tr = np.asarray(idx_tr, dtype=int)
    if idx_tr.ndim != 1 or idx_tr.size == 0:
        raise ValueError("idx_tr must be a non-empty 1D int array.")

    # --- slice train rows only
    Xtr = X_all[idx_tr]

    # --- number of features
    P = Xtr.shape[1]
    keep = np.ones(P, dtype=bool)

    # --- (a) zero-variance on TRAIN
    if zero_var:
        # works for dense; for sparse, var isn't directly available
        if hasattr(Xtr, "toarray") and not isinstance(Xtr, np.ndarray):
            Xtr_dense = Xtr.toarray()
            var = Xtr_dense.var(axis=0)
        else:
            var = np.asarray(Xtr).var(axis=0)
        keep &= (var > 0)

    # --- (b) too-sparse (too many zeros) on TRAIN
    if max_zero_frac is not None:
        # compute zero fraction robustly for dense/sparse
        n = Xtr.shape[0]

        if hasattr(Xtr, "getnnz") and not isinstance(Xtr, np.ndarray):
            # sparse matrix: nnz per column => zero_frac = 1 - nnz/n
            nnz = np.asarray(Xtr.getnnz(axis=0)).ravel()
            zero_frac = 1.0 - (nnz / float(n))
        else:
            Xtr_arr = np.asarray(Xtr)
            zero_frac = (Xtr_arr == 0).mean(axis=0)

        keep &= (zero_frac <= max_zero_frac)

    return keep


#7
def apply_mask_and_slice(X_all, idx_tr, idx_te, keep_mask):
    """
    Scheme-2: apply a TRAIN-fitted column mask, then slice into train/test by indices.

    Inputs:
        X_all: full feature matrix, shape (N, P). np.ndarray or scipy sparse matrix.
        idx_tr: 1D int array of train indices
        idx_te: 1D int array of test indices
        keep_mask: 1D bool array of length P (columns to keep)

    Returns:
        Xtr_f: filtered train matrix, shape (len(idx_tr), P_keep)
        Xte_f: filtered test matrix,  shape (len(idx_te), P_keep)
    """
    if idx_tr is None or idx_te is None:
        raise ValueError("idx_tr and idx_te must be provided.")
    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_te = np.asarray(idx_te, dtype=int)
    if idx_tr.ndim != 1 or idx_te.ndim != 1:
        raise ValueError("idx_tr and idx_te must be 1D int arrays.")
    if idx_tr.size == 0 or idx_te.size == 0:
        raise ValueError("idx_tr and idx_te must be non-empty.")

    if keep_mask is None:
        raise ValueError("keep_mask must be provided.")
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.ndim != 1:
        raise ValueError("keep_mask must be a 1D bool array.")

    # --- sanity: feature dimension match
    P = X_all.shape[1]
    if keep_mask.shape[0] != P:
        raise ValueError(f"keep_mask length {keep_mask.shape[0]} != n_features {P}")

    # --- slice rows first (faster for big N), then filter columns
    Xtr = X_all[idx_tr]
    Xte = X_all[idx_te]

    # works for dense and scipy sparse
    Xtr_f = Xtr[:, keep_mask]
    Xte_f = Xte[:, keep_mask]

    return Xtr_f, Xte_f

#8
def fit_desc_filter_and_scaler_on_train(
    X_desc_raw_all,
    idx_tr,
    desc_names,
    var_thresh=1e-12,
    corr_thresh=0.95,
):
    """
    Scheme-2 (train-only fit):
    Fit descriptor cleaning + filtering + scaling using TRAIN indices only.

    Inputs:
        X_desc_raw_all : np.ndarray, shape (N, D)  (may contain NaN/Inf)
        idx_tr         : 1D int array of train indices
        desc_names     : list of length D (descriptor names)
        var_thresh     : float, drop columns with variance <= var_thresh (train only)
        corr_thresh    : float, drop one of highly-correlated pairs with |corr| > corr_thresh (train only)

    Outputs:
        keep_idx_desc    : np.ndarray of kept column indices (into original D)
        kept_desc_names  : list of kept names (aligned with keep_idx_desc)
        scaler_desc      : fitted StandardScaler (fit on TRAIN, filtered cols)

    Strategy for NaN/Inf (train-only):
        - Column filtering: drop descriptor columns that contain ANY NaN/Inf in TRAIN.
        - Row handling: keep rows (do not drop molecules); NaNs are handled by column dropping.
          (If after column drop, TRAIN still has NaN/Inf -> raise error.)
    """
    X_all = np.asarray(X_desc_raw_all, dtype=float)
    idx_tr = np.asarray(idx_tr, dtype=int)

    if X_all.ndim != 2:
        raise ValueError("X_desc_raw_all must be a 2D array (N, D).")
    N, D = X_all.shape

    if len(desc_names) != D:
        raise ValueError(f"desc_names length {len(desc_names)} != number of columns {D}.")

    if idx_tr.ndim != 1 or idx_tr.size == 0:
        raise ValueError("idx_tr must be a non-empty 1D int array.")

    # --- TRAIN slice
    Xtr = X_all[idx_tr, :]  # shape (n_tr, D)

    # 1) train-only NaN/Inf column filtering (keep rows)
    finite_tr = np.isfinite(Xtr)  # False for NaN/Inf
    keep_finite_cols = finite_tr.all(axis=0)  # keep only cols fully finite in TRAIN

    if keep_finite_cols.sum() == 0:
        raise ValueError("All descriptor columns contain NaN/Inf in TRAIN; cannot proceed.")

    # reduce to finite cols
    Xtr0 = Xtr[:, keep_finite_cols]
    names0 = [n for n, k in zip(desc_names, keep_finite_cols) if k]
    idx0 = np.where(keep_finite_cols)[0]  # mapping to original indices

    # After dropping bad cols, train must be fully finite
    if not np.isfinite(Xtr0).all():
        raise ValueError("Descriptors still contain NaN/Inf in TRAIN after finite-column filtering.")

    # 2) variance filter (train)
    variances = np.var(Xtr0, axis=0)
    keep_var = variances > var_thresh

    # fallback: if everything dropped, keep all from step 1
    if keep_var.sum() == 0:
        keep_idx_desc = idx0
        kept_desc_names = names0
        Xtr_f = Xtr0
    else:
        Xtr_f = Xtr0[:, keep_var]
        kept_desc_names = [n for n, k in zip(names0, keep_var) if k]
        keep_idx_desc = idx0[keep_var]

    # 3) correlation filter (train)
    # If 0/1 columns after var filter, correlation filtering is unnecessary
    if Xtr_f.shape[1] >= 2:
        df_tr = pd.DataFrame(Xtr_f, columns=kept_desc_names)
        corr = df_tr.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [col for col in upper.columns if (upper[col] > corr_thresh).any()]
        keep_corr_names = [c for c in df_tr.columns if c not in to_drop]

        if len(keep_corr_names) == 0:
            # fallback: keep at least one column
            keep_corr_names = [df_tr.columns[0]]

        # map corr-kept names back to indices
        name_to_idx = {n: i for i, n in enumerate(kept_desc_names)}
        keep_corr_local_idx = np.array([name_to_idx[n] for n in keep_corr_names], dtype=int)

        # apply
        Xtr_f = Xtr_f[:, keep_corr_local_idx]
        keep_idx_desc = keep_idx_desc[keep_corr_local_idx]
        kept_desc_names = keep_corr_names

    # 4) scaler fit (train)
    scaler_desc = StandardScaler()
    scaler_desc.fit(Xtr_f)

    return keep_idx_desc, kept_desc_names, scaler_desc

#9
def apply_desc_filter_and_scaler(
    X_desc_raw_all,
    idx_tr,
    idx_te,
    keep_idx_desc,
    scaler_desc,
):
    """
    Scheme-2 (apply step):
    Apply TRAIN-fitted descriptor column filter + TRAIN-fitted scaler to get
    scaled descriptor matrices for train/test.

    Inputs:
        X_desc_raw_all : np.ndarray, shape (N, D) (may include NaN/Inf in dropped cols)
        idx_tr         : 1D int array of train indices
        idx_te         : 1D int array of test indices
        keep_idx_desc  : 1D int array of kept descriptor column indices (into original D)
        scaler_desc    : fitted StandardScaler (fit on TRAIN after filtering)

    Outputs:
        Xtr_desc_s : np.ndarray, shape (n_tr, d_kept)  (scaled)
        Xte_desc_s : np.ndarray, shape (n_te, d_kept)  (scaled)
    """
    X_all = np.asarray(X_desc_raw_all, dtype=float)
    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_te = np.asarray(idx_te, dtype=int)
    keep_idx_desc = np.asarray(keep_idx_desc, dtype=int)

    if X_all.ndim != 2:
        raise ValueError("X_desc_raw_all must be a 2D array (N, D).")

    if idx_tr.ndim != 1 or idx_te.ndim != 1:
        raise ValueError("idx_tr and idx_te must be 1D int arrays.")

    if keep_idx_desc.ndim != 1 or keep_idx_desc.size == 0:
        raise ValueError("keep_idx_desc must be a non-empty 1D int array.")

    # ---- slice rows then columns (kept cols are defined in original D space)
    Xtr = X_all[idx_tr, :][:, keep_idx_desc]
    Xte = X_all[idx_te, :][:, keep_idx_desc]

    # ---- safety: after column filtering, train/test must be finite
    # If not, it means keep_idx_desc was inconsistent with the fit function,
    # or the raw matrix changed.
    if not np.isfinite(Xtr).all():
        bad_cols = np.where(~np.isfinite(Xtr).all(axis=0))[0]
        raise ValueError(
            f"TRAIN descriptors contain NaN/Inf after applying keep_idx_desc. "
            f"Bad kept-col positions (0..d_kept-1): {bad_cols.tolist()}"
        )
    if not np.isfinite(Xte).all():
        bad_cols = np.where(~np.isfinite(Xte).all(axis=0))[0]
        raise ValueError(
            f"TEST descriptors contain NaN/Inf after applying keep_idx_desc. "
            f"Bad kept-col positions (0..d_kept-1): {bad_cols.tolist()}"
        )

    # ---- apply scaler (fit on train only)
    Xtr_s = scaler_desc.transform(Xtr)  # OK even if scaler was fit on same Xtr shape
    Xte_s = scaler_desc.transform(Xte)

    return Xtr_s, Xte_s

#10
def fit_selfies_tfidf_on_train(
    smiles_series_all,
    idx_tr,
    idx_te,
    ngram_range=(2, 5),
    min_df=2,
    max_df=0.95,
    # optional extra cleaning like你之前做的“zero-var / too-sparse”
    apply_bit_filter=True,
    max_zero_frac=0.99,
    verbose=True,
):
    """
    Scheme-2 (strict):
    Fit TF-IDF vectorizer on TRAIN ONLY (idx_tr), transform train/test,
    then (optionally) apply a train-only "too-sparse/zero-var" filter on the dense TF-IDF matrix.

    Inputs:
        smiles_series_all : pd.Series or list-like, length N (SMILES strings, RDKit-valid already)
        idx_tr, idx_te    : 1D int arrays (train/test indices into smiles_series_all)
        ngram_range       : tuple for char n-grams over SELFIES
        min_df, max_df    : TFIDF pruning settings (fit on train only)
        apply_bit_filter  : if True, apply train-only zero-var + max_zero_frac filter on dense TFIDF
        max_zero_frac     : drop columns where fraction of zeros in TRAIN > max_zero_frac
        verbose           : print shapes

    Outputs:
        Xtr_selfies_dense : np.ndarray shape (n_tr, d)
        Xte_selfies_dense : np.ndarray shape (n_te, d)
        selfies_vec       : fitted TfidfVectorizer
        keep_mask_selfies : np.ndarray bool shape (d_raw,) if filtering applied, else None
                            (mask corresponds to TF-IDF feature columns after vectorizer)
    """
    # ---- prepare indices
    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_te = np.asarray(idx_te, dtype=int)
    if idx_tr.ndim != 1 or idx_te.ndim != 1:
        raise ValueError("idx_tr and idx_te must be 1D int arrays.")

    # ---- prepare SMILES list
    if hasattr(smiles_series_all, "iloc"):
        smiles_all = smiles_series_all.astype(str).tolist()
        get_by_idx = lambda idx: [smiles_all[i] for i in idx]
    else:
        smiles_all = list(smiles_series_all)
        get_by_idx = lambda idx: [smiles_all[i] for i in idx]

    smiles_tr = get_by_idx(idx_tr)
    smiles_te = get_by_idx(idx_te)

    # ---- SMILES -> SELFIES (keep alignment; failures -> "")
    def _smiles_list_to_selfies(smiles_list):
        out = []
        for s in smiles_list:
            try:
                s = "" if s is None else str(s)
                s = s.strip()
                if s == "":
                    out.append("")
                else:
                    out.append(sf.encoder(s))
            except Exception:
                out.append("")
        return out

    selfies_tr = _smiles_list_to_selfies(smiles_tr)
    selfies_te = _smiles_list_to_selfies(smiles_te)

    # ---- TFIDF (fit on TRAIN only)
    selfies_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    Xtr_sparse = selfies_vec.fit_transform(selfies_tr)
    Xte_sparse = selfies_vec.transform(selfies_te)

    # ---- drop train all-zero columns (safety; usually not needed but harmless)
    keep_nonzero = np.asarray((Xtr_sparse != 0).sum(axis=0)).ravel() > 0
    Xtr_sparse = Xtr_sparse[:, keep_nonzero]
    Xte_sparse = Xte_sparse[:, keep_nonzero]

    # ---- dense
    Xtr_dense = Xtr_sparse.toarray()
    Xte_dense = Xte_sparse.toarray()

    keep_mask_selfies = None

    # ---- optional: your "train-only zero-var / too-sparse" filter on dense TFIDF
    if apply_bit_filter:
        # train-only variance > 0
        var = Xtr_dense.var(axis=0)
        keep = var > 0

        # train-only max_zero_frac
        if max_zero_frac is not None:
            zero_frac = (Xtr_dense == 0).mean(axis=0)
            keep &= (zero_frac <= max_zero_frac)

        keep_mask_selfies = keep

        Xtr_dense = Xtr_dense[:, keep_mask_selfies]
        Xte_dense = Xte_dense[:, keep_mask_selfies]

    if verbose:
        print("SELFIES TFIDF (train-fit) shapes:")
        print("  sparse raw(after nonzero):", Xtr_sparse.shape, Xte_sparse.shape)
        print("  dense:", Xtr_dense.shape, Xte_dense.shape)
        if apply_bit_filter:
            print(f"  applied train-only bit filter: kept {int(keep_mask_selfies.sum())} / {len(keep_mask_selfies)} cols")

    return Xtr_dense, Xte_dense, selfies_vec, keep_mask_selfies

#11

def build_feature_sets_9(
    Xtr_morgan, Xte_morgan,
    Xtr_selfies, Xte_selfies,
    Xtr_maccs, Xte_maccs,
    Xtr_desc_s, Xte_desc_s,
    verbose=True
):
    """
    Build 9 fusion feature sets from 4 base blocks (train/test pairs).

    Inputs (all MUST be numpy arrays, already aligned to the same train/test indices):
        - Morgan     : Xtr_morgan, Xte_morgan
        - SELFIES    : Xtr_selfies, Xte_selfies
        - MACCS      : Xtr_maccs, Xte_maccs
        - Desc_s     : Xtr_desc_s, Xte_desc_s  (descriptors already filtered + scaled)

    Output:
        feature_sets: dict[str, tuple[np.ndarray, np.ndarray]]
            name -> (Xtr, Xte) for 9 fusions
    """
    # ---- basic sanity checks
    def _check_pair(Xtr, Xte, name):
        if not isinstance(Xtr, np.ndarray) or not isinstance(Xte, np.ndarray):
            raise TypeError(f"{name} must be numpy arrays. Got {type(Xtr)} / {type(Xte)}")
        if Xtr.ndim != 2 or Xte.ndim != 2:
            raise ValueError(f"{name} must be 2D arrays. Got shapes {Xtr.shape} / {Xte.shape}")
        if Xtr.shape[0] <= 0 or Xte.shape[0] <= 0:
            raise ValueError(f"{name} has empty rows: {Xtr.shape} / {Xte.shape}")

    _check_pair(Xtr_morgan,  Xte_morgan,  "Morgan")
    _check_pair(Xtr_selfies, Xte_selfies, "SELFIES")
    _check_pair(Xtr_maccs,   Xte_maccs,   "MACCS")
    _check_pair(Xtr_desc_s,  Xte_desc_s,  "Desc_s")

    # ---- row alignment checks (train rows must match across blocks; test rows must match across blocks)
    ntr = Xtr_morgan.shape[0]
    nte = Xte_morgan.shape[0]
    for nm, Xtr, Xte in [
        ("SELFIES", Xtr_selfies, Xte_selfies),
        ("MACCS",   Xtr_maccs,   Xte_maccs),
        ("Desc_s",  Xtr_desc_s,  Xte_desc_s),
    ]:
        if Xtr.shape[0] != ntr or Xte.shape[0] != nte:
            raise ValueError(
                f"Row mismatch vs Morgan. Morgan train/test = {ntr}/{nte}, "
                f"{nm} train/test = {Xtr.shape[0]}/{Xte.shape[0]}"
            )

    def _hstack(parts_tr, parts_te):
        Xtr = np.hstack(parts_tr) if len(parts_tr) > 1 else parts_tr[0]
        Xte = np.hstack(parts_te) if len(parts_te) > 1 else parts_te[0]
        return Xtr, Xte

    feature_sets = {
        # 1) single blocks
        "Morgan": (Xtr_morgan, Xte_morgan),
        "SELFIES": (Xtr_selfies, Xte_selfies),
        "Descriptors": (Xtr_desc_s, Xte_desc_s),

        # 2) two-way fusions
        "Morgan+Desc": _hstack([Xtr_morgan, Xtr_desc_s], [Xte_morgan, Xte_desc_s]),
        "Morgan+SELFIES": _hstack([Xtr_morgan, Xtr_selfies], [Xte_morgan, Xte_selfies]),
        "SELFIES+Desc": _hstack([Xtr_selfies, Xtr_desc_s], [Xte_selfies, Xte_desc_s]),

        # 3) three-way
        "Morgan+SELFIES+Desc": _hstack(
            [Xtr_morgan, Xtr_selfies, Xtr_desc_s],
            [Xte_morgan, Xte_selfies, Xte_desc_s],
        ),
        "Morgan+MACCS+Desc": _hstack(
            [Xtr_morgan, Xtr_maccs, Xtr_desc_s],
            [Xte_morgan, Xte_maccs, Xte_desc_s],
        ),

        # 4) full
        "ALL (Morgan+SELFIES+MACCS+Desc)": _hstack(
            [Xtr_morgan, Xtr_selfies, Xtr_maccs, Xtr_desc_s],
            [Xte_morgan, Xte_selfies, Xte_maccs, Xte_desc_s],
        ),
    }

    if verbose:
        print("Fusion feature shapes (9 sets):")
        for k, (Xtr, Xte) in feature_sets.items():
            print(f"  {k:32s} | train {Xtr.shape} | test {Xte.shape}")

    return feature_sets

#12

def get_model_zoo(seed=42, pca_dim=100):
    """
    Return a dict of 11 models (6 tree/boosting + 5 linear/kernel/knn/gpr-style),
    using your existing imports.

    Output:
        models: dict[str, estimator]
            Each value is a fresh sklearn-compatible estimator (or Pipeline).
    """
    SEED = int(seed)

    # ---- 6 "tree-style" models (no scaling needed)
    tree_models = {
        "RF": RandomForestRegressor(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=800,
            random_state=SEED,
            n_jobs=-1
        ),
        "HistGB": HistGradientBoostingRegressor(
            random_state=SEED,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400
        ),
        "XGB": XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
            verbose=-1,
        ),
        # linear baseline (you used this in the "tree table" previously)
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-6, 6, 25)))
        ]),
    }

    # ---- 5 "linear/kernel/knn/gpr" models (your earlier set, excluding MLP)
    # Note: PCA is included for SVR/KNN/GPR for stability in high-dim.
    linear_kernel_models = {
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=1e-3, max_iter=20000, random_state=SEED)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000, random_state=SEED)),
        ]),
        "SVR+PCA": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(pca_dim), random_state=SEED)),
            ("model", SVR(C=10.0, gamma="scale", epsilon=0.1)),
        ]),
        "KNN+PCA": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(pca_dim), random_state=SEED)),
            ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
        "GPR+PCA50": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=50, random_state=SEED)),
            ("model", GaussianProcessRegressor(
                kernel=C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3),
                random_state=SEED,
                normalize_y=True
            )),
        ]),
    }

    # ---- merge into 11-model zoo
    models = {}
    models.update(tree_models)
    models.update(linear_kernel_models)

    return models

#13
def evaluate_feature_model_grid_one_seed(
    feature_sets,
    models,
    Ytr,
    Yte,
    return_predictions=False,
):
    """
    Evaluate (feature_set × model) grid for ONE seed / ONE split.

    Inputs:
        feature_sets: dict[str, (Xtr, Xte)]
            9 fusions (or any number). Xtr/Xte must be numpy arrays (dense).
        models: dict[str, estimator]
            Model zoo (e.g., 11 models). Each must support .fit(X,y) and .predict(X).
        Ytr, Yte: pd.DataFrame or np.ndarray
            Targets aligned to Xtr/Xte rows.
            If DataFrame: must contain columns ['homo','lumo','gap'] (or at least these three).
            If ndarray: assumed shape (n_samples, 3) in order [homo, lumo, gap].

    Outputs:
        df_seed_results: pd.DataFrame
            One row per (feature, model) containing:
              - R2/RMSE/MAE for HOMO
              - R2/RMSE/MAE for LUMO
              - R2/RMSE/MAE for GAP_direct
              - R2/RMSE/MAE for GAP_derived (from LUMO_pred - HOMO_pred)
              - Delta_GAP = R2_GAP_derived - R2_GAP_direct

        If return_predictions=True:
            returns (df_seed_results, pred_store)
            pred_store is a dict keyed by (feature_name, model_name) -> dict of preds
    """
    # ---------- helpers ----------
    def _to_col(Y, colname, colidx):
        if isinstance(Y, pd.DataFrame):
            return Y[colname].to_numpy()
        Y = np.asarray(Y)
        return Y[:, colidx]

    def _rmse(y_true, y_pred):
        # avoid squared=False deprecation differences across sklearn versions
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))

    def _safe_clone(est):
        # use sklearn.clone when possible; fall back to manual param copy
        try:
            return clone(est)
        except Exception:
            return est.__class__(**est.get_params())

    def _safe_fit_predict(est, Xtr, ytr, Xte, seed_for_pca=42):
        """
        Fit and predict, with a safety fix:
        if Pipeline contains PCA and n_components is too large for (n_samples, n_features),
        shrink PCA components automatically.
        """
        est2 = _safe_clone(est)

        if isinstance(est2, Pipeline):
            step_names = [name for name, _ in est2.steps]
            if "pca" in step_names:
                # determine k
                pca_step = est2.named_steps["pca"]
                k = pca_step.n_components
                nsamp, nfeat = int(Xtr.shape[0]), int(Xtr.shape[1])
                k2 = int(min(k, nsamp, nfeat))
                if k2 < 1:
                    k2 = 1
                if k2 != k:
                    # rebuild pipeline with adjusted PCA
                    new_steps = []
                    for name, step in est2.steps:
                        if name == "pca":
                            new_steps.append(("pca", PCA(n_components=k2, random_state=seed_for_pca)))
                        else:
                            new_steps.append((name, step))
                    est2 = Pipeline(new_steps)

        est2.fit(Xtr, ytr)
        pred_tr = est2.predict(Xtr)
        pred_te = est2.predict(Xte)
        return pred_tr, pred_te, est2

    # ---------- extract targets ----------
    ytr_h = _to_col(Ytr, "homo", 0)
    ytr_l = _to_col(Ytr, "lumo", 1)
    ytr_g = _to_col(Ytr, "gap",  2)

    yte_h = _to_col(Yte, "homo", 0)
    yte_l = _to_col(Yte, "lumo", 1)
    yte_g = _to_col(Yte, "gap",  2)

    rows = []
    pred_store = {} if return_predictions else None

    # ---------- main loop ----------
    for feat_name, (Xtr, Xte) in feature_sets.items():
        Xtr = np.asarray(Xtr)
        Xte = np.asarray(Xte)

        for model_name, est in models.items():

            # ---- HOMO
            p_h_tr, p_h_te, est_h = _safe_fit_predict(est, Xtr, ytr_h, Xte)

            # ---- LUMO
            p_l_tr, p_l_te, est_l = _safe_fit_predict(est, Xtr, ytr_l, Xte)

            # ---- GAP direct
            p_g_tr, p_g_te, est_g = _safe_fit_predict(est, Xtr, ytr_g, Xte)

            # ---- GAP derived
            p_gd_tr = p_l_tr - p_h_tr
            p_gd_te = p_l_te - p_h_te

            # ---- metrics
            r2_h = float(r2_score(yte_h, p_h_te))
            r2_l = float(r2_score(yte_l, p_l_te))
            r2_gd = float(r2_score(yte_g, p_g_te))
            r2_gv = float(r2_score(yte_g, p_gd_te))  # derived

            rmse_h = _rmse(yte_h, p_h_te)
            rmse_l = _rmse(yte_l, p_l_te)
            rmse_gd = _rmse(yte_g, p_g_te)
            rmse_gv = _rmse(yte_g, p_gd_te)

            mae_h = float(mean_absolute_error(yte_h, p_h_te))
            mae_l = float(mean_absolute_error(yte_l, p_l_te))
            mae_gd = float(mean_absolute_error(yte_g, p_g_te))
            mae_gv = float(mean_absolute_error(yte_g, p_gd_te))

            rows.append({
                "feature": feat_name,
                "model": model_name,

                "R2_HOMO": r2_h,
                "RMSE_HOMO": rmse_h,
                "MAE_HOMO": mae_h,

                "R2_LUMO": r2_l,
                "RMSE_LUMO": rmse_l,
                "MAE_LUMO": mae_l,

                "R2_GAP_direct": r2_gd,
                "RMSE_GAP_direct": rmse_gd,
                "MAE_GAP_direct": mae_gd,

                "R2_GAP_derived": r2_gv,
                "RMSE_GAP_derived": rmse_gv,
                "MAE_GAP_derived": mae_gv,

                "Delta_GAP": r2_gv - r2_gd,
            })

            if return_predictions:
                pred_store[(feat_name, model_name)] = {
                    "HOMO_tr": p_h_tr, "HOMO_te": p_h_te,
                    "LUMO_tr": p_l_tr, "LUMO_te": p_l_te,
                    "GAP_direct_tr": p_g_tr, "GAP_direct_te": p_g_te,
                    "GAP_derived_tr": p_gd_tr, "GAP_derived_te": p_gd_te,
                }

    df_seed_results = pd.DataFrame(rows)

    if return_predictions:
        return df_seed_results, pred_store
    return df_seed_results

#14
def run_grid_over_seeds(
    df_all_v,
    X_morgan_base_all,
    X_maccs_all,
    X_desc_raw_all,
    desc_names,
    seeds,
    test_size=0.2,
    smiles_col="smiles",
    targets=("homo", "lumo", "gap"),
    # ---- TFIDF / SELFIES params (科研严格：每个seed都fit在train)
    ngram_range=(2, 5),
    min_df=2,
    max_df=0.95,
    selfies_max_zero_frac=0.99,
    # ---- binary filters (Morgan/MACCS)
    morgan_max_zero_frac=0.99,
    maccs_max_zero_frac=None,  
    # ---- descriptor filters
    var_thresh_desc=1e-12,
    corr_thresh_desc=0.95,
    # ---- choose which morgan to use in fusion
    use_morgan="base",  # "base" or "var" (if you also pass X_morgan_var_all in future)
    # ---- models
    models=None,  # if None -> get_model_zoo()
    # ---- verbosity
    verbose=True,
):
    """
    Run full grid (9 feature fusions × models × 1 seed) over multiple seeds.

    Inputs:
        df_all_v: cleaned + RDKit-valid dataframe (must include smiles_col + targets)
        X_morgan_base_all: (N, 1024) full Morgan base matrix aligned to df_all_v
        X_maccs_all: (N, 167) full MACCS matrix aligned to df_all_v
        X_desc_raw_all: (N, D) raw descriptors (may include NaN) aligned to df_all_v
        desc_names: list of descriptor names (len D)
        seeds: list[int]
        test_size: float

    Outputs:
        df_all_results: long table with columns:
            seed, feature, model, (metrics columns...)
        df_summary: aggregated mean/std by (feature, model) for each metric
    """
    # ---------- basic checks ----------
    N = len(df_all_v)
    if X_morgan_base_all.shape[0] != N:
        raise ValueError(f"X_morgan_base_all rows {X_morgan_base_all.shape[0]} != N {N}")
    if X_maccs_all.shape[0] != N:
        raise ValueError(f"X_maccs_all rows {X_maccs_all.shape[0]} != N {N}")
    if X_desc_raw_all.shape[0] != N:
        raise ValueError(f"X_desc_raw_all rows {X_desc_raw_all.shape[0]} != N {N}")
    if isinstance(targets, tuple):
        targets = list(targets)

    # models default
    if models is None:
        models = get_model_zoo()

    # ---------- per-seed loop ----------
    all_seed_dfs = []

    for sd in seeds:
        if verbose:
            print(f"\n=== Seed {sd} ===")

        # ---- 1) indices split (scheme-2)
        idx_tr, idx_te = make_train_test_indices(N=N, seed=sd, test_size=test_size, stratify=None)

        # ---- 2) slice targets
        Ytr, Yte = slice_targets_by_idx(df_all_v, targets=targets, idx_tr=idx_tr, idx_te=idx_te)

        # ---- 3) Morgan + MACCS: fit mask on train, apply to train/test
        # choose morgan block
        if use_morgan.lower() != "base":
            raise ValueError("Currently only use_morgan='base' supported in this function signature.")

        keep_morgan = fit_binary_feature_filter_on_train(
            X_all=X_morgan_base_all,
            idx_tr=idx_tr,
            zero_var=True,
            max_zero_frac=morgan_max_zero_frac
        )
        Xtr_morgan, Xte_morgan = apply_mask_and_slice(
            X_all=X_morgan_base_all,
            idx_tr=idx_tr,
            idx_te=idx_te,
            keep_mask=keep_morgan
        )

        keep_maccs = fit_binary_feature_filter_on_train(
            X_all=X_maccs_all,
            idx_tr=idx_tr,
            zero_var=True,
            max_zero_frac=maccs_max_zero_frac
        )
        Xtr_maccs, Xte_maccs = apply_mask_and_slice(
            X_all=X_maccs_all,
            idx_tr=idx_tr,
            idx_te=idx_te,
            keep_mask=keep_maccs
        )

        # ---- 4) descriptors: fit filters+scaler on train, apply to train/test
        keep_idx_desc, kept_desc_names, scaler_desc = fit_desc_filter_and_scaler_on_train(
            X_desc_raw_all=X_desc_raw_all,
            idx_tr=idx_tr,
            desc_names=desc_names,
            var_thresh=var_thresh_desc,
            corr_thresh=corr_thresh_desc,
        )
        Xtr_desc_s, Xte_desc_s = apply_desc_filter_and_scaler(
            X_desc_raw_all=X_desc_raw_all,
            idx_tr=idx_tr,
            idx_te=idx_te,
            keep_idx_desc=keep_idx_desc,
            scaler_desc=scaler_desc
        )

        # ---- 5) SELFIES TFIDF: fit on train only , transform test
        Xtr_selfies_dense, Xte_selfies_dense, selfies_vec, keep_mask_selfies = fit_selfies_tfidf_on_train(
            smiles_series=df_all_v[smiles_col],
            idx_tr=idx_tr,
            idx_te=idx_te,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_zero_frac=selfies_max_zero_frac
        )

        # ---- 6) 9 fusions
        feature_sets = build_feature_sets_9(
            Xtr_morgan=Xtr_morgan, Xte_morgan=Xte_morgan,
            Xtr_selfies=Xtr_selfies_dense, Xte_selfies=Xte_selfies_dense,
            Xtr_maccs=Xtr_maccs, Xte_maccs=Xte_maccs,
            Xtr_desc_s=Xtr_desc_s, Xte_desc_s=Xte_desc_s,
            verbose=False
        )

        # ---- 7) evaluate grid for this seed
        df_seed = evaluate_feature_model_grid_one_seed(
            feature_sets=feature_sets,
            models=models,
            Ytr=Ytr,
            Yte=Yte,
            return_predictions=False
        )

        # add seed column
        df_seed.insert(0, "seed", sd)
        all_seed_dfs.append(df_seed)

    # ---------- concat all seeds ----------
    df_all_results = pd.concat(all_seed_dfs, ignore_index=True)

    # ---------- summary mean/std ----------
    metric_cols = [
        "R2_HOMO","RMSE_HOMO","MAE_HOMO",
        "R2_LUMO","RMSE_LUMO","MAE_LUMO",
        "R2_GAP_direct","RMSE_GAP_direct","MAE_GAP_direct",
        "R2_GAP_derived","RMSE_GAP_derived","MAE_GAP_derived",
        "Delta_GAP",
    ]
    present_metric_cols = [c for c in metric_cols if c in df_all_results.columns]

    agg = {}
    for c in present_metric_cols:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_std"]  = (c, "std")

    df_summary = (
        df_all_results
        .groupby(["feature", "model"], as_index=False)
        .agg(**agg)
        .sort_values("R2_GAP_derived_mean", ascending=False, ignore_index=True)
        if "R2_GAP_derived_mean" in [k for k in agg.keys()]
        else df_all_results.groupby(["feature","model"], as_index=False).agg(**agg)
    )

    return df_all_results, df_summary

#15

def plot_multi_seed_stability(
    df_all_results,
    targets=("HOMO", "LUMO", "GAP_direct", "GAP_derived"),
    # if True: each box = (feature, model); if False: aggregate models -> each box = feature
    group_by_model=True,
    # metric to visualize: "R2" or "RMSE" or "MAE"
    metric="R2",
    # show top-k groups (by mean of the chosen metric) to keep plot readable
    top_k=20,
    # sort by mean score (descending for R2, ascending for RMSE/MAE)
    sort_by="mean",
    figsize=(14, 4),
    rotate_xticks=45,
    show=True,
):
    """
    Plot multi-seed stability (boxplots) for 4 targets:
      - HOMO
      - LUMO
      - GAP_direct (directly predicted)
      - GAP_derived (LUMO_pred - HOMO_pred)

    Input:
        df_all_results: long table from run_grid_over_seeds()
          must include columns: seed, feature, model and metric columns:
            R2_HOMO, R2_LUMO, R2_GAP_direct, R2_GAP_derived
            (or RMSE_*, MAE_* if metric != R2)

    Output:
        fig, axes, df_rank (ranking table used for plotting)
    """

    import numpy as _np
    import pandas as _pd

    # --------- map naming ----------
    metric = metric.upper()
    if metric not in ("R2", "RMSE", "MAE"):
        raise ValueError("metric must be one of: 'R2', 'RMSE', 'MAE'")

    col_map = {
        "HOMO": f"{metric}_HOMO",
        "LUMO": f"{metric}_LUMO",
        "GAP_direct": f"{metric}_GAP_direct",
        "GAP_derived": f"{metric}_GAP_derived",
    }

    missing = [col_map[t] for t in targets if col_map[t] not in df_all_results.columns]
    if missing:
        raise KeyError(f"df_all_results is missing required columns: {missing}")

    # --------- define grouping key ----------
    dfp = df_all_results.copy()
    if group_by_model:
        dfp["group"] = dfp["feature"].astype(str) + " | " + dfp["model"].astype(str)
    else:
        dfp["group"] = dfp["feature"].astype(str)

    # --------- ranking to pick top_k groups ----------
    # Decide sorting direction
    higher_is_better = (metric == "R2")
    ascending = not higher_is_better

    # rank by mean score on GAP_derived by default (most relevant stability target)
    rank_target = "GAP_derived" if "GAP_derived" in targets else targets[0]
    score_col = col_map[rank_target]

    df_rank = (
        dfp.groupby("group")[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=ascending)
        .reset_index(drop=True)
    )

    if top_k is not None:
        df_rank = df_rank.head(int(top_k))

    keep_groups = set(df_rank["group"])
    dfp = dfp[dfp["group"].isin(keep_groups)].copy()

    # keep x-order consistent with ranking
    order = df_rank["group"].tolist()
    dfp["group"] = _pd.Categorical(dfp["group"], categories=order, ordered=True)

    # --------- plotting (no seaborn; matplotlib only) ----------
    import matplotlib.pyplot as plt

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=(metric == "R2"))

    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, targets):
        ycol = col_map[t]
        # collect data per group in the ranked order
        data = [dfp.loc[dfp["group"] == g, ycol].dropna().values for g in order]

        ax.boxplot(
            data,
            vert=True,
            patch_artist=False,
            showmeans=True,
            meanline=True,
        )

        ax.set_title(f"{metric} stability: {t}")
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=rotate_xticks, ha="right")

        if metric == "R2":
            ax.set_ylabel("R²")
        else:
            ax.set_ylabel(metric)

        ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes, df_rank



#----------------------15 function for cv and feature/model select-------------------
#----------------------for use those 15 function-------------------------------------

#
# =========================
# Multi-seed CV plotting
# requires: df_all_results with columns:
#   ["seed","feature","model",
#    "R2_HOMO","R2_LUMO","R2_GAP_direct","R2_GAP_derived", ...]
# =========================

def make_cv_summary_tables(df_all_results):
    df = df_all_results.copy()
    df["combo"] = df["model"].astype(str) + " + " + df["feature"].astype(str)

    metrics = ["R2_HOMO", "R2_LUMO", "R2_GAP_direct", "R2_GAP_derived"]
    grp = df.groupby("combo")[metrics].agg(["mean", "std", "min", "max"])
    # flatten columns
    grp.columns = [f"{m}_{stat}" for (m, stat) in grp.columns]
    grp = grp.reset_index()

    # also keep split-out model/feature for readability
    grp["model"] = grp["combo"].str.split(" + ", n=1).str[0]
    grp["feature"] = grp["combo"].str.split(" + ", n=1).str[1]

    # sort by GAP_derived mean (usually what you care about), fallback if missing
    sort_key = "R2_GAP_derived_mean" if "R2_GAP_derived_mean" in grp.columns else "R2_GAP_direct_mean"
    grp = grp.sort_values(sort_key, ascending=False).reset_index(drop=True)
    return grp


def plot_cv_boxplots(df_all_results, top_k=20, sort_by="mean"):
    """
    Plot 4 panels (HOMO/LUMO/GAP_direct/GAP_derived) as horizontal boxplots
    across seeds, for top_k combos ranked by mean R2 of that target.
    """
    df = df_all_results.copy()
    df["combo"] = df["model"].astype(str) + " + " + df["feature"].astype(str)

    targets = [
        ("R2_HOMO", "HOMO"),
        ("R2_LUMO", "LUMO"),
        ("R2_GAP_direct", "GAP (direct)"),
        ("R2_GAP_derived", "GAP (derived: LUMO-HOMO)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)

    for ax, (col, title) in zip(axes, targets):
        # rank combos for this target
        stats = (
            df.groupby("combo")[col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        # pick top_k combos
        top_combos = stats["combo"].head(top_k).tolist()
        sub = df[df["combo"].isin(top_combos)].copy()

        # ensure plotting order matches ranking
        order = stats.set_index("combo").loc[top_combos].reset_index()["combo"].tolist()

        # collect arrays for boxplot
        data = [sub[sub["combo"] == c][col].values for c in order]

        ax.boxplot(
            data,
            vert=False,
            labels=order,
            showmeans=True,
            meanline=True,
            whis=(5, 95),   # robust whiskers
        )
        ax.set_title(f"Multi-seed CV: {title}\n(top {top_k} by mean R²)")
        ax.set_xlabel("R²")
        ax.grid(True, axis="x", alpha=0.25)
        # make labels smaller
        ax.tick_params(axis="y", labelsize=7)

        # add mean values as text (right side)
        means = [np.mean(d) if len(d) else np.nan for d in data]
        for i, mu in enumerate(means, start=1):
            ax.text(mu, i, f"{mu:.2f}", va="center", ha="left", fontsize=7)

    plt.tight_layout()
    plt.show()












# ----------------------------for stacking--------------------------------------------
# helpers (safe + metrics)
# ----------------------------
def _safe_clone(est):
    try:
        return clone(est)
    except Exception:
        return est.__class__(**est.get_params())


def _metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def parse_combo(combo_str):
    """
    'Model + FeatureName' -> ('Model', 'FeatureName')
    """
    if not isinstance(combo_str, str) or " + " not in combo_str:
        raise ValueError(f"Bad combo string: {combo_str!r} (expected 'Model + Feature').")
    model, feat = combo_str.split(" + ", 1)
    return model.strip(), feat.strip()


def oof_predictions_one_model(Xtr, Xte, ytr, est, kfold=5, seed=0):
    """
    Strict OOF:
      - OOF preds for train via KFold
      - full-train fit preds for test
    """
    Xtr = np.asarray(Xtr)
    Xte = np.asarray(Xte)
    ytr = np.asarray(ytr).ravel()

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    oof = np.zeros(Xtr.shape[0], dtype=float)

    for tr_idx, va_idx in kf.split(Xtr):
        m = _safe_clone(est)
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        oof[va_idx] = m.predict(Xtr[va_idx])

    m_full = _safe_clone(est)
    m_full.fit(Xtr, ytr)
    te_pred = m_full.predict(Xte)

    return oof, te_pred


# ----------------------------
# STRICT stacking (ONE seed)
# ----------------------------
def strict_stacking_gap_one_seed(
    df_all_v,
    smiles_col,
    targets,
    X_morgan_base_all,
    X_maccs_all,
    X_desc_raw_all,
    desc_names,
    seed=0,
    test_size=0.2,

    homo_combo="XGB + Morgan+MACCS+Desc",
    lumo_combo="LightGBM + Morgan+MACCS+Desc",

    gap_direct_combos=("RidgeCV + Morgan+MACCS+Desc",),

    kfold=5,
    pca_dim=100,
    max_zero_frac=0.99,
    var_thresh=1e-12,
    corr_thresh=0.95,
):
    """
    Returns:
      df_gap_compare: per gap combo metrics (direct vs stacked), sorted by R2_stacked desc
      pred_bundle   : useful arrays (idx, y, preds)
      recipe        : reproducibility info
    """

    # ---------- split
    N = len(df_all_v)
    idx_tr, idx_te = make_train_test_indices(N, seed=seed, test_size=test_size)

    # ---------- targets
    Ytr, Yte = slice_targets_by_idx(df_all_v, targets, idx_tr, idx_te)
    ytr_h = Ytr["homo"].values
    ytr_l = Ytr["lumo"].values
    ytr_g = Ytr["gap"].values
    yte_g = Yte["gap"].values

    # ---------- per-seed strict feature prep (train-only fit)
    keep_m = fit_binary_feature_filter_on_train(
        X_morgan_base_all, idx_tr, zero_var=True, max_zero_frac=max_zero_frac
    )
    Xtr_m, Xte_m = apply_mask_and_slice(X_morgan_base_all, idx_tr, idx_te, keep_m)

    keep_mac = fit_binary_feature_filter_on_train(
        X_maccs_all, idx_tr, zero_var=True, max_zero_frac=max_zero_frac
    )
    Xtr_mac, Xte_mac = apply_mask_and_slice(X_maccs_all, idx_tr, idx_te, keep_mac)

    keep_idx_desc, kept_desc_names, scaler_desc = fit_desc_filter_and_scaler_on_train(
        X_desc_raw_all, idx_tr, desc_names, var_thresh=var_thresh, corr_thresh=corr_thresh
    )
    Xtr_desc_s, Xte_desc_s = apply_desc_filter_and_scaler(
        X_desc_raw_all, idx_tr, idx_te, keep_idx_desc, scaler_desc
    )

    # TFIDF strict: fit on TRAIN only
    Xtr_self, Xte_self, vec_self, keep_self = fit_selfies_tfidf_on_train(
        df_all_v[smiles_col], idx_tr, idx_te,
        ngram_range=(2, 5), min_df=2, max_df=0.95,
        max_zero_frac=max_zero_frac
    )

    # 9 fusion feature sets (already sliced)
    feature_sets = build_feature_sets_9(
        Xtr_m, Xte_m,
        Xtr_self, Xte_self,
        Xtr_mac, Xte_mac,
        Xtr_desc_s, Xte_desc_s,
    )

    # ---------- model zoo
    models = get_model_zoo(seed=seed, pca_dim=pca_dim)

    # ---------- base combos for HOMO/LUMO (strict OOF)
    homo_model_name, homo_feat_name = parse_combo(homo_combo)
    lumo_model_name, lumo_feat_name = parse_combo(lumo_combo)

    if homo_model_name not in models:
        raise KeyError(f"HOMO model '{homo_model_name}' not in models: {list(models.keys())}")
    if lumo_model_name not in models:
        raise KeyError(f"LUMO model '{lumo_model_name}' not in models: {list(models.keys())}")
    if homo_feat_name not in feature_sets:
        raise KeyError(f"HOMO feature '{homo_feat_name}' not in feature_sets: {list(feature_sets.keys())}")
    if lumo_feat_name not in feature_sets:
        raise KeyError(f"LUMO feature '{lumo_feat_name}' not in feature_sets: {list(feature_sets.keys())}")

    Xtr_h, Xte_h = feature_sets[homo_feat_name]
    Xtr_l, Xte_l = feature_sets[lumo_feat_name]

    homo_oof, homo_te = oof_predictions_one_model(
        Xtr_h, Xte_h, ytr_h, models[homo_model_name], kfold=kfold, seed=seed
    )
    lumo_oof, lumo_te = oof_predictions_one_model(
        Xtr_l, Xte_l, ytr_l, models[lumo_model_name], kfold=kfold, seed=seed
    )

    # stacking features (train=OOF; test=full-train preds)
    Ztr = np.column_stack([homo_oof, lumo_oof, (lumo_oof - homo_oof)])
    Zte = np.column_stack([homo_te,  lumo_te,  (lumo_te  - homo_te )])

    # ---------- evaluate each gap candidate
    rows = []
    pred_details = {}

    for combo in gap_direct_combos:
        gap_model_name, gap_feat_name = parse_combo(combo)

        if gap_model_name not in models:
            raise KeyError(f"GAP model '{gap_model_name}' not in models: {list(models.keys())}")
        if gap_feat_name not in feature_sets:
            raise KeyError(f"GAP feature '{gap_feat_name}' not in feature_sets: {list(feature_sets.keys())}")

        Xtr_g, Xte_g = feature_sets[gap_feat_name]
        est_g = models[gap_model_name]

        # --- direct
        m_direct = _safe_clone(est_g)
        m_direct.fit(Xtr_g, ytr_g)
        gap_direct_te = m_direct.predict(Xte_g)

        # --- stacked (augment with OOF-based Z)
        Xtr_aug = np.hstack([Xtr_g, Ztr])
        Xte_aug = np.hstack([Xte_g, Zte])

        m_stack = _safe_clone(est_g)
        m_stack.fit(Xtr_aug, ytr_g)
        gap_stacked_te = m_stack.predict(Xte_aug)

        md = _metrics(yte_g, gap_direct_te)
        ms = _metrics(yte_g, gap_stacked_te)

        rows.append({
            "seed": seed,
            "gap_combo": combo,
            "gap_model": gap_model_name,
            "gap_feature": gap_feat_name,

            "R2_direct": md["R2"],
            "RMSE_direct": md["RMSE"],
            "MAE_direct": md["MAE"],

            "R2_stacked": ms["R2"],
            "RMSE_stacked": ms["RMSE"],
            "MAE_stacked": ms["MAE"],

            "Delta_stack_vs_direct": ms["R2"] - md["R2"],
        })

        pred_details[combo] = {
            "gap_direct_te": gap_direct_te,
            "gap_stacked_te": gap_stacked_te,
        }

    df_gap_compare = (
        pd.DataFrame(rows)
        .sort_values(["R2_stacked", "R2_direct"], ascending=False)
        .reset_index(drop=True)
    )

    pred_bundle = {
        "idx_tr": idx_tr,
        "idx_te": idx_te,
        "yte_gap": yte_g,
        "homo_oof": homo_oof,
        "lumo_oof": lumo_oof,
        "homo_te": homo_te,
        "lumo_te": lumo_te,
        "Ztr": Ztr,
        "Zte": Zte,
        "gap_preds": pred_details,
    }

    recipe = {
        "seed": seed,
        "test_size": test_size,
        "kfold": kfold,
        "homo_combo": homo_combo,
        "lumo_combo": lumo_combo,
        "gap_direct_combos": list(gap_direct_combos),

        "desc_kept_names": kept_desc_names,
        "morgan_kept_bits": int(np.sum(keep_m)),
        "maccs_kept_bits": int(np.sum(keep_mac)),

        "tfidf": {
            "analyzer": "char",
            "ngram_range": (2, 5),
            "min_df": 2,
            "max_df": 0.95,
            "kept_cols_after_filters": int(np.sum(keep_self)) if keep_self is not None else None,
        },
    }

    return df_gap_compare, pred_bundle, recipe



def cv_gap_model_strict(
    df_all_v,
    X_morgan_base_all,
    X_maccs_all,
    X_desc_raw_all,
    desc_names,
    combo,                # e.g. "XGB + Morgan+Desc"
    kfold=5,
    seed=0,
    max_zero_frac=0.99,
    var_thresh=1e-12,
    corr_thresh=0.95,
    pca_dim=100,
):
    """
    Strict CV:
      every fold re-fits feature filters + scaler on TRAIN ONLY
    """

    model_name, feat_name = combo.split(" + ", 1)

    models = get_model_zoo(seed=seed, pca_dim=pca_dim)
    est = models[model_name]

    y = df_all_v["gap"].values
    N = len(y)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    rows = []

    for fold, (tr, te) in enumerate(kf.split(np.arange(N))):

        # ---- Morgan
        keep_m = fit_binary_feature_filter_on_train(X_morgan_base_all, tr, True, max_zero_frac)
        Xm_tr, Xm_te = apply_mask_and_slice(X_morgan_base_all, tr, te, keep_m)

        # ---- MACCS
        keep_mac = fit_binary_feature_filter_on_train(X_maccs_all, tr, True, max_zero_frac)
        Xmac_tr, Xmac_te = apply_mask_and_slice(X_maccs_all, tr, te, keep_mac)

        # ---- Desc
        keep_desc, kept_names, scaler = fit_desc_filter_and_scaler_on_train(
            X_desc_raw_all, tr, desc_names, var_thresh, corr_thresh
        )
        Xd_tr, Xd_te = apply_desc_filter_and_scaler(X_desc_raw_all, tr, te, keep_desc, scaler)

        # ---- build feature
        if feat_name == "Morgan+Desc":
            Xtr = np.hstack([Xm_tr, Xd_tr])
            Xte = np.hstack([Xm_te, Xd_te])

        elif feat_name == "Morgan+MACCS+Desc":
            Xtr = np.hstack([Xm_tr, Xmac_tr, Xd_tr])
            Xte = np.hstack([Xm_te, Xmac_te, Xd_te])

        else:
            raise ValueError(f"Unsupported feature: {feat_name}")

        ytr = y[tr]
        yte = y[te]

        m = _safe_clone(est)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)

        rows.append({
            "fold": fold,
            "R2": r2_score(yte, p),
            "RMSE": mean_squared_error(yte, p, squared=False),
            "MAE": mean_absolute_error(yte, p),
        })

    df = pd.DataFrame(rows)

    summary = (
        df[["R2","RMSE","MAE"]]
        .agg(["mean","std","min","max"])
        .T
        .reset_index()
        .rename(columns={"index":"metric"})
    )

    return df, summary