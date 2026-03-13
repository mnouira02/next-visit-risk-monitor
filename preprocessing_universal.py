import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UniversalSDTMPreprocessor:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.label_map = {"<PAD>": "Padding", "<UNK>": "Unknown"}
        self.visit_map = {0.0: 0, -1.0: 1}
        self.protocol_schedule = {}
        self.max_len = 256
        self.cache_path = "sdtm_data_cache.pkl"

    def fit(self, df_events):
        unique_tokens = df_events["TOKEN"].dropna().unique()
        for t in unique_tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)

        mappings = df_events[["TOKEN", "HUMAN_LABEL"]].drop_duplicates()
        for _, row in mappings.iterrows():
            self.label_map[row["TOKEN"]] = row["HUMAN_LABEL"]

        visit_candidates = set(df_events["VISITNUM"].dropna().tolist())
        visit_candidates.update(self.protocol_schedule.keys())
        unique_visits = sorted([float(v) for v in visit_candidates if pd.notna(v)])

        for v in unique_visits:
            if v not in self.visit_map:
                self.visit_map[v] = len(self.visit_map)

    def load_trial_design(self, data_dir):
        tv_path = os.path.join(data_dir, "tv.xpt")
        if os.path.exists(tv_path):
            try:
                df_tv = pd.read_sas(tv_path, format="xport")
                df_tv.columns = [c.upper() for c in df_tv.columns]

                for col in ["VISITNUM", "VISITDY"]:
                    if col in df_tv.columns:
                        df_tv[col] = df_tv[col].apply(self._decode)

                if "VISITNUM" in df_tv.columns and "VISITDY" in df_tv.columns:
                    df_tv["VISITNUM"] = pd.to_numeric(df_tv["VISITNUM"], errors="coerce")
                    df_tv["VISITDY"] = pd.to_numeric(df_tv["VISITDY"], errors="coerce")
                    schedule = df_tv[["VISITNUM", "VISITDY"]].drop_duplicates().dropna()
                    self.protocol_schedule = dict(zip(schedule["VISITNUM"], schedule["VISITDY"]))
                    print(f" [PROTO] Loaded {len(self.protocol_schedule)} planned visits from TV domain.")
            except Exception:
                pass

    def auto_detect_columns(self, df, domain_name):
        cols = df.columns
        date_col, val_col, label_col = None, None, None

        prefix_2char = domain_name[:2]

        candidates_date = [
            f"{domain_name}STDTC",
            f"{domain_name}DTC",
            f"{prefix_2char}STDTC",
            f"{prefix_2char}DTC",
        ]

        for c in candidates_date:
            if c in cols:
                date_col = c
                break

        if not date_col:
            dtc = [c for c in cols if c.endswith("DTC")]
            if dtc:
                date_col = dtc[0]

        prefixes = [domain_name, prefix_2char]

        if not val_col:
            for p in prefixes:
                if f"{p}TESTCD" in cols:
                    val_col = f"{p}TESTCD"
                    if f"{p}TEST" in cols:
                        label_col = f"{p}TEST"
                    break

        if not val_col:
            for p in prefixes:
                if f"{p}DECOD" in cols:
                    val_col = f"{p}DECOD"
                    label_col = f"{p}DECOD"
                    break

        if not val_col:
            for p in prefixes:
                if f"{p}TRT" in cols:
                    val_col = f"{p}TRT"
                    label_col = f"{p}TRT"
                    break

        if not val_col:
            for p in prefixes:
                if f"{p}TERM" in cols:
                    val_col = f"{p}TERM"
                    label_col = f"{p}DECOD" if f"{p}DECOD" in cols else f"{p}TERM"
                    break

        if not val_col and "ELEMENT" in cols:
            val_col = "ELEMENT"
            label_col = "ELEMENT"

        if not val_col and "VISIT" in cols:
            val_col = "VISIT"
            label_col = "VISIT"

        return date_col, val_col, label_col

    def load_data(self, data_dir="data", force_reload=False):
        if not force_reload and os.path.exists(self.cache_path):
            print(f" [CACHE] Loading pre-processed data from {self.cache_path}...")
            try:
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    self.protocol_schedule = cache_data["schedule"]
                    return cache_data["events"]
            except Exception as e:
                print(f" [CACHE] Failed to load cache: {e}. Reloading from source.")

        print(f" [SCAN] Scanning {data_dir} for SDTM files...")
        self.load_trial_design(data_dir)

        try:
            df_dm = pd.read_sas(os.path.join(data_dir, "dm.xpt"), format="xport")
            df_dm.columns = [c.upper() for c in df_dm.columns]
        except Exception:
            raise FileNotFoundError("dm.xpt required.")

        for col in ["USUBJID", "RFSTDTC"]:
            if col in df_dm.columns:
                df_dm[col] = df_dm[col].apply(self._decode)

        df_dm["RFSTDTC"] = pd.to_datetime(df_dm["RFSTDTC"], errors="coerce")

        all_events = []
        files = [f for f in os.listdir(data_dir) if f.endswith(".xpt")]

        for f in files:
            domain = f.split(".")[0].upper()

            if domain in ["DM", "TV", "TS", "TE", "TA", "TI", "SUPP", "RELREC", "SUPPAE", "SUPPDM", "SUPPDS"]:
                continue

            try:
                df = pd.read_sas(os.path.join(data_dir, f), format="xport")
                df.columns = [c.upper() for c in df.columns]
            except Exception:
                print(f" [WARN] Could not read {f}")
                continue

            if "USUBJID" not in df.columns:
                continue

            date_col, val_col, label_col = self.auto_detect_columns(df, domain)
            if not date_col or not val_col:
                continue

            cols_to_decode = ["USUBJID", date_col, val_col]
            if label_col and label_col not in cols_to_decode:
                cols_to_decode.append(label_col)
            if "VISITNUM" in df.columns:
                cols_to_decode.append("VISITNUM")

            prefix = domain[:2] if len(domain) > 2 else domain

            display_candidates = [
                f"{domain}TERM",
                f"{prefix}TERM",
                f"{domain}TEST",
                f"{prefix}TEST",
                f"{domain}TRT",
                f"{prefix}TRT",
                "VISIT",
                "ELEMENT",
            ]

            res_col_candidates = [
                f"{domain}STRESN",
                f"{prefix}STRESN",
            ]

            for c in set(cols_to_decode + display_candidates + res_col_candidates):
                if c in df.columns:
                    df[c] = df[c].apply(self._decode)

            raw_val_original = self._clean_series(df[val_col]) if val_col in df.columns else pd.Series("", index=df.index)
            dtc_series = self._clean_series(df[date_col]) if date_col in df.columns else pd.Series("", index=df.index)

            df = df.copy()
            df["DTC"] = dtc_series
            df["RAW_VAL"] = raw_val_original.fillna("").astype(str).str.strip()
            df["RAW_VAL"] = df["RAW_VAL"].replace("", "UNK")
            df["RAW_VAL_STD"] = df["RAW_VAL"].str.upper().str.strip()

            df["TOKEN"] = prefix + "_" + df["RAW_VAL_STD"]

            df["HUMAN_LABEL"] = self._build_human_label(
                df=df,
                domain=domain,
                prefix=prefix,
                label_col=label_col,
                fallback_raw=df["RAW_VAL"]
            )

            df["VAL"] = 1.0
            for res_col in res_col_candidates:
                if res_col in df.columns:
                    df["VAL"] = pd.to_numeric(df[res_col], errors="coerce").fillna(0)
                    break

            if "VISITNUM" not in df.columns:
                df["VISITNUM"] = -1.0
            df["VISITNUM"] = pd.to_numeric(df["VISITNUM"], errors="coerce").fillna(-1.0)

            all_events.append(df[["USUBJID", "DTC", "VISITNUM", "TOKEN", "HUMAN_LABEL", "VAL"]])
            print(f" [LOAD] {domain}: {len(df)} events")

        if not all_events:
            raise ValueError("No valid event data found.")

        events = pd.concat(all_events, ignore_index=True)

        events["DTC"] = events["DTC"].fillna("").astype(str).str.strip()
        events = events[events["DTC"].str.len() >= 10]
        events["DTC"] = pd.to_datetime(events["DTC"], errors="coerce")
        events = events.dropna(subset=["DTC"])

        events = events.merge(df_dm[["USUBJID", "RFSTDTC"]], on="USUBJID", how="inner")
        events["DY"] = (events["DTC"] - events["RFSTDTC"]).dt.days
        events["PLANNED_DY"] = events["VISITNUM"].map(self.protocol_schedule).fillna(0)

        events = events[(events["DY"] > -100) & (events["DY"] < 3000)]
        events = events.sort_values(["USUBJID", "DY"]).reset_index(drop=True)

        print(f" [CACHE] Saving processed data to {self.cache_path}...")
        with open(self.cache_path, "wb") as f:
            pickle.dump({"events": events, "schedule": self.protocol_schedule}, f)

        return events

    def _decode(self, x):
        return x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x

    def _clean_series(self, series):
        return series.apply(self._decode).fillna("").astype(str).str.strip()

    def _build_human_label(self, df, domain, prefix, label_col, fallback_raw):
        result = pd.Series("", index=df.index, dtype="object")

        preferred_display_cols = [
            f"{domain}TERM",
            f"{prefix}TERM",
            f"{domain}TEST",
            f"{prefix}TEST",
            f"{domain}TRT",
            f"{prefix}TRT",
            "VISIT",
            "ELEMENT",
        ]

        for col in preferred_display_cols:
            if col in df.columns:
                series = self._clean_series(df[col])
                result = result.mask(result.eq(""), series)

        coded_label = pd.Series("", index=df.index, dtype="object")
        if label_col and label_col in df.columns:
            coded_label = self._clean_series(df[label_col])

        use_coded = (~coded_label.str.upper().eq("UNCODED")) & coded_label.ne("")
        result = result.mask(use_coded, coded_label)

        fallback_raw = fallback_raw.fillna("").astype(str).str.strip()
        result = result.mask(result.eq(""), fallback_raw)
        result = result.replace("", "Unknown")

        return result

    def _get_protocol_visits(self, events_df):
        if self.protocol_schedule:
            visits = sorted(
                [float(v) for v in self.protocol_schedule.keys() if pd.notna(v) and float(v) >= 0]
            )
        else:
            visits = sorted(
                [float(v) for v in events_df["VISITNUM"].dropna().unique() if float(v) >= 0]
            )
        return visits

    def _classify_visit_status(self, actual_day, planned_day, window_days, medium_multiplier):
        if pd.isna(actual_day):
            return 2, "High", np.nan, "No visit observed", False, False

        diff_days = float(actual_day - planned_day)
        abs_diff = abs(diff_days)
        within_window = abs_diff <= window_days

        if abs_diff <= window_days:
            return 0, "Low", diff_days, "Within window", True, within_window
        if abs_diff <= (window_days * medium_multiplier):
            return 1, "Medium", diff_days, "Slightly out of window", True, within_window
        return 2, "High", diff_days, "Severely out of window", True, within_window

    def build_risk_dataset(
        self,
        events_df,
        default_window_days=7,
        medium_multiplier=2.0,
        min_history_events=5
    ):
        protocol_visits = self._get_protocol_visits(events_df)
        if len(protocol_visits) < 2:
            raise ValueError("Need at least two scheduled visits to build next-visit risk examples.")

        target_visits = protocol_visits[1:]

        X_cat, X_num, X_time, X_visit, X_planned = [], [], [], [], []
        X_target_visit, X_target_planned, y = [], [], []
        meta_rows = []

        example_idx = 0

        for sub_id, subj in events_df.groupby("USUBJID"):
            subj = subj.sort_values(["DY", "VISITNUM"]).reset_index(drop=True)

            observed_visit_days = (
                subj[subj["VISITNUM"] >= 0]
                .groupby("VISITNUM")["DY"]
                .median()
                .to_dict()
            )

            parts = str(sub_id).split("-")
            site_id = parts[1] if len(parts) >= 2 else "UNK"

            for target_visit in target_visits:
                planned_day = self.protocol_schedule.get(target_visit, np.nan)

                if pd.isna(planned_day):
                    plan_series = (
                        subj.loc[subj["VISITNUM"] == target_visit, "PLANNED_DY"]
                        .replace(0, np.nan)
                        .dropna()
                    )
                    if len(plan_series) == 0:
                        continue
                    planned_day = float(plan_series.iloc[0])

                history = subj[
                    (
                        ((subj["VISITNUM"] >= 0) & (subj["VISITNUM"] < target_visit))
                        | ((subj["VISITNUM"] < 0) & (subj["DY"] < planned_day))
                    )
                ].copy()

                history = history[history["DY"] < planned_day].copy()

                if len(history) < min_history_events:
                    continue

                actual_day = observed_visit_days.get(target_visit, np.nan)
                risk_class, risk_name, diff_days, outcome_desc, has_actual_visit, within_window = self._classify_visit_status(
                    actual_day=actual_day,
                    planned_day=planned_day,
                    window_days=default_window_days,
                    medium_multiplier=medium_multiplier
                )

                cats = [self.vocab.get(t, self.vocab["<UNK>"]) for t in history["TOKEN"].values]
                nums = pd.to_numeric(history["VAL"], errors="coerce").fillna(0).values.tolist()
                times = pd.to_numeric(history["DY"], errors="coerce").fillna(0).values.tolist()
                visits = [self.visit_map.get(v, 0) for v in history["VISITNUM"].values]
                planned = pd.to_numeric(history["PLANNED_DY"], errors="coerce").fillna(0).values.tolist()

                seq_len = min(len(cats), self.max_len)
                pad_len = self.max_len - seq_len

                X_cat.append(cats[:seq_len] + [0] * pad_len)
                X_num.append(nums[:seq_len] + [0] * pad_len)
                X_time.append(times[:seq_len] + [0] * pad_len)
                X_visit.append(visits[:seq_len] + [0] * pad_len)
                X_planned.append(planned[:seq_len] + [0] * pad_len)

                X_target_visit.append(self.visit_map.get(target_visit, 0))
                X_target_planned.append(float(planned_day))
                y.append(int(risk_class))

                history_last_day = float(history["DY"].max()) if len(history) else 0.0
                visit_gap = float(planned_day - history_last_day)

                meta_rows.append({
                    "Example Index": example_idx,
                    "USUBJID": sub_id,
                    "Site": site_id,
                    "Target Visit": float(target_visit),
                    "Target Visit Label": f"Visit {float(target_visit):.1f}",
                    "Target Planned Day": float(planned_day),
                    "Target Actual Day": np.nan if pd.isna(actual_day) else float(actual_day),
                    "Window Days": float(default_window_days),
                    "Diff Days": np.nan if pd.isna(diff_days) else float(diff_days),
                    "Actual Risk": risk_name,
                    "Outcome Description": outcome_desc,
                    "Has Actual Visit": bool(has_actual_visit),
                    "Within Window": bool(within_window),
                    "History Events": int(len(history)),
                    "History Last Day": history_last_day,
                    "Days Until Target": visit_gap
                })

                example_idx += 1

        if len(y) == 0:
            raise ValueError("No next-visit risk examples could be created from the data.")

        tensors = (
            torch.tensor(X_cat, dtype=torch.long),
            torch.tensor(X_num, dtype=torch.float),
            torch.tensor(X_time, dtype=torch.float),
            torch.tensor(X_visit, dtype=torch.long),
            torch.tensor(X_planned, dtype=torch.float),
            torch.tensor(X_target_visit, dtype=torch.long),
            torch.tensor(X_target_planned, dtype=torch.float),
            torch.tensor(y, dtype=torch.long),
        )

        meta_df = pd.DataFrame(meta_rows)
        return tensors, meta_df


class RiskPredictionDataset(Dataset):
    def __init__(
        self,
        x_cat,
        x_num,
        x_time,
        x_visit,
        x_planned,
        x_target_visit,
        x_target_planned,
        y
    ):
        self.x_cat = x_cat
        self.x_num = x_num
        self.x_time = x_time
        self.x_visit = x_visit
        self.x_planned = x_planned
        self.x_target_visit = x_target_visit
        self.x_target_planned = x_target_planned
        self.y = y

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        return (
            self.x_cat[idx],
            self.x_num[idx],
            self.x_time[idx],
            self.x_visit[idx],
            self.x_planned[idx],
            self.x_target_visit[idx],
            self.x_target_planned[idx],
            self.y[idx],
        )
