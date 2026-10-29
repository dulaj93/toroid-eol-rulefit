"""
Cleaning and feature engineering for the toroidal transformer EOL test dataset.

Used by the training notebook and the Streamlit app so that both apply exactly
the same transformations.

    raw  = load_raw(path)              # read CSV, keep raw strings
    df   = clean(raw)                  # fix duplicates, labels, sentinels, timestamps
    first = first_attempts(df)         # one row per unit (modelling grain)
    fb   = FeatureBuilder().fit(train) # learns product / grade reference values
    X    = fb.transform(any_frame)     # model-ready features
"""
import json
import numpy as np
import pandas as pd

# Columns measured during / after the EOL test (or derived from its outcome).
# They must never be used as predictors -> target leakage.
LEAKAGE_COLS = [
    "test_station", "test_attempt", "test_ambient_C", "test_voltage_V", "ir_Mohm",
    "hipot_kV", "hipot_leak_mA", "surge_EAR_pct", "pri_dcr_ohm", "sec_dcr_ohm",
    "noload_current_mA", "noload_loss_W", "noload_current_limit_mA", "noload_loss_limit_W",
    "sec_ocv_V", "ocv_spec_V", "ocv_error_pct", "result", "fail_code", "disposition",
    "operator_remarks",
]
ID_COLS = ["record_id", "serial_no", "work_order", "anneal_date", "wind_end_ts", "test_ts"]

CAT_FEATURES = [
    "product_code", "primary_config", "secondary_config", "application_segment",
    "steel_grade", "steel_supplier", "steel_coil_lot", "core_insulation", "core_deburred",
    "anneal_furnace", "winding_machine", "operator_id", "shift", "wire_supplier",
    "interwinding_tape", "impregnation",
]
NUM_FEATURES = [
    "rated_power_VA", "frequency_Hz", "coil_cert_loss_W_kg", "core_OD_mm", "core_ID_mm",
    "core_height_mm", "core_mass_g", "core_coat_thickness_um", "anneal_peak_temp_C",
    "anneal_soak_min", "anneal_dewpoint_C", "anneal_ref_ring_loss_W_kg",
    "operator_experience_months", "enamel_thermal_class", "pri_wire_dia_mm", "sec_wire_dia_mm",
    "sec_parallel_strands", "pri_turns_target", "sec_turns_target", "pri_tension_N",
    "sec_tension_N", "pri_wind_speed_rpm", "sec_wind_speed_rpm", "interwinding_layers",
    "tape_overlap_pct", "winding_area_RH_pct", "winding_area_temp_C", "inproc_pri_dcr_ohm",
]
ENGINEERED = [
    "wip_hold_h", "anneal_to_wind_days", "pri_turn_dev", "sec_turn_dev", "sec_turn_step_pct",
    "pri_tension_rel", "sec_tension_rel", "pri_speed_rel", "sec_speed_rel", "core_mass_rel",
    "cert_loss_rel_grade", "inproc_dcr_rel", "core_section_mm2",
]
FEATURES = CAT_FEATURES + NUM_FEATURES + ENGINEERED


def load_raw(path):
    return pd.read_csv(path, low_memory=False, dtype={"ir_Mohm": str})


def _parse_test_ts(s):
    """TS-01 exports ISO 'YYYY-MM-DD HH:MM:SS', TS-02 exports 'DD/MM/YYYY HH:MM'."""
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    regional = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(regional)


def clean(raw):
    df = raw.drop_duplicates().copy()

    # categorical label harmonisation
    df["shift"] = df["shift"].astype(str).str.strip().str[-1].str.upper()
    deb = df["core_deburred"].fillna("").astype(str).str.strip().str.upper().str[:1]
    df["core_deburred"] = deb.replace({"": "UNKNOWN"})
    df["product_code"] = df["product_code"].str.upper().str.strip()
    df["wire_supplier"] = df["wire_supplier"].str.strip().str.replace(" ", "-", regex=False)

    # sentinels / impossible values -> missing
    df.loc[df["anneal_peak_temp_C"] <= 0, "anneal_peak_temp_C"] = np.nan
    rh = df["winding_area_RH_pct"]
    df.loc[(rh <= 0) | (rh > 100), "winding_area_RH_pct"] = np.nan
    # core mass occasionally typed in mg: compare with the product's median mass
    med = df.groupby("product_code")["core_mass_g"].transform("median")
    mg = df["core_mass_g"] > 100 * med
    df.loc[mg, "core_mass_g"] = df.loc[mg, "core_mass_g"] / 1000

    # meter over-range text -> numeric ceiling (test-side column, kept for analysis only)
    ir = df["ir_Mohm"].astype(str).str.strip()
    df["ir_overrange"] = ir.eq("OVR")
    df["ir_Mohm"] = pd.to_numeric(ir.replace({"OVR": "10000", "nan": np.nan, "": np.nan}), errors="coerce")

    # timestamps
    df["wind_end_ts"] = pd.to_datetime(df["wind_end_ts"], format="%Y-%m-%d %H:%M:%S")
    df["test_ts"] = _parse_test_ts(df["test_ts"])
    df["anneal_date"] = pd.to_datetime(df["anneal_date"], format="%Y-%m-%d")

    df["fail_code"] = df["fail_code"].fillna("PASS")
    df["y_fail"] = (df["result"] == "FAIL").astype(int)
    return df.reset_index(drop=True)


def first_attempts(df):
    """Modelling grain: the first EOL test of every serial number."""
    return (df[df["test_attempt"] == 1]
            .sort_values("wind_end_ts")
            .drop_duplicates("serial_no")
            .reset_index(drop=True))


class FeatureBuilder:
    """Recipe-relative features. Reference values are learned on the training
    split only (fit) and re-used for validation, test and the app (transform)."""

    def fit(self, df):
        g = df.groupby("product_code")
        self.ref_ = {
            "pri_tension_N": g["pri_tension_N"].median().to_dict(),
            "sec_tension_N": g["sec_tension_N"].median().to_dict(),
            "pri_wind_speed_rpm": g["pri_wind_speed_rpm"].median().to_dict(),
            "sec_wind_speed_rpm": g["sec_wind_speed_rpm"].median().to_dict(),
            "core_mass_g": g["core_mass_g"].median().to_dict(),
            "inproc_pri_dcr_ohm": g["inproc_pri_dcr_ohm"].median().to_dict(),
        }
        self.grade_ref_ = df.groupby("steel_grade")["coil_cert_loss_W_kg"].median().to_dict()
        # fallback by rated power for product codes not seen in training
        self.va_ref_ = {c: df.groupby("rated_power_VA")[c].median().to_dict() for c in self.ref_}
        return self

    def _rel(self, df, col):
        ref = df["product_code"].map(self.ref_[col])
        ref = ref.fillna(df["rated_power_VA"].map(self.va_ref_[col]))
        return df[col] / ref

    def transform(self, df):
        X = pd.DataFrame(index=df.index)
        for c in CAT_FEATURES:
            X[c] = df[c].fillna("NA").astype(str)
        for c in NUM_FEATURES:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        X["wip_hold_h"] = (df["test_ts"] - df["wind_end_ts"]).dt.total_seconds() / 3600
        X["anneal_to_wind_days"] = (df["wind_end_ts"].dt.normalize() - df["anneal_date"]).dt.days
        X["pri_turn_dev"] = df["pri_turns_logged"] - df["pri_turns_target"]
        X["sec_turn_dev"] = df["sec_turns_logged"] - df["sec_turns_target"]
        X["sec_turn_step_pct"] = 100.0 / df["sec_turns_target"]   # ratio change from one turn
        X["pri_tension_rel"] = self._rel(df, "pri_tension_N")
        X["sec_tension_rel"] = self._rel(df, "sec_tension_N")
        X["pri_speed_rel"] = self._rel(df, "pri_wind_speed_rpm")
        X["sec_speed_rel"] = self._rel(df, "sec_wind_speed_rpm")
        X["core_mass_rel"] = self._rel(df, "core_mass_g")
        X["inproc_dcr_rel"] = self._rel(df, "inproc_pri_dcr_ohm")
        X["cert_loss_rel_grade"] = df["coil_cert_loss_W_kg"] / df["steel_grade"].map(self.grade_ref_)
        X["core_section_mm2"] = (df["core_OD_mm"] - df["core_ID_mm"]) / 2 * df["core_height_mm"]
        return X[FEATURES]

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"ref": self.ref_, "grade_ref": self.grade_ref_,
                       "va_ref": {k: {str(a): b for a, b in v.items()} for k, v in self.va_ref_.items()}},
                      f, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        fb = cls()
        fb.ref_, fb.grade_ref_ = d["ref"], d["grade_ref"]
        fb.va_ref_ = {k: {int(a): b for a, b in v.items()} for k, v in d["va_ref"].items()}
        return fb
