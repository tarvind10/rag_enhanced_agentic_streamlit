import os
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Optional: try to import Hugging Face inference
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

warnings.filterwarnings("ignore")


# -------------------------
# Utility functions
# -------------------------

def load_tabular(path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    path = path_or_df
    if isinstance(path, str):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            return pd.read_json(path)
        else:
            return pd.read_csv(path)
    if hasattr(path, "name"):
        if path.name.endswith(".csv"):
            return pd.read_csv(path)
        elif path.name.endswith(".json"):
            return pd.read_json(path)
    return pd.read_csv(path)


def ensure_column(df: pd.DataFrame, col: str, default: Any = None) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


# -------------------------
# Sample data generator
# -------------------------

def generate_sample_data(
    n_transactions: int = 500,
    n_entities: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    np.random.seed(seed)

    entity_ids = [f"E{str(i).zfill(4)}" for i in range(1, n_entities + 1)]

    countries = ["US", "GB", "IN", "SG", "AE", "UNKNOWN"]
    risk_ratings = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    kyc_rows = []
    for eid in entity_ids:
        kyc_rows.append({
            "entity_id": eid,
            "name": f"Entity {eid}",
            "country": rng.choice(countries),
            "risk_rating": rng.choice(risk_ratings),
            "onboard_date": (datetime.now() - timedelta(days=rng.randint(30, 720))).strftime("%Y-%m-%d"),
        })
    kyc_df = pd.DataFrame(kyc_rows)

    channels = ["ONLINE", "BRANCH", "ATM", "WIRE", "MOBILE"]
    base_date = datetime.now() - timedelta(days=180)
    tx_rows = []
    for i in range(n_transactions):
        eid = rng.choice(entity_ids)
        days_offset = rng.randint(0, 180)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        ts = base_date + timedelta(days=days_offset, hours=hour, minutes=minute)

        base_amount = (
            rng.uniform(50, 500) if rng.random() > 0.15
            else rng.uniform(2000, 15000)
        )

        tx_rows.append({
            "transaction_id": f"TX{str(i+1).zfill(6)}",
            "entity_id": eid,
            "amount": round(base_amount, 2),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": rng.choice(channels),
            "currency": "USD",
            "counterparty_country": rng.choice(countries),
        })
    transactions_df = pd.DataFrame(tx_rows)

    entity_agg = (
        transactions_df.groupby("entity_id")
        .agg(
            avg_amount=("amount", "mean"),
            tx_count=("transaction_id", "count"),
        )
        .reset_index()
    )
    entity_agg["risk_label"] = "LOW"
    high_amount_thresh = entity_agg["avg_amount"].quantile(0.8)
    high_tx_thresh = entity_agg["tx_count"].quantile(0.8)

    entity_agg.loc[
        (entity_agg["avg_amount"] > high_amount_thresh) &
        (entity_agg["tx_count"] > high_tx_thresh),
        "risk_label"
    ] = "HIGH"

    entity_agg.loc[
        ~((entity_agg["risk_label"] == "HIGH")),
        "risk_label"
    ] = entity_agg.loc[
        ~((entity_agg["risk_label"] == "HIGH")),
        "risk_label"
    ].apply(lambda x: rng.choice(["LOW", "MEDIUM"]))

    cases_df = entity_agg[["entity_id", "risk_label"]]

    return transactions_df, kyc_df, cases_df


# -------------------------
# DataPrepAgent
# -------------------------

class DataPrepAgent:
    def __init__(self):
        self.transactions: Optional[pd.DataFrame] = None
        self.kyc: Optional[pd.DataFrame] = None
        self.cases: Optional[pd.DataFrame] = None
        self.merged: Optional[pd.DataFrame] = None

    def load_data(
        self,
        transactions: Optional[Union[str, pd.DataFrame, Any]] = None,
        kyc: Optional[Union[str, pd.DataFrame, Any]] = None,
        cases: Optional[Union[str, pd.DataFrame, Any]] = None,
        entity_id_col: str = "entity_id",
        use_sample_if_missing: bool = True,
    ) -> pd.DataFrame:
        if use_sample_if_missing and transactions is None and kyc is None and cases is None:
            tx, kyc_df, cases_df = generate_sample_data(
                n_transactions=500,
                n_entities=50,
                seed=42,
            )
            self.transactions = tx
            self.kyc = kyc_df
            self.cases = cases_df
        else:
            if transactions is not None:
                self.transactions = load_tabular(transactions).copy()
            if kyc is not None:
                self.kyc = load_tabular(kyc).copy()
            if cases is not None:
                self.cases = load_tabular(cases).copy()

        self.transactions = self._normalize_transactions()
        if self.kyc is not None:
            self.kyc = self._normalize_kyc()
        if self.cases is not None:
            self.cases = self._normalize_cases()

        df = self.transactions.copy()
        if self.kyc is not None:
            df = df.merge(
                self.kyc,
                on=entity_id_col,
                how="left",
                suffixes=("", "_kyc"),
            )
        if self.cases is not None:
            cases_agg = self.cases.groupby(entity_id_col).first().reset_index()
            df = df.merge(
                cases_agg,
                on=entity_id_col,
                how="left",
                suffixes=("", "_case"),
            )

        self.merged = df
        return df

    def _normalize_transactions(self) -> pd.DataFrame:
        df = self.transactions
        df = ensure_column(df, "amount", 0.0)
        df = ensure_column(df, "timestamp", None)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def _normalize_kyc(self) -> pd.DataFrame:
        df = self.kyc
        df = ensure_column(df, "risk_rating", "UNKNOWN")
        df = ensure_column(df, "country", "UNKNOWN")
        return df

    def _normalize_cases(self) -> pd.DataFrame:
        df = self.cases
        label_candidates = ["risk_label", "is_fraud", "entity_type", "label"]
        has_label = any(c in df.columns for c in label_candidates)
        if not has_label:
            return df
        return df


# -------------------------
# FeatureEngineeringAgent
# -------------------------

class FeatureEngineeringAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        features["amount"] = df["amount"].fillna(0)
        features["log_amount"] = np.log1p(features["amount"])

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            features["hour"] = ts.dt.hour.fillna(12).astype(int)
            features["dayofweek"] = ts.dt.dayofweek.fillna(3).astype(int)
            features["is_weekend"] = (features["dayofweek"] >= 5).astype(int)
            features["is_night"] = ((features["hour"] < 6) | (features["hour"] >= 22)).astype(int)
        else:
            features["hour"] = 12
            features["dayofweek"] = 3
            features["is_weekend"] = 0
            features["is_night"] = 0

        if "channel" in df.columns:
            channel = df["channel"].fillna("UNKNOWN")
            for ch in channel.unique():
                features[f"channel_{ch}"] = (channel == ch).astype(int)
        else:
            features["channel_UNKNOWN"] = 1

        if "country" in df.columns:
            country = df["country"].fillna("UNKNOWN")
            for c in country.unique():
                features[f"country_{c}"] = (country == c).astype(int)
        else:
            features["country_UNKNOWN"] = 1

        if "entity_id" in df.columns:
            entity_id = df["entity_id"]
            entity_counts = entity_id.map(entity_id.value_counts())
            features["entity_tx_count"] = entity_counts

            entity_amount_mean = entity_id.map(
                df.groupby("entity_id")["amount"].transform("mean")
            )
            entity_amount_std = entity_id.map(
                df.groupby("entity_id")["amount"].transform("std")
            ).fillna(0)
            features["entity_amount_mean"] = entity_amount_mean
            features["entity_amount_std"] = entity_amount_std
        else:
            features["entity_tx_count"] = 1
            features["entity_amount_mean"] = features["amount"]
            features["entity_amount_std"] = 0

        features["amount_to_mean_ratio"] = features["amount"] / (
            features["entity_amount_mean"] + 1e-6
        )

        features = features.fillna(0)
        self.feature_cols = list(features.columns)
        return features

    def get_feature_matrix(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_cols].values
        return X


# -------------------------
# SupervisedMLAgent
# -------------------------

class SupervisedMLAgent:
    def __init__(self):
        self.model = None
        self.label_col: Optional[str] = None
        self.is_classification = True

    def detect_label(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ["risk_label", "is_fraud", "entity_type", "label"]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SupervisedMLAgent: model not trained yet.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SupervisedMLAgent: model not trained yet.")
        return self.model.predict(X)


# -------------------------
# UnsupervisedMLAgent
# -------------------------

class UnsupervisedMLAgent:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self.fitted = True

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("UnsupervisedMLAgent: model not fitted yet.")
        return self.model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("UnsupervisedMLAgent: model not fitted yet.")
        return self.model.predict(X)


# -------------------------
# LLMAnalysisAgent
# -------------------------

class LLMAnalysisAgent:
    def __init__(self, hf_token: Optional[str] = None, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.client = None
        if HF_AVAILABLE and self.hf_token:
            try:
                self.client = InferenceClient(token=self.hf_token)
            except Exception:
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def generate_analysis(self, summary_stats: Dict[str, Any], top_anomalies: List[Dict[str, Any]]) -> str:
        if not self.is_available():
            return self._rule_based_summary(summary_stats, top_anomalies)

        prompt = self._build_prompt(summary_stats, top_anomalies)
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=256,
                temperature=0.2,
            )
            return response.strip()
        except Exception:
            return self._rule_based_summary(summary_stats, top_anomalies)

    def _build_prompt(self, summary_stats: Dict[str, Any], top_anomalies: List[Dict[str, Any]]) -> str:
        stats_text = "\n".join(f"- {k}: {v}" for k, v in summary_stats.items())
        anomalies_text = "\n".join(
            f"- Entity {a.get('entity_id', 'N/A')}: score={a.get('anomaly_score', 0):.3f}, "
            f"amount={a.get('amount', 0)}, features={a.get('top_features', [])}"
            for a in top_anomalies[:5]
        )
        prompt = (
            "You are an expert AML/shell-entity analyst. Based on the following summary statistics "
            "and top anomalous entities, provide a short analysis (3–6 sentences) describing:\n"
            "- Overall risk patterns\n"
            "- Likely typologies (e.g., structuring, layering, mule accounts)\n"
            "- Suggested next investigative steps\n\n"
            f"Summary statistics:\n{stats_text}\n\n"
            f"Top anomalies:\n{anomalies_text}\n\n"
            "Answer concisely in plain English."
        )
        return prompt

    def _rule_based_summary(self, summary_stats: Dict[str, Any], top_anomalies: List[Dict[str, Any]]) -> str:
        lines = [
            "Rule-based summary (LLM not available):",
            f"- Total transactions analyzed: {summary_stats.get('total_transactions', 0)}",
            f"- Entities analyzed: {summary_stats.get('total_entities', 0)}",
            f"- Anomalies detected: {summary_stats.get('anomalies_detected', 0)}",
            "Top anomalous entities (by score):",
        ]
        for a in top_anomalies[:5]:
            lines.append(
                f"- Entity {a.get('entity_id', 'N/A')}: score={a.get('anomaly_score', 0):.3f}, "
                f"amount={a.get('amount', 0)}"
            )
        return "\n".join(lines)


# -------------------------
# RuleBasedEngineAgent
# -------------------------

class RuleBasedEngineAgent:
    def __init__(self):
        self.rules = []

    def evaluate(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        rules_output = pd.DataFrame(index=df.index)
        rules_output["rule_high_amount"] = 0
        rules_output["rule_velocity_spike"] = 0
        rules_output["rule_night_activity"] = 0
        rules_output["rule_weekend_activity"] = 0
        rules_output["rule_high_risk_country"] = 0
        rules_output["rule_kyc_mismatch"] = 0

        if "amount" in features.columns:
            high_thresh = features["amount"].quantile(0.95)
            rules_output["rule_high_amount"] = (features["amount"] > high_thresh).astype(int)

        if "is_night" in features.columns:
            rules_output["rule_night_activity"] = features["is_night"]
        if "is_weekend" in features.columns:
            rules_output["rule_weekend_activity"] = features["is_weekend"]

        high_risk_countries = ["UNKNOWN"]
        if "country" in df.columns:
            country = df["country"].fillna("UNKNOWN")
            rules_output["rule_high_risk_country"] = country.isin(high_risk_countries).astype(int)

        if "risk_rating" in df.columns:
            risk = df["risk_rating"].fillna("UNKNOWN")
            rules_output["rule_kyc_mismatch"] = (risk.isin(["HIGH", "UNKNOWN"])).astype(int)

        if "entity_tx_count" in features.columns:
            thresh = features["entity_tx_count"].quantile(0.9)
            rules_output["rule_velocity_spike"] = (features["entity_tx_count"] > thresh).astype(int)

        rules_output["rule_score"] = rules_output.sum(axis=1)
        return rules_output


# -------------------------
# MultiAgentShellDetector (Orchestrator)
# -------------------------

class MultiAgentShellDetector:
    """
    Orchestrates all agents and computes a weighted score per entity.
    Also adds a column showing the formula used for weighted score calculation.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.data_agent = DataPrepAgent()
        self.feat_agent = FeatureEngineeringAgent()
        self.supervised_agent = SupervisedMLAgent()
        self.unsupervised_agent = UnsupervisedMLAgent(contamination=0.05)
        self.llm_agent = LLMAnalysisAgent(hf_token=hf_token)
        self.rule_agent = RuleBasedEngineAgent()

        self.mode: str = "unknown"

        # Weights for final score (can be tuned)
        self.weight_ml = 0.5
        self.weight_rule = 0.35
        self.weight_llm = 0.15

    def _normalize_series(self, s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    def run(
        self,
        transactions: Optional[Union[str, pd.DataFrame, Any]] = None,
        kyc: Optional[Union[str, pd.DataFrame, Any]] = None,
        cases: Optional[Union[str, pd.DataFrame, Any]] = None,
        entity_id_col: str = "entity_id",
        use_sample_if_missing: bool = True,
    ) -> Dict[str, Any]:
        df = self.data_agent.load_data(
            transactions=transactions,
            kyc=kyc,
            cases=cases,
            entity_id_col=entity_id_col,
            use_sample_if_missing=use_sample_if_missing,
        )

        features = self.feat_agent.build_features(df)
        X = self.feat_agent.get_feature_matrix(features)

        label_col = self.supervised_agent.detect_label(df)
        has_labels = label_col is not None

        rule_output = self.rule_agent.evaluate(df, features)

        result = {
            "mode": "",
            "entities": [],
            "summary": {},
            "llm_analysis": "",
        }

        # ML scores
        if has_labels:
            y = df[label_col].fillna(0)
            if y.dtype == "object":
                y = y.astype("category").cat.codes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            self.supervised_agent.train(X_train, y_train)
            y_pred_proba = self.supervised_agent.predict_proba(X)
            if y_pred_proba.ndim == 1:
                risk_scores = y_pred_proba
            else:
                risk_scores = y_pred_proba.max(axis=1)

            result["mode"] = "supervised"
            result["risk_scores"] = risk_scores.tolist()
            result["label_column"] = label_col
            ml_score_series = pd.Series(risk_scores, index=df.index)

        else:
            self.unsupervised_agent.fit(X)
            anomaly_scores = self.unsupervised_agent.score_samples(X)
            anomaly_pred = self.unsupervised_agent.predict(X)

            result["mode"] = "unsupervised"
            result["anomaly_scores"] = anomaly_scores.tolist()
            result["anomaly_flags"] = (anomaly_pred == -1).astype(int).tolist()

            ml_score_series = -pd.Series(anomaly_scores, index=df.index)

        # Rule score
        rule_score_series = rule_output["rule_score"].astype(float)

        # LLM-based risk boost (simple heuristic)
        combined_signal = (
            self._normalize_series(ml_score_series) * 0.7 +
            self._normalize_series(rule_score_series) * 0.3
        )
        llm_boost = combined_signal * self.weight_llm

        # Normalize components to 0–1
        ml_norm = self._normalize_series(ml_score_series)
        rule_norm = self._normalize_series(rule_score_series)

        # Weighted score
        weighted_score = (
            ml_norm * self.weight_ml +
            rule_norm * self.weight_rule +
            llm_boost
        )
        weighted_score = weighted_score.clip(0, 1)

        # Build formula string per row
        formula_rows = []
        for i in range(len(df)):
            ml_val = float(ml_norm.iloc[i])
            rule_val = float(rule_norm.iloc[i])
            llm_val = float(llm_boost.iloc[i])
            formula = (
                f"{self.weight_ml:.2f}*ml_norm({ml_val:.3f}) + "
                f"{self.weight_rule:.2f}*rule_norm({rule_val:.3f}) + "
                f"llm_boost({llm_val:.3f})"
            )
            formula_rows.append(formula)

        # Build output dataframe
        out_df = df.copy()
        out_df["rule_score"] = rule_output["rule_score"]
        if result["mode"] == "supervised":
            out_df["risk_score"] = result["risk_scores"]
        else:
            out_df["anomaly_score"] = result["anomaly_scores"]
            out_df["is_anomaly"] = result["anomaly_flags"]

        out_df["weighted_score"] = weighted_score.values
        out_df["weighted_score_formula"] = formula_rows

        entities = out_df.to_dict(orient="records")
        result["entities"] = entities

        # Summary stats
        summary = {
            "total_transactions": len(df),
            "total_entities": df[entity_id_col].nunique() if entity_id_col in df.columns else len(df),
        }
        if result["mode"] == "supervised":
            summary["avg_risk_score"] = float(np.mean(result["risk_scores"]))
        else:
            anomalies_count = int(np.sum(result["anomaly_flags"]))
            summary["anomalies_detected"] = anomalies_count
            summary["avg_anomaly_score"] = float(np.mean(result["anomaly_scores"]))

        summary["avg_weighted_score"] = float(weighted_score.mean())
        summary["max_weighted_score"] = float(weighted_score.max())

        # Top entities by weighted score
        top_idx = weighted_score.argsort()[::-1][:10]
        top_anomalies = []
        for i in top_idx:
            row = out_df.iloc[i]
            feat_row = features.iloc[i]
            top_feats = np.abs(feat_row.values).argsort()[::-1][:5]
            top_features = [feat_row.index[j] for j in top_feats]

            entry = {
                "entity_id": row.get(entity_id_col, i),
                "weighted_score": float(weighted_score.iloc[i]),
                "amount": float(row.get("amount", 0)),
                "top_features": top_features,
            }
            if result["mode"] == "supervised":
                entry["risk_score"] = float(result["risk_scores"][i])
            else:
                entry["anomaly_score"] = float(result["anomaly_scores"][i])

            top_anomalies.append(entry)

        result["summary"] = summary

        # LLM analysis
        llm_text = self.llm_agent.generate_analysis(summary, top_anomalies)
        result["llm_analysis"] = llm_text

        return result