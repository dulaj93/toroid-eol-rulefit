"""
RuleFit classifier (Friedman & Popescu, 2008, "Predictive learning via rule
ensembles", Annals of Applied Statistics 2(3)).

The model is a sparse logistic model over two kinds of terms:

    logit P(fail | x) = a0 + sum_k a_k * r_k(x) + sum_j b_j * l_j(x_j)

* r_k(x) in {0, 1} are *rules*: conjunctions of simple conditions such as
  "winding_area_RH_pct > 78.4 and wip_hold_h > 20.5". Every node of every tree
  in a randomised tree ensemble defines one rule (the path from the root).
* l_j(x_j) are winsorised, scaled linear terms of the numeric features.
* The coefficients are fitted with an L1 (lasso) penalty, which keeps only a
  small subset of rules/terms -> a compact, human-readable model.

Rule generation follows the paper's ISLE recipe: trees with a random number of
leaves (exponential distribution), each grown on a random half of the data,
fitted to the current residuals and added with a small memory/shrinkage
parameter (nu = 0.01). The trees only *propose* candidate rules; the final
prediction comes from the lasso-logistic model, not from tree voting.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class RuleFitClassifier:
    def __init__(self, n_trees=300, mean_leaves=6, max_depth=3, memory_par=0.01, sample_fract=0.5,
                 min_samples_leaf=30, min_support=0.005, C=0.05, class_weight=None,
                 lin_trim_quantile=0.025, max_iter=1000, random_state=0):
        self.n_trees = n_trees
        self.mean_leaves = mean_leaves
        self.max_depth = max_depth
        self.memory_par = memory_par
        self.sample_fract = sample_fract
        self.min_samples_leaf = min_samples_leaf
        self.min_support = min_support
        self.C = C
        self.class_weight = class_weight
        self.lin_trim_quantile = lin_trim_quantile
        self.max_iter = max_iter
        self.random_state = random_state

    # ------------------------------------------------------------------ encoding
    def _fit_encoder(self, X, cat_features):
        self.cat_features_ = [c for c in X.columns if c in cat_features]
        self.num_features_ = [c for c in X.columns if c not in cat_features]
        self.input_features_ = list(X.columns)
        self.ohe_ = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  sparse_output=False).fit(X[self.cat_features_])
        num = X[self.num_features_].astype(float)
        self.medians_ = num.median().fillna(0.0)          # all-missing column -> 0
        # trees: missing values go to a sentinel far below the observed range,
        # so a split can isolate "value is missing" cleanly
        self.obs_min_ = num.min().fillna(0.0)
        self.sentinel_ = self.obs_min_ - (num.max() - num.min()).fillna(1.0).replace(0, 1.0) - 1.0
        self.lo_ = num.quantile(self.lin_trim_quantile).fillna(0.0)
        self.hi_ = num.quantile(1 - self.lin_trim_quantile).fillna(0.0)

    def _encode(self, X):
        """Design matrix for the trees: numeric (missing -> sentinel) + one-hot categoricals."""
        num = X[self.num_features_].astype(float).fillna(self.sentinel_)
        oh = pd.DataFrame(self.ohe_.transform(X[self.cat_features_].astype(str)),
                          columns=[n.replace("_infrequent_sklearn", "=OTHER")
                                   for n in self._ohe_names()], index=X.index)
        E = pd.concat([num, oh], axis=1)
        return E

    def _ohe_names(self):
        names = []
        for col, cats in zip(self.cat_features_, self.ohe_.categories_):
            infreq = getattr(self.ohe_, "infrequent_categories_", None)
            idx = self.cat_features_.index(col)
            inf = set(infreq[idx]) if infreq is not None and infreq[idx] is not None else set()
            names += [f"{col}={c}" for c in cats if c not in inf]
            if inf:
                names.append(f"{col}=OTHER")
        return names

    def _linear_terms(self, X):
        num = X[self.num_features_].astype(float).fillna(self.medians_)
        num = num.clip(self.lo_, self.hi_, axis=1)
        return (num - self.lin_mean_) * self.lin_scale_

    # ------------------------------------------------------------------ rules
    def _node_rules(self, tree, names):
        """Text of the path condition for every node (root excluded)."""
        t = tree.tree_
        conds = {0: []}
        stack = [0]
        while stack:
            n = stack.pop()
            if t.children_left[n] == -1:
                continue
            f, thr = names[t.feature[n]], t.threshold[n]
            for child, op in ((t.children_left[n], "<="), (t.children_right[n], ">")):
                conds[child] = conds[n] + [(f, op, thr)]
                stack.append(child)
        out = {}
        for node, cs in conds.items():
            if node == 0:
                continue
            # simplify: keep the tightest bound per (feature, direction)
            best = {}
            for f, op, thr in cs:
                k = (f, op)
                if k not in best or (op == "<=" and thr < best[k]) or (op == ">" and thr > best[k]):
                    best[k] = thr
            parts = []
            for (f, op), thr in sorted(best.items()):
                if "=" in f:                                        # one-hot column
                    parts.append(f if op == ">" else f.replace("=", " != ", 1))
                elif thr < self.obs_min_[f]:                        # split between sentinel and data
                    parts.append(f"{f} is missing" if op == "<=" else f"{f} is present")
                else:
                    parts.append(f"{f} {op} {thr:.4g}")
            out[node] = " and ".join(parts)
        return out

    def _rule_matrix(self, E):
        cols = []
        for t, nodes in self.rule_nodes_:
            ind = t.decision_path(E.values.astype(np.float32)).tocsc()
            cols.append(ind[:, nodes])
        return sparse.hstack(cols).tocsr().astype(np.float32)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X, y, cat_features=()):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y, dtype=float)
        self._fit_encoder(X, cat_features)
        E = self._encode(X)
        self.design_names_ = list(E.columns)
        Ev = E.values.astype(np.float32)
        n = len(y)

        # 1) ISLE tree ensemble -> candidate rules
        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        F = np.full(n, np.log(p0 / (1 - p0)))
        trees = []
        for m in range(self.n_trees):
            idx = rng.choice(n, int(self.sample_fract * n), replace=False)
            leaves = 2 + int(rng.exponential(self.mean_leaves - 2))
            r = y - _sigmoid(F)
            t = DecisionTreeRegressor(max_leaf_nodes=leaves, max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=int(rng.integers(1 << 30)))
            t.fit(Ev[idx], r[idx])
            F += self.memory_par * 4 * t.predict(Ev)   # 4 = 1/p(1-p) at p=0.5 (Newton step scale)
            trees.append(t)

        # 2) unique rules with reasonable support
        all_nodes, texts, seen = [], [], {}
        for t in trees:
            nr = self._node_rules(t, self.design_names_)
            ind = t.decision_path(Ev).tocsc()
            keep = []
            for node, txt in nr.items():
                col = ind[:, node]
                s = col.nnz / n
                if not (self.min_support <= s <= 1 - self.min_support):
                    continue
                key = hash(col.indices.tobytes())
                comp = hash(np.setdiff1d(np.arange(n), col.indices).tobytes())
                if key in seen or comp in seen:
                    continue
                seen[key] = True
                keep.append(node)
                texts.append(txt)
            if keep:
                all_nodes.append((t, keep))
        self.rule_nodes_ = all_nodes
        self.rules_text_ = texts

        # 3) linear terms (winsorised, scaled 0.4/std as in the paper)
        num = X[self.num_features_].astype(float).fillna(self.medians_).clip(self.lo_, self.hi_, axis=1)
        self.lin_mean_ = num.mean()
        std = num.std().fillna(0.0).replace(0, 1.0)
        self.lin_scale_ = 0.4 / std
        L = self._linear_terms(X)

        R = self._rule_matrix(E)
        self.rule_support_ = np.asarray(R.mean(axis=0)).ravel()
        Z = sparse.hstack([R, sparse.csr_matrix(L.values.astype(np.float32))]).tocsr()

        self.term_names_ = self.rules_text_ + [f"linear: {c}" for c in self.num_features_]
        self.term_type_ = ["rule"] * len(self.rules_text_) + ["linear"] * len(self.num_features_)
        # term std on training data -> Friedman importance
        lin_std = L.std().values
        self.term_std_ = np.r_[np.sqrt(self.rule_support_ * (1 - self.rule_support_)), lin_std]
        self.term_mean_ = np.r_[self.rule_support_, np.zeros(len(self.num_features_))]
        self._Z, self._y = Z, y.astype(int)
        return self.fit_lasso(self.C, self.class_weight)

    def fit_lasso(self, C, class_weight=None):
        """4) Sparse (L1) logistic regression over rules + linear terms.
        Can be re-run with another C / class weight without regenerating rules."""
        self.C, self.class_weight = C, class_weight
        self.lasso_ = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, class_weight=class_weight,
                                         intercept_scaling=1000.0,  # large -> intercept ~unpenalised
                                         max_iter=self.max_iter, tol=1e-4,
                                         random_state=self.random_state)
        self.lasso_.fit(self._Z, self._y)
        self.coef_ = self.lasso_.coef_.ravel()
        self.intercept_ = float(self.lasso_.intercept_[0])
        self.n_terms_ = int((self.coef_ != 0).sum())
        self.n_rules_ = int((self.coef_[:len(self.rules_text_)] != 0).sum())
        return self

    def compact(self):
        """Drop cached training matrix and unused rules before saving."""
        self._Z = self._y = None
        return self

    def _terms(self, X):
        E = self._encode(X)
        R = self._rule_matrix(E)
        L = self._linear_terms(X)
        return sparse.hstack([R, sparse.csr_matrix(L.values.astype(np.float32))]).tocsr()

    def decision_function(self, X):
        return np.asarray(self._terms(X) @ self.coef_).ravel() + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.c_[1 - p, p]

    # ------------------------------------------------------------------ interpretation
    def _term_features(self):
        """Original input features used by each term."""
        feats = []
        for name, typ in zip(self.term_names_, self.term_type_):
            if typ == "linear":
                feats.append([name.replace("linear: ", "")])
                continue
            used = set()
            for part in name.split(" and "):
                tok = part.split(" ")[0].split("=")[0]
                used.add(tok)
            feats.append(sorted(used))
        return feats

    def get_rules(self, nonzero=True):
        imp = np.abs(self.coef_) * self.term_std_
        df = pd.DataFrame({"term": self.term_names_, "type": self.term_type_, "coef": self.coef_,
                           "odds_ratio": np.exp(self.coef_),
                           "support": np.r_[self.rule_support_, np.full(len(self.num_features_), np.nan)],
                           "importance": imp})
        if nonzero:
            df = df[df.coef != 0]
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def feature_importance(self):
        """Friedman & Popescu (2008) eq. 35: linear importance plus each rule's
        importance shared equally among the features in that rule."""
        imp = np.abs(self.coef_) * self.term_std_
        out = pd.Series(0.0, index=self.input_features_)
        for k, fs in enumerate(self._term_features()):
            if imp[k] == 0:
                continue
            for f in fs:
                if f in out.index:
                    out[f] += imp[k] / len(fs)
        return out.sort_values(ascending=False)

    def local_contributions(self, X):
        """Additive log-odds contribution of every input feature for each row:
        term contribution a_k * (t_k(x) - mean_k), shared equally among the term's
        features. Rows sum to decision_function(x) - expected logit."""
        T = self._terms(X).toarray()
        C = (T - self.term_mean_) * self.coef_
        out = np.zeros((len(X), len(self.input_features_)))
        pos = {f: i for i, f in enumerate(self.input_features_)}
        for k, fs in enumerate(self._term_features()):
            if self.coef_[k] == 0:
                continue
            fs = [f for f in fs if f in pos]
            for f in fs:
                out[:, pos[f]] += C[:, k] / len(fs)
        base = self.intercept_ + float(self.term_mean_ @ self.coef_)
        return pd.DataFrame(out, columns=self.input_features_, index=X.index), base

    def term_activation(self, x_row):
        """All non-zero terms for a single unit (DataFrame with one row): the term
        value (1/0 = rule fires or not; scaled value for linear terms) and its
        contribution a_k * (t_k(x) - mean_k) to the log-odds."""
        t = self._terms(x_row).toarray().ravel()
        nz = np.flatnonzero(self.coef_)
        n_rules = len(self.rules_text_)
        return pd.DataFrame({
            "term": [self.term_names_[k] for k in nz],
            "type": [self.term_type_[k] for k in nz],
            "coef": self.coef_[nz],
            "odds_ratio": np.exp(self.coef_[nz]),
            "support": [self.rule_support_[k] if k < n_rules else np.nan for k in nz],
            "value": t[nz],
            "contribution": (t[nz] - self.term_mean_[nz]) * self.coef_[nz],
        })

    def active_rules(self, x_row):
        """Non-zero rules that fire for a single unit (DataFrame with one row)."""
        T = self._terms(x_row).toarray().ravel()
        rules = self.get_rules()
        fired = []
        for _, r in rules.iterrows():
            k = self.term_names_.index(r.term)
            if r.type == "rule" and T[k] == 1:
                fired.append(r)
        return pd.DataFrame(fired)
