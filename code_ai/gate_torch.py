# gate_torch.py ───────────────────────────────────────────────────────────────
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import code_ai.shared_variables as sv
from code_ai.plots import plot_correlation_map_from_sv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# Use all available CPU threads (fallback if sv.CPUS missing)
# ───────────────────── ④ Forecast head ──────────────────────
class ForecastHead(nn.Module):
    """
    Depthwise (per-feature) lag pooling + linear output.
    Each feature gets its own lag weights; no cross-feature coupling.
    """
    def __init__(self, feats, steps_ahead):
        super().__init__()
        # one learnable scalar per feature to modulate per-lag scores for that feature
        self.alpha = nn.Parameter(torch.zeros(feats))  # init ~ 0 => near-uniform early
        self.out   = nn.Linear(feats, steps_ahead)

    def forward(self, x):  # x: (B,L,F)
        # per-feature scores: s[b,l,f] = alpha[f] * x[b,l,f]
        s = x * self.alpha.view(1, 1, -1)            # (B,L,F)
        w = torch.softmax(s, dim=1)                  # normalize along L, independently for each feature
        ctx = (x * w).sum(dim=1)                     # (B,F) pooled per feature
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        return self.out(ctx)                         # (B,steps)
# ─────────────── full contextual gate ───────────────
def hard_topk_mask(P, M, K, min_prob=None):
    """
    P: (L,F) probabilities (or PD)
    M: max number of features to keep
    K: number of lags per kept feature (fixed)
    min_prob: if not None, drop a feature entirely if even its best lag < min_prob
    """
    L, F = P.shape
    M = max(1, min(M, F))
    K = max(1, min(K, L))

    # rank features by total mass
    s_f = P.sum(dim=0)                      # (F,)
    order = torch.argsort(s_f, descending=True)  # (F,)

    H = torch.zeros_like(P, dtype=torch.float32)
    kept = []

    for c in order.tolist():
        if len(kept) >= M:
            break

        col = P[:, c]                       # (L,)
        vals, idx = torch.topk(col, k=K)    # top-K lags for this feature

        if min_prob is not None:
            # if even the best lag is below threshold, skip this feature
            if vals.max().item() < min_prob:
                continue

        # keep this feature with exactly K lags
        H[idx, c] = 1.0
        kept.append(c)

    if len(kept) == 0:
        # no feature passed threshold; return all-zero mask and empty index list
        fidx = torch.empty(0, dtype=torch.long, device=P.device)
    else:
        fidx = torch.tensor(kept, dtype=torch.long, device=P.device)

    return H, fidx
class GateContextualModel(nn.Module):
    def __init__(self, win, feats, params, steps_ahead: int = None):
        super().__init__()
        temp0 = float(params.get("temp_start", 1.0))
        init_logit = float(params.get("init_logit", -2.5))

        self.M_feats = int(params.get("M_feats", feats))  # start fully open
        self.k_lags = int(params.get("k_lags", min(win, 2)))  # e.g., 2 lags/feat
        self.sparsity_warmup = int(params.get("sparsity_warmup_steps", 0))

        # joint logit gate (L, F)
        base = torch.full((win, feats), init_logit)
        self.joint_logits = nn.Parameter(base)

        self.joint_temp = temp0
        self.register_buffer("lambda_dead", torch.tensor(3e-3))  # was 2e-3
        self.register_buffer("lambda_feat", torch.tensor(float(params.get("lambda_feat", 3e-3))))
        self.register_buffer("lambda_lag", torch.tensor(float(params.get("lambda_lag", 2e-3))))
        self.hard_warmup_epochs = int(params.get("hard_warmup_epochs", 5))

        self._epoch = 0
        if sv.DECAY:
            self.register_buffer("decay", torch.linspace(2.0, 1.0, steps=win))
        else:
            self.register_buffer("decay", torch.ones(win))
        self.drop = nn.Dropout(p=float(params.get("dropout", 0.0)))
        self.head = ForecastHead(feats, steps_ahead)

    def clamp_parameters(self):
        with torch.no_grad():
            self.joint_logits.clamp_(-10.0, 10.0)

    def forward(self, x):
        logits = self.joint_logits.clamp(-10, 10)
        tau = max(self.joint_temp, 1e-6)
        P = torch.sigmoid(logits / tau)          # (L,F)

        M = int(max(1, min(self.M_feats, P.shape[1])))
        K = int(max(1, min(self.k_lags,  P.shape[0])))
        PD = P * self.decay[:, None]

        if self._epoch >= self.hard_warmup_epochs:
            H, _ = hard_topk_mask(PD, M, K)
            G = H + P - P.detach()
        else:
            G = P

        x = x * G
        x = self.drop(x)
        return self.head(x)

    def regularisation_loss(self):
        """
        λ_feat / λ_lag are always applied (after warmup, with a gentle ramp).
        λ_dead only ramps up when we're over the M×K budget (like before).
        """
        # warmup (unchanged)
        if self._epoch < self.sparsity_warmup:
            return self.joint_logits.new_tensor(0.0)

        logits = self.joint_logits.clamp(-10, 10)
        tau = max(self.joint_temp, 1e-6)
        P = torch.sigmoid(logits / tau)  # (L,F)
        PD = P * self.decay[:, None]  # decay-weighted

        nL, nF = P.shape
        M = int(max(1, min(self.M_feats, nF)))
        K = int(max(1, min(self.k_lags, nL)))

        # --- hard mask from PD, but detached so we don't backprop through the top-k indices
        PD_det = PD.detach()
        H, _ = hard_topk_mask(PD_det, M, K, min_prob=0.05)
        H = H.to(P.device)

        # ---- dead-gate penalty (budget-gated, like your old version)
        dens_now = P.mean()
        dens_budget = (M * K) / float(nL * nF)
        # <=1 → at/under budget → pressure=1 (base);  >1 → over budget → >1 pressure (bites harder)
        pressure = (dens_now / (dens_budget + 1e-8)).clamp_min(1.0).pow(2.0).detach()

        dead_w = (1.0 - P).detach()
        reg_dead = (self.lambda_dead * pressure) * (dead_w * F.softplus(logits)).mean()

        # ---- gentle ramp after warmup (unchanged style)
        T_total = max(1, sv.FINAL_EPOCHS - self.sparsity_warmup)
        t = self._epoch - self.sparsity_warmup
        ramp = max(0.0, min(t / T_total, 1.0))

        total_mass = PD.sum().clamp_min(1e-8)
        spill_mass = (PD * (1.0 - H)).sum()  # mass outside mask
        spill_frac = spill_mass / total_mass  # ∈ [0,1]

        reg_spill = ramp * self.lambda_feat * spill_frac

        align = (P - H).pow(2).mean()  # MSE between soft probs and hard mask
        reg_align = ramp * self.lambda_lag * align

        return reg_spill + reg_align + reg_dead

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y    # can be np.ndarray or np.memmap
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        # cast per sample to avoid duplicating arrays in RAM
        return (torch.from_numpy(self.X[i]).to(torch.float32),
                torch.from_numpy(self.y[i]).to(torch.float32))
# ─────────────────── train_gates (torch) ───────────────────

@torch.no_grad()
def p_raw(model):
    logits = model.joint_logits.clamp(-10, 10)
    tau = max(float(model.joint_temp), 1e-6)
    return torch.sigmoid(logits / tau)
def gate_train(
    X_train, y_train, X_val, y_val,
    win, feats, best_params, num_epochs,
    lr=1e-3, weight_decay=1e-4, batch_size=64, use_bar=False
):
    """
    CPU-oriented trainer. Optimizer steps per batch; LR scheduler per epoch.
    """
    device = torch.device("cpu")
    def _logit(p):
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1 - p))

    def quick_val():
        with torch.no_grad():
            s, n = 0.0, 0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                s += loss_fn(model(xb), yb).item() * yb.numel()
                n += yb.numel()
            return s / max(1, n)

    best_params["sparsity_warmup_steps"] = int(best_params["sparsity_warmup_frac"] * sv.FINAL_EPOCHS)
    prune_every = int(best_params.get("auto_prune_every", 1))
    M_floor = 1
    k_floor = min(3, win)
    best_params["M_feats"] = feats
    best_params["k_lags"] = win

    model = GateContextualModel(win, feats, best_params, sv.STEPS_AHEAD).to(device)

    temp_start = float(best_params.get("temp_start", 0.5))
    temp_end   = float(best_params.get("temp_end",   0.05))
    prune_frac_k = float(best_params.get("prune_frac_k", 0.25))
    prune_frac_M = float(best_params.get("prune_frac_M", 0.25))
    alpha_k_chunk = float(best_params.get("alpha_k_chunk", 0.50))  # chunk step toward target k
    alpha_M_chunk = float(best_params.get("alpha_M_chunk", 0.5))
    # real tolerance band (NOT 1e-12)
    prune_tol_rel = float(best_params.get("prune_tol_rel", 5e-3))  # 0.5% by default
    prune_tol_abs = float(best_params.get("prune_tol_abs", 0.0))
    # feature-mass target for M
    beta_k = float(best_params.get("beta_k", 0.2)) # closer to 0 more sparse
    beta_M = float(best_params.get("beta_M", 0.5))


    train_dl = DataLoader(ArrayDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_dl = DataLoader(ArrayDataset(X_val, y_val), batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

    # optim & sched
    # Collect model params excluding the lambdas
    regular_params = [p for n, p in model.named_parameters()
                      if not n.startswith("lambda_")]


    opt = optim.AdamW([{'params': regular_params}], lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    loss_fn = nn.MSELoss()

    iterator = trange(num_epochs, desc="Gate fit", dynamic_ncols=True) if use_bar else range(num_epochs)
    burn_in = int(best_params.get("auto_burn_in", 3))
    best_loss, wait = float("inf"), 0
    prune_calm_events = 0  # count consecutive prune events with no accepted cuts
    for epoch in iterator:
        model._epoch = int(epoch)
        # anneal temps
        progress = epoch / float(max(1, sv.FINAL_EPOCHS - 1))
        t = temp_start * (1.0 - progress) + temp_end * progress  # linear decay
        model.joint_temp = float(t)
        # train
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            data_loss = loss_fn(pred, yb)
            loss = data_loss + model.regularisation_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            model.clamp_parameters()

        model.eval()
        with torch.no_grad():
            loss_sum, count = 0.0, 0
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb.to(device))
                batch_loss = loss_fn(pred, yb)
                loss_sum += batch_loss.item() * yb.numel()
                count += yb.numel()
            val_loss = loss_sum / max(1, count)
        sched.step()

        if use_bar:
            with torch.no_grad():
                P = p_raw(model)
                open_mass = (P * model.decay[:, None]).sum().item()
                mean_p = P.mean().item()
                frac_on_50 = (P > 0.5).float().mean().item()
            if use_bar:
                iterator.set_postfix({
                    "val": round(val_loss, 6),
                    "p_open": round(mean_p, 3),
                    "mass": round(open_mass, 1),
                    "on>0.5": round(frac_on_50, 3),
                })

        # ----- early-stopping best by val_loss -----
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= sv.FINAL_PATIENCE and prune_calm_events >= 2:
                break
        if use_bar:
            print("")
        # ----- adaptive prune + rollback (chunk k AND chunk M, 1x quick_val per while-iter) -----
        if (epoch + 1) % prune_every == 0 and epoch >= burn_in:
            accepted_k = 0
            accepted_M = 0

            # budgets
            max_k_cuts = max(0, min(int(prune_frac_k * model.k_lags), model.k_lags - k_floor))
            max_M_cuts = max(0, min(int(prune_frac_M * model.M_feats), model.M_feats - M_floor))

            # baseline for this prune event
            baseline_now = float(val_loss)

            def tol(x: float) -> float:
                return max(prune_tol_abs, prune_tol_rel * max(1e-12, x))

            while (accepted_k < max_k_cuts or accepted_M < max_M_cuts):

                prev_k = int(model.k_lags)
                prev_M = int(model.M_feats)

                with torch.no_grad():
                    P_obs = p_raw(model)  # (L,F)
                    PD_obs = P_obs * model.decay[:, None]  # (L,F)

                    # effective hard mask with current budgets
                    H_obs, fidx_obs = hard_topk_mask(PD_obs, prev_M, prev_k, min_prob=0.05)
                    effective_M = len(fidx_obs)  # number of features actually used by the hard mask

                    open_mass = PD_obs.sum().item()

                    # ---------------- propose k_next (chunked) ----------------
                k_next = prev_k
                cap_left_k = max_k_cuts - accepted_k
                if prev_k > k_floor and cap_left_k > 0:
                    M_now = max(1, effective_M)  # use effective M, not nominal model.M_feats
                    raw_target_k = open_mass / M_now
                    target_k = int(math.ceil((1 - beta_k) * raw_target_k + beta_k * prev_k))
                    target_k = max(k_floor, min(prev_k, target_k))

                    if prev_k > target_k:
                        step_raw = int(math.ceil(alpha_k_chunk * (prev_k - target_k)))
                        step = min(step_raw, cap_left_k, prev_k - k_floor)
                        k_next = prev_k - step

                # ---------------- propose M_next (chunked) ----------------
                M_next = prev_M
                if prev_M > M_floor and accepted_M < max_M_cuts:
                    # HERE is the key change:
                    raw_target_M = effective_M  # aim at what the mask actually uses
                    target_M = int(math.ceil((1 - beta_M) * raw_target_M + beta_M * prev_M))
                    target_M = max(M_floor, min(prev_M, target_M))

                    if prev_M > target_M:
                        cap_left_M = max_M_cuts - accepted_M
                        step_raw = int(math.ceil(alpha_M_chunk * (prev_M - target_M)))
                        step = min(step_raw, cap_left_M, prev_M - M_floor)
                        M_next = prev_M - step

                # if nothing to do, stop
                if (k_next == prev_k) and (M_next == prev_M):
                    break

                # ---------------- evaluate JOINT move once ----------------
                model.k_lags = k_next
                model.M_feats = M_next
                v_joint = float(quick_val())
                # rollback for now
                model.k_lags = prev_k
                model.M_feats = prev_M

                # accept only if within tolerance band
                if v_joint <= baseline_now + tol(baseline_now):
                    model.k_lags = k_next
                    model.M_feats = M_next

                    accepted_k += (prev_k - k_next)
                    accepted_M += (prev_M - M_next)

                    baseline_now = v_joint
                else:
                    # reject and stop this prune event (keeps it to 1x quick_val per while-iter)
                    break

            prev_best = best_loss
            best_loss = min(best_loss, baseline_now)
            if best_loss < prev_best - 1e-5:
                wait = 0
            if (accepted_k + accepted_M) == 0:
                prune_calm_events += 1
            else:
                prune_calm_events = 0

    return model, best_loss

@torch.no_grad()
def compare_gates_torch(model, min_prob=0.05): #min_prob=0.05
    # raw probabilities
    P = p_raw(model)                          # (L,F)
    PD = P * model.decay[:, None]             # (L,F)

    L, F = PD.shape
    M = max(1, min(int(model.M_feats), F))
    K = max(1, min(int(model.k_lags),  L))

    # use EXACTLY the same hard mask logic as in forward
    H, _ = hard_topk_mask(PD, M, K, min_prob=min_prob)

    # visualise soft probs (as before)
    plot_correlation_map_from_sv(P.cpu().numpy())

    # return boolean mask derived from hard gate
    return (H > 0.5).cpu().numpy()