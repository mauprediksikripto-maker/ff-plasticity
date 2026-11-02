# ffplast.py — Forward-Forward vs Backprop under loss of plasticity
# Experiment library for the Colab notebook. Every run is resumable:
# per-task metrics are appended to <run>.jsonl, a checkpoint is written
# every few tasks, and a <run>.done marker skips finished runs.

import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("bp", "bp_cbp", "bpn", "bpg", "ff", "ff_cbp", "dfa", "dfan", "lcn", "gcn")
# bp      : standard MLP + softmax head, end-to-end backprop
# bp_cbp  : bp + simplified utility-based continual backprop (unit reset)
# bpg     : same architecture/objective as FF (goodness), but GLOBAL gradients
#           -> isolates the effect of locality (bpg vs ff) from the objective (bp vs bpg)
# ff      : Forward-Forward, layer-local goodness objective (Hinton, 2022)
# ff_cbp  : ff + the same unit-reset mechanism
# bpn     : bp + the same RMS normalisation before every layer as bpg/ff (softmax head, no label input)
#           -> tests whether normalisation alone explains the bp vs bpg difference
# dfa     : direct feedback alignment (Noekland, 2016): the output error reaches every hidden layer
#           through a fixed random matrix; no weight transport, no gradient between layers
# dfan    : dfa + RMS normalisation (local counterpart of bpn)
# lcn     : layer-wise local classifiers (Noekland & Eidnes, 2019; Belilovsky et al., 2020): every
#           hidden layer has its own softmax head and learns only from its own cross-entropy; RMS norm
# gcn     : identical to lcn but the summed loss is backpropagated through all layers (global twin)


# ----------------------------------------------------------------- data
def load_dataset(name, root):
    """Return (Xtr uint8 [N,D], ytr int64, Xte, yte) as CPU tensors, cached to `root`."""
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, f"{name}_u8.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    if name == "synthetic":  # tiny dataset for smoke tests only
        g = torch.Generator().manual_seed(0)
        protos = torch.rand(10, 64, generator=g)
        def make(n):
            y = torch.randint(0, 10, (n,), generator=g)
            x = (protos[y] + 0.3 * torch.rand(n, 64, generator=g)).clamp(0, 1)
            return (x * 255).to(torch.uint8), y
        Xtr, ytr = make(3000); Xte, yte = make(1000)
    else:
        import torchvision
        if name == "mnist":
            tr = torchvision.datasets.MNIST(root, train=True, download=True)
            te = torchvision.datasets.MNIST(root, train=False, download=True)
            Xtr, ytr = tr.data.reshape(len(tr.data), -1), tr.targets.long()
            Xte, yte = te.data.reshape(len(te.data), -1), te.targets.long()
        elif name in ("cifar10", "cifar100"):
            ds = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
            tr = ds(root, train=True, download=True)
            te = ds(root, train=False, download=True)
            Xtr = torch.from_numpy(tr.data).reshape(len(tr.data), -1)
            Xte = torch.from_numpy(te.data).reshape(len(te.data), -1)
            ytr, yte = torch.tensor(tr.targets).long(), torch.tensor(te.targets).long()
        else:
            raise ValueError(name)
    out = (Xtr.contiguous(), ytr, Xte.contiguous(), yte)
    torch.save(out, cache)
    return out


class TaskStream:
    """Deterministic task sequence. kind = 'permuted' (input permutation per task)
    or 'label_permuted' (fixed images, label permutation per task)."""

    def __init__(self, data, kind, n_train, n_test, seed, device):
        Xtr, ytr, Xte, yte = data
        self.Xtr, self.ytr = Xtr.to(device), ytr.to(device)   # kept as uint8 (RAM-friendly)
        self.Xte, self.yte = Xte.to(device), yte.to(device)
        x = Xtr.float() / 255.0
        self.mean, self.std = x.mean().item(), x.std().item()
        del x
        self.kind, self.n_train, self.n_test = kind, n_train, n_test
        self.seed, self.device = seed, device
        self.dim = Xtr.shape[1]

    def _gen(self, t):
        return torch.Generator().manual_seed(1_000_003 * self.seed + t)

    def prep(self, xu8):
        return (xu8.float() / 255.0 - self.mean) / self.std

    def _pair(self, X, y, cls, n, g):
        idx = ((y == cls[0]) | (y == cls[1])).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g).to(self.device)][:n]
        return X[idx], (y[idx] == cls[1]).long()

    def task(self, t):
        g = self._gen(t)
        if self.kind == "class_pairs":
            # Continual-ImageNet-style binary tasks (Dohare et al., 2024): two new classes per task,
            # all their training images, accuracy on their held-out test images (chance 0.5)
            cls = torch.randperm(int(self.ytr.max()) + 1, generator=g)[:2].to(self.device)
            xtr, ytr = self._pair(self.Xtr, self.ytr, cls, self.n_train, g)
            xte, yte = self._pair(self.Xte, self.yte, cls, self.n_test, g)
            return xtr, ytr, xte, yte
        itr = torch.randperm(len(self.ytr), generator=g)[: self.n_train].to(self.device)
        ite = torch.randperm(len(self.yte), generator=g)[: self.n_test].to(self.device)
        xtr, ytr = self.Xtr[itr], self.ytr[itr]
        xte, yte = self.Xte[ite], self.yte[ite]
        if self.kind == "permuted":
            p = torch.randperm(self.dim, generator=g).to(self.device)
            xtr, xte = xtr[:, p], xte[:, p]
        elif self.kind == "label_permuted":
            lp = torch.randperm(10, generator=g).to(self.device)
            ytr, yte = lp[ytr], lp[yte]
        elif self.kind == "random_label":
            # memorisation regime (cf. Lyle et al., 2023): fresh random labels each task;
            # plasticity = ability to fit them, so accuracy is measured on the task's own data
            ytr = torch.randint(0, 10, (len(itr),), generator=g).to(self.device)
            xte, yte = xtr[: self.n_test], ytr[: self.n_test]
        else:
            raise ValueError(self.kind)
        return xtr, ytr, xte, yte


# ---------------------------------------------------------------- model
def rms_norm(h, eps=1e-6):
    # length normalisation between layers (Hinton, 2022); scaled to unit mean square.
    # Works for vectors [B, D] and feature maps [B, C, H, W] (normalised over all features).
    ms = h.pow(2).flatten(1).mean(1).view(-1, *([1] * (h.dim() - 1)))
    return h / (ms.sqrt() + eps)


IMG_SHAPE = {"mnist": (1, 28, 28, False), "cifar10": (3, 32, 32, True), "cifar100": (3, 32, 32, True),
             "synthetic": (1, 8, 8, False)}
# (channels, height, width, stored channels-last)


class ConvNet(nn.Module):
    """Small LeNet-style CNN with the same condition logic as Net (robustness check):
    `depth - 1` conv layers (3x3, stride 2, `width` channels, ReLU) followed by one fully
    connected hidden layer of `fc_width` ReLU units; every hidden layer is a goodness layer in
    BP-G/FF. Goodness-based conditions receive the label as extra constant input channels."""

    def __init__(self, dataset, width, depth, mode, n_classes=10, label_energy=0.5, fc_width=500):
        super().__init__()
        if mode.endswith("_cbp"):
            raise ValueError("unit reset is implemented for the MLP only")
        if mode in ("dfa", "dfan"):
            raise ValueError("feedback alignment is implemented for the MLP only")
        self.mode = mode
        self.goodness_based = mode in ("bpg", "ff")
        self.local = mode in ("ff", "lcn")
        self.normed = self.goodness_based or mode in ("bpn", "lcn", "gcn")
        self.n_classes = n_classes
        self.C, self.H, self.W, self.channels_last = IMG_SHAPE[dataset]
        # per position, the label code carries label_energy x the energy of the image channels
        self.label_scale = math.sqrt(label_energy * self.C)
        c = self.C + (n_classes if self.goodness_based else 0)
        self.layers = nn.ModuleList()
        h, w = self.H, self.W
        for _ in range(depth - 1):
            self.layers.append(nn.Conv2d(c, width, 3, stride=2, padding=1)); c = width
            h, w = (h + 1) // 2, (w + 1) // 2
        self.layers.append(nn.Linear(width * h * w, fc_width))
        self.head = None if self.goodness_based else nn.Linear(fc_width, n_classes)
        # auxiliary heads of the layer-wise classifier conditions (all hidden layers but the last)
        self.aux = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(width * 16, n_classes))
                                  for _ in range(depth - 1)]) if mode in ("lcn", "gcn") else None
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)

    def _img(self, x):
        B = x.shape[0]
        if self.channels_last:
            return x.view(B, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
        return x.view(B, self.C, self.H, self.W)

    def embed(self, x, labels=None):
        img = self._img(x)
        B = img.shape[0]
        if labels is None:
            code = torch.full((B, self.n_classes), 1.0 / self.n_classes, device=x.device)
        else:
            code = F.one_hot(labels, self.n_classes).float()
        code = (self.label_scale * code)[:, :, None, None].expand(B, self.n_classes, self.H, self.W)
        return torch.cat([img, code], dim=1)

    def hidden(self, inp):
        h = inp if inp.dim() == 4 else self._img(inp)
        hs = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                h = h.flatten(1)
            if self.normed:
                h = rms_norm(h)
            if self.local:
                h = h.detach()
            h = F.relu(layer(h))
            hs.append(h)
        return hs

    def logits(self, hs):
        return self.head(hs[-1])

    def aux_logits(self, hs):
        return [a(h) for a, h in zip(self.aux, hs[:-1])] + [self.head(hs[-1])]


def build_net(arch, dim, dataset, width, depth, method, label_energy=0.5, n_classes=10):
    if arch == "mlp":
        return Net(dim, width, depth, method, n_classes=n_classes, label_energy=label_energy)
    if arch == "cnn":
        return ConvNet(dataset, width, depth, method, n_classes=n_classes, label_energy=label_energy)
    raise ValueError(arch)


class Net(nn.Module):
    def __init__(self, in_dim, width, depth, mode, n_classes=10, label_energy=0.5):
        super().__init__()
        self.mode = mode
        self.goodness_based = mode in ("bpg", "ff", "ff_cbp")
        self.local = mode in ("ff", "ff_cbp", "lcn", "dfa", "dfan")
        self.normed = self.goodness_based or mode in ("bpn", "dfan", "lcn", "gcn")
        self.in_dim, self.n_classes = in_dim, n_classes
        self.label_scale = math.sqrt(label_energy * in_dim)
        d = in_dim + (n_classes if self.goodness_based else 0)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(d, width)); d = width
        self.head = None if self.goodness_based else nn.Linear(width, n_classes)
        self.aux = nn.ModuleList([nn.Linear(width, n_classes) for _ in range(depth - 1)]) \
            if mode in ("lcn", "gcn") else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)
        if mode in ("dfa", "dfan"):
            # fixed random feedback matrices B_l (n_classes x width), drawn once from the run seed
            for l in range(depth):
                self.register_buffer(f"fb{l}", torch.randn(n_classes, width) / math.sqrt(n_classes))

    def feedback(self):
        return [getattr(self, f"fb{l}") for l in range(len(self.layers))]

    def aux_logits(self, hs):
        return [a(h) for a, h in zip(self.aux, hs[:-1])] + [self.head(hs[-1])]

    def embed(self, x, labels=None):
        """Concatenate a label code to the image (goodness-based nets only).
        labels=None -> neutral code (uniform 1/C), used for representation probes."""
        B = x.shape[0]
        if labels is None:
            code = torch.full((B, self.n_classes), 1.0 / self.n_classes, device=x.device)
        else:
            code = F.one_hot(labels, self.n_classes).float()
        return torch.cat([x, self.label_scale * code], dim=1)

    def hidden(self, inp):
        hs, h = [], inp
        for layer in self.layers:
            if self.normed:
                h = rms_norm(h)
            if self.local:
                h = h.detach()            # no gradient flows between layers -> local learning
            h = F.relu(layer(h))
            hs.append(h)
        return hs

    def outgoing_weights(self, l):
        if l + 1 < len(self.layers):
            return self.layers[l + 1].weight
        return self.head.weight if self.head is not None else None

    def logits(self, hs):
        return self.head(hs[-1])


def goodness(h):
    return h.pow(2).flatten(1).mean(dim=1)


def ff_loss(hs_pos, hs_neg, theta):
    loss, sep = 0.0, 0.0
    for hp, hn in zip(hs_pos, hs_neg):
        gp, gn = goodness(hp), goodness(hn)
        loss = loss + F.softplus(torch.cat([theta - gp, gn - theta])).mean()
        sep += (gp.mean() - gn.mean()).item()
    return loss, sep / len(hs_pos)


def random_wrong_labels(y, n_classes, gen):
    off = torch.randint(1, n_classes, y.shape, device=y.device, generator=gen)
    return (y + off) % n_classes


@torch.no_grad()
def predict(net, x, chunk=1000):
    preds = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        if net.goodness_based:
            scores = []
            for c in range(net.n_classes):
                lab = torch.full((xb.shape[0],), c, dtype=torch.long, device=xb.device)
                hs = net.hidden(net.embed(xb, lab))
                scores.append(sum(goodness(h) for h in hs))
            preds.append(torch.stack(scores, 1).argmax(1))
        else:
            preds.append(net.logits(net.hidden(xb)).argmax(1))
    return torch.cat(preds)


# -------------------------------------------- simplified continual backprop
class UnitReset:
    """Utility-based reinitialisation of low-utility hidden units, in the spirit of
    continual backprop (Dohare et al., 2024). Utility = running mean of |activation|
    x total magnitude of outgoing weights (1 for the last goodness layer)."""

    def __init__(self, net, opt, rho=1e-4, maturity=100, decay=0.99):
        self.net, self.opt = net, opt
        self.rho, self.maturity, self.decay = rho, maturity, decay
        dev = next(net.parameters()).device
        self.util = [torch.zeros(l.out_features, device=dev) for l in net.layers]
        self.age = [torch.zeros(l.out_features, device=dev) for l in net.layers]
        self.acc = [0.0 for _ in net.layers]
        self.n_reset = 0

    def state_dict(self):
        return {"util": self.util, "age": self.age, "acc": self.acc, "n_reset": self.n_reset}

    def load_state_dict(self, s):
        self.util, self.age, self.acc, self.n_reset = s["util"], s["age"], s["acc"], s["n_reset"]

    def _zero_opt_state(self, p, idx, dim):
        st = self.opt.state.get(p, {})
        for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            if k in st and st[k] is not None:
                st[k].index_fill_(dim, idx, 0.0)

    @torch.no_grad()
    def step(self, hs):
        for l, h in enumerate(hs):
            w_out = self.net.outgoing_weights(l)
            out_mag = w_out.abs().sum(0) if w_out is not None else torch.ones_like(self.util[l])
            contrib = h.abs().mean(0) * out_mag
            self.util[l].mul_(self.decay).add_((1 - self.decay) * contrib)
            self.age[l] += 1
            self.acc[l] += self.rho * h.shape[1]
            n = int(self.acc[l])
            if n < 1:
                continue
            self.acc[l] -= n
            eligible = (self.age[l] > self.maturity).nonzero().flatten()
            if eligible.numel() == 0:
                continue
            k = min(n, eligible.numel())
            idx = eligible[torch.topk(self.util[l][eligible], k, largest=False).indices]
            layer = self.net.layers[l]
            fan_in = layer.weight.shape[1]
            layer.weight[idx] = torch.randn(k, fan_in, device=idx.device) * math.sqrt(2.0 / fan_in)
            layer.bias[idx] = 0.0
            self._zero_opt_state(layer.weight, idx, 0)
            self._zero_opt_state(layer.bias, idx, 0)
            if w_out is not None:
                w_out[:, idx] = 0.0
                self._zero_opt_state(w_out, idx, 1)
            self.util[l][idx] = 0.0
            self.age[l][idx] = 0.0
            self.n_reset += k


# ---------------------------------------------------------------- metrics
def _dormant_erank(hs):
    dormant, erank = [], []
    for h in hs:
        if h.dim() == 4:              # conv layer: one unit = one channel (spatial mean)
            h = h.mean(dim=(2, 3))
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)  # diverged runs
        active_frac = (h > 0).float().mean(0)
        dormant.append((active_frac <= 0.01).float().mean().item())
        s = torch.linalg.svdvals(h)
        p = s / (s.sum() + 1e-12)
        p = p[p > 0]
        erank.append(float(torch.exp(-(p * p.log()).sum())))
    return dormant, erank


@torch.no_grad()
def representation_metrics(net, x_probe, y_probe=None):
    """Goodness-based nets are probed with the TRUE label code (the input they process during
    learning; primary metric) and with a neutral code (secondary). A neutral code can silence
    label-selective units and so inflate dormancy; reporting both makes that visible."""
    out = {"wmag": [l.weight.abs().mean().item() for l in net.layers]}
    if net.goodness_based:
        out["dormant"], out["erank"] = _dormant_erank(net.hidden(net.embed(x_probe, y_probe)))
        out["dormant_neutral"], out["erank_neutral"] = _dormant_erank(net.hidden(net.embed(x_probe)))
    else:
        out["dormant"], out["erank"] = _dormant_erank(net.hidden(x_probe))
        out["dormant_neutral"] = out["erank_neutral"] = None
    return out


# ------------------------------------------------------------------ runner
def run_id(bench, method, width, seed, arch="mlp", optimizer="adam"):
    extra = "" if (arch, optimizer) == ("mlp", "adam") else f"_{arch}_{optimizer}"
    return f"{bench}_{method}{extra}_w{width}_s{seed}"


def _sync():
    # push OS buffers to the Google Drive FUSE layer so uploads start immediately
    if hasattr(os, "sync"):
        os.sync()


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _sync()


def _append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())


def _truncate_jsonl(path, keep_below_task):
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and json.loads(ln)["task"] < keep_below_task]
    with open(path, "w") as f:
        f.writelines(lines)


def make_optimizer(params, name, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgdm":                      # SGD with heavy-ball momentum 0.9
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(name)


def run(bench_cfg, method, seed, width, depth, lr, out_dir, data, device,
        optimizer="adam", theta=2.0, label_energy=0.5, rho=1e-4, maturity=100,
        ckpt_every=10, n_tasks=None, log=print, tag=None, deadline=None, arch="mlp"):
    """Train one (benchmark, method, width, seed) run over the task sequence. Resumable.
    deadline: wall-clock time (time.time()) after which the run checkpoints and returns None."""
    os.makedirs(out_dir, exist_ok=True)
    rid = tag or run_id(bench_cfg["name"], method, width, seed, arch, optimizer)
    jsonl = os.path.join(out_dir, rid + ".jsonl")
    ckpt = os.path.join(out_dir, rid + ".ckpt")
    done = os.path.join(out_dir, rid + ".done")
    n_tasks = n_tasks or bench_cfg["n_tasks"]
    if os.path.exists(done):
        log(f"[skip] {rid} already finished"); return rid

    torch.manual_seed(seed); np.random.seed(seed)
    stream = TaskStream(data, bench_cfg["kind"], bench_cfg["n_train"], bench_cfg["n_test"], seed, device)
    net = build_net(arch, stream.dim, bench_cfg.get("dataset", "synthetic"), width, depth, method,
                    label_energy=label_energy, n_classes=bench_cfg.get("n_classes", 10)).to(device)
    opt = make_optimizer(net.parameters(), optimizer, lr)
    reset = UnitReset(net, opt, rho=rho, maturity=maturity) if method.endswith("_cbp") else None

    start = 0
    s = None
    if os.path.exists(ckpt):
        try:
            s = torch.load(ckpt, map_location=device, weights_only=False)
        except Exception as e:          # e.g. a checkpoint cut off when a Colab session died
            log(f"[restart] {rid}: unreadable checkpoint ({e!r}); rerunning from task 0")
            os.remove(ckpt)
    if s is not None:
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"])
        if reset is not None: reset.load_state_dict(s["reset"])
        torch.set_rng_state(s["rng_cpu"])
        if device.type == "cuda" and s.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(s["rng_cuda"])
        start = s["task"]
        log(f"[resume] {rid} from task {start}")
    _truncate_jsonl(jsonl, start)

    bs, epochs = bench_cfg["batch"], bench_cfg["epochs"]
    t0 = time.time()
    for t in range(start, n_tasks):
        xtr, ytr, xte, yte = stream.task(t)
        gdev = torch.Generator(device=device).manual_seed(7919 * seed + t)
        steps_per_epoch = max(1, xtr.shape[0] // bs)
        total = steps_per_epoch * epochs
        q_step = max(1, total // 4)
        acc_q1, sep_sum, loss_sum = None, 0.0, 0.0
        x_eval = stream.prep(xte)
        step = 0
        net.train()
        for _ in range(epochs):
            order = torch.randperm(xtr.shape[0], device=device, generator=gdev)
            for i in range(steps_per_epoch):
                b = order[i * bs:(i + 1) * bs]
                xb, yb = stream.prep(xtr[b]), ytr[b]
                if net.goodness_based:
                    yneg = random_wrong_labels(yb, net.n_classes, gdev)
                    hs_pos = net.hidden(net.embed(xb, yb))
                    hs_neg = net.hidden(net.embed(xb, yneg))
                    loss, sep = ff_loss(hs_pos, hs_neg, theta)
                    sep_sum += sep; hs_for_reset = hs_pos
                elif method in ("lcn", "gcn"):
                    hs = net.hidden(xb)
                    loss = sum(F.cross_entropy(z, yb) for z in net.aux_logits(hs))
                    hs_for_reset = hs
                elif method in ("dfa", "dfan"):
                    hs = net.hidden(xb)
                    logits = net.head(hs[-1].detach())
                    with torch.no_grad():
                        err = (F.softmax(logits, 1) - F.one_hot(yb, net.n_classes).float()) / yb.shape[0]
                    # surrogate whose gradient w.r.t. layer l's pre-activation is (B_l^T e) * relu'(a_l)
                    loss = F.cross_entropy(logits, yb) + sum((h * (err @ B)).sum() for h, B in zip(hs, net.feedback()))
                    hs_for_reset = hs
                else:
                    hs = net.hidden(xb)
                    loss = F.cross_entropy(net.logits(hs), yb)
                    hs_for_reset = hs
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.item() if (step % 20 == 0) else 0.0
                if reset is not None:
                    reset.step([h.detach() for h in hs_for_reset])
                step += 1
                if step == q_step:
                    net.eval()
                    acc_q1 = (predict(net, x_eval) == yte).float().mean().item()
                    net.train()
        net.eval()
        acc_end = (predict(net, x_eval) == yte).float().mean().item()
        npb = min(1000, x_eval.shape[0])
        rep = representation_metrics(net, x_eval[:npb], yte[:npb])
        rec = {"run": rid, "bench": bench_cfg["name"], "method": method, "width": width,
               "arch": arch, "optimizer": optimizer,
               "seed": seed, "lr": lr, "task": t, "acc_q1": acc_q1, "acc_end": acc_end,
               "goodness_sep": (sep_sum / step) if net.goodness_based else None,
               "n_reset": reset.n_reset if reset is not None else 0,
               "dormant": rep["dormant"], "erank": rep["erank"], "wmag": rep["wmag"],
               "dormant_neutral": rep["dormant_neutral"], "erank_neutral": rep["erank_neutral"],
               "elapsed_s": round(time.time() - t0, 1)}
        _append_jsonl(jsonl, rec)
        out_of_time = deadline is not None and time.time() > deadline and t + 1 < n_tasks
        if (t + 1) % ckpt_every == 0 or t + 1 == n_tasks or out_of_time:
            _atomic_save({"net": net.state_dict(), "opt": opt.state_dict(),
                          "reset": reset.state_dict() if reset is not None else None,
                          "rng_cpu": torch.get_rng_state(),
                          "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                          "task": t + 1}, ckpt)
        if t % 10 == 0 or t + 1 == n_tasks:
            log(f"{rid} task {t:4d} acc_end={acc_end:.3f} dormant={np.mean(rep['dormant']):.3f} "
                f"erank={np.mean(rep['erank']):.1f} ({time.time() - t0:.0f}s)")
        del xtr, ytr, xte, yte, x_eval
        if out_of_time:
            log(f"[pause] {rid} at task {t + 1}/{n_tasks}: time budget reached, checkpoint saved")
            return None
    with open(done, "w") as f:
        f.write("ok"); f.flush(); os.fsync(f.fileno())
    if os.path.exists(ckpt):
        os.remove(ckpt)   # keep Drive tidy; jsonl holds all results
    _sync()
    return rid


# ------------------------------------------------------- time budgeting
def seconds_per_task(bench_cfg, method, width, depth, lr, data, device, tmp_dir, n=3, **kw):
    """Measure wall-clock seconds per task on this GPU (first task excluded as warm-up)."""
    tag = f"cal_{bench_cfg['name']}_{method}_{kw.get('arch', 'mlp')}_{kw.get('optimizer', 'adam')}_w{width}"
    for ext in (".jsonl", ".ckpt", ".done"):
        p = os.path.join(tmp_dir, tag + ext)
        if os.path.exists(p):
            os.remove(p)
    run(bench_cfg, method, 0, width, depth, lr, tmp_dir, data, device, n_tasks=n, tag=tag,
        log=lambda *a: None, **kw)
    with open(os.path.join(tmp_dir, tag + ".jsonl")) as f:
        e = [json.loads(ln)["elapsed_s"] for ln in f if ln.strip()]
    return (e[-1] - e[0]) / (len(e) - 1) if len(e) > 1 else e[0]


def make_plan(spt, bench, budget_s, min_tasks, seed_options=(5, 4, 3)):
    """Choose the number of seeds and tasks so the main grid fits in budget_s.
    spt[bench][method] = seconds per task. All benchmarks are scaled by the same factor,
    preferring more seeds as long as every benchmark keeps at least min_tasks[bench] tasks."""
    best = None
    for ns in seed_options:
        per_task = {b: ns * sum(spt[b].values()) for b in bench}
        full = sum(per_task[b] * bench[b]["n_tasks"] for b in bench)
        f = min(1.0, budget_s / full)
        nt = {b: int(bench[b]["n_tasks"] * f) for b in bench}
        plan = {"seeds": ns, "n_tasks": nt, "scale": round(f, 3),
                "est_min": round(sum(per_task[b] * nt[b] for b in bench) / 60, 1)}
        if all(nt[b] >= min_tasks[b] for b in bench):
            return plan
        best = plan
    nt = {b: min_tasks[b] for b in bench}       # fallback: minimum tasks, fewest seeds
    ns = seed_options[-1]
    best.update(seeds=ns, n_tasks=nt, scale=None,
                est_min=round(sum(ns * sum(spt[b].values()) * nt[b] for b in bench) / 60, 1),
                warning="budget too small for min_tasks; plan exceeds budget")
    return best


def pilot_retention(bench_cfg, data, device, out_dir, lr=1e-3, n_tasks=40, width=500, depth=3,
                    window=10, **kw):
    """Run BP alone (seed 0) to check that the regime actually removes plasticity.
    Returns (retention, acc_peak, acc_last). Deletes and reruns any earlier pilot of the same config."""
    tag = f"pilot_{bench_cfg['name']}_n{bench_cfg['n_train']}_e{bench_cfg['epochs']}"
    for ext in (".jsonl", ".ckpt", ".done"):
        p = os.path.join(out_dir, tag + ext)
        if os.path.exists(p):
            os.remove(p)
    run(bench_cfg, "bp", 0, width, depth, lr, out_dir, data, device, n_tasks=n_tasks, tag=tag,
        log=lambda *a: None, **kw)
    with open(os.path.join(out_dir, tag + ".jsonl")) as f:
        acc = np.array([json.loads(ln)["acc_end"] for ln in f if ln.strip()])
    roll = np.convolve(acc, np.ones(window) / window, mode="valid") if len(acc) >= window else acc
    peak, last = float(roll.max()), float(acc[-window:].mean())
    return last / peak, peak, last


def tune_lr(bench_cfg, methods, grid, width, depth, data, device, out_dir, n_tasks=5, seed=100,
            deadline=None, log=print, **kw):
    """Pick, per method, the learning rate with the highest mean end-of-task accuracy on the first
    `n_tasks` tasks, using a seed not used for evaluation. Tuning runs are resumable and tagged
    'tune_' so that analysis code can exclude them. Returns {method: lr} and the full score table."""
    scores = {}
    for m in methods:
        for lr in grid:
            tag = f"tune_{bench_cfg['name']}_{m}_{kw.get('arch', 'mlp')}_{kw.get('optimizer', 'adam')}_lr{lr:g}"
            if deadline is not None and time.time() > deadline and not os.path.exists(os.path.join(out_dir, tag + ".done")):
                return None, scores
            run(bench_cfg, m, seed, width, depth, lr, out_dir, data, device, n_tasks=n_tasks, tag=tag,
                log=lambda *a: None, **kw)
            with open(os.path.join(out_dir, tag + ".jsonl")) as f:
                acc = [json.loads(ln)["acc_end"] for ln in f if ln.strip()]
            scores[(m, lr)] = float(np.mean(acc))
            if not np.isfinite(scores[(m, lr)]):
                scores[(m, lr)] = -1.0
    choice = {m: max(grid, key=lambda lr: scores[(m, lr)]) for m in methods}
    for m in methods:
        edge = " (grid edge)" if choice[m] in (grid[0], grid[-1]) else ""
        log(f"  {bench_cfg['name']:8s} {m:4s} -> lr {choice[m]:g}{edge}  "
            + "  ".join(f"{lr:g}:{scores[(m, lr)]:.3f}" for lr in grid))
    return choice, scores


# ---------------------------------------------------------------- analysis
def load_results(out_dir):
    import pandas as pd
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(out_dir, fn)) as f:
                rows += [json.loads(ln) for ln in f if ln.strip()]
    df = pd.DataFrame(rows)
    if len(df):
        for k in ("dormant", "erank", "wmag", "dormant_neutral", "erank_neutral"):
            if k in df:
                df[k + "_mean"] = df[k].apply(lambda v: np.mean(v) if isinstance(v, list) else np.nan)
    return df


def plasticity_table(df, window=10, last=10):
    """Per run: peak = best rolling-`window` mean of end-of-task accuracy, acc_last = mean over
    the last `last` tasks, retention = acc_last / peak (1 = no loss of plasticity).
    Using the peak (not the first tasks) avoids rewarding methods that merely start slowly."""
    import pandas as pd
    recs = []
    for (bench, method, width, seed), g in df.groupby(["bench", "method", "width", "seed"]):
        g = g.sort_values("task")
        roll = g["acc_end"].rolling(window, min_periods=1).mean()
        peak = roll.max(); a1 = g["acc_end"].tail(last).mean()
        recs.append({"bench": bench, "method": method, "width": width, "seed": seed,
                     "acc_peak": peak, "task_peak": int(g["task"].iloc[int(roll.values.argmax())]),
                     "acc_last": a1, "retention": a1 / peak if peak > 0 else np.nan,
                     "acc_q1_last": g["acc_q1"].tail(last).mean(),
                     "dormant_last": g["dormant_mean"].tail(last).mean(),
                     "dormant_neutral_last": g["dormant_neutral_mean"].tail(last).mean() if "dormant_neutral_mean" in g else np.nan,
                     "erank_neutral_last": g["erank_neutral_mean"].tail(last).mean() if "erank_neutral_mean" in g else np.nan,
                     "erank_last": g["erank_mean"].tail(last).mean(),
                     "n_tasks": g["task"].max() + 1})
    per_run = pd.DataFrame(recs)
    summ = per_run.groupby(["bench", "method", "width"]).agg(["mean", "std", "count"])
    return per_run, summ


def pairwise_tests(per_run, metric="retention", pairs=(("ff", "bpg"), ("ff", "bp"), ("bpg", "bp"),
                                                         ("ff_cbp", "ff"), ("bp_cbp", "bp"),
                                                         ("bpn", "bp"), ("bpg", "bpn"))):
    """Welch t-test + Mann-Whitney U per benchmark/width for the method pairs."""
    from scipy import stats
    import pandas as pd
    out = []
    for (bench, width), g in per_run.groupby(["bench", "width"]):
        for a, b in pairs:
            xa = g.loc[g.method == a, metric].dropna(); xb = g.loc[g.method == b, metric].dropna()
            if len(xa) < 2 or len(xb) < 2:
                continue
            t = stats.ttest_ind(xa, xb, equal_var=False)
            u = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            d = (xa.mean() - xb.mean()) / math.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2 + 1e-12)
            out.append({"bench": bench, "width": width, "A": a, "B": b, "metric": metric,
                        "mean_A": xa.mean(), "mean_B": xb.mean(), "cohen_d": d,
                        "welch_p": t.pvalue, "mwu_p": u.pvalue, "n_A": len(xa), "n_B": len(xb)})
    return pd.DataFrame(out)


def plot_curves(df, bench, width, metric, out_path, smooth=5, ylabel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"bp": "BP", "bp_cbp": "BP + reset", "bpg": "BP-G (global goodness)",
              "ff": "FF (local)", "ff_cbp": "FF + reset", "bpn": "BP + RMS norm",
              "dfa": "DFA", "dfan": "DFA-N", "lcn": "LC-N (local)", "gcn": "GC-N (global)"}
    styles = {"bp": ("#1f77b4", "-"), "bp_cbp": ("#1f77b4", "--"), "bpg": ("#7f7f7f", "-"),
              "ff": ("#d62728", "-"), "ff_cbp": ("#d62728", "--"), "bpn": ("#2ca02c", "-"),
              "dfa": ("#8c564b", "-"), "dfan": ("#8c564b", "--"), "lcn": ("#e377c2", "-"), "gcn": ("#17becf", "-")}
    sub = df[(df.bench == bench) & (df.width == width)]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=300)
    for m in METHODS:
        g = sub[sub.method == m]
        if g.empty:
            continue
        piv = g.pivot_table(index="task", columns="seed", values=metric)
        piv = piv.rolling(smooth, min_periods=1).mean()
        mu, sd = piv.mean(1), piv.std(1).fillna(0)
        c, ls = styles[m]
        ax.plot(mu.index, mu.values, color=c, ls=ls, lw=1.2, label=labels[m])
        ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Task number"); ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path
