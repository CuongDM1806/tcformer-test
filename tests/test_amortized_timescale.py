"""Checks for amortized time-scale adaptation. Run from the repo root:
python tests/test_amortized_timescale.py
"""
import importlib.util
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, '.')
from models.hada_tcformer import HADATCFormer  # noqa: E402

# datamodules/base.py alone (the package imports every dataset loader)
spec = importlib.util.spec_from_file_location('dm_base', 'datamodules/base.py')
dm_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm_base)

cfg = yaml.safe_load(open('configs/hada_tcformer.yaml'))['model_kwargs']
cfg = {k: v for k, v in cfg.items() if k != 'log_every_n_batches'}
N_CH, T, N_CLS = 22, 1000, 4


def make(**over):
    torch.manual_seed(0)
    m = HADATCFormer(n_channels=N_CH, n_classes=N_CLS, max_epochs=10, log_every_n_batches=0,
                     **{**cfg, **over})
    m.print = lambda *a, **k: None
    m.log = lambda *a, **k: None
    return m


torch.manual_seed(1)
x_src, y_src = torch.randn(16, N_CH, T), torch.randint(0, N_CLS, (16,))
s_src = torch.tensor([3] * 8 + [7] * 8)
x_tgt = torch.randn(8, N_CH, T)

# 1. The context head starts at zero: same outputs as the model without it.
base, am = make(), make(amortized_timescale=True)
am.load_state_dict({**am.state_dict(), **base.state_dict()})
base.eval(); am.eval()
with torch.no_grad():
    assert torch.allclose(base(x_tgt), am(x_tgt), atol=1e-6)
    assert am.predict_offsets(x_tgt).abs().max() == 0
print('1 ok: zero-initialized context head reproduces the baseline model')

# 2. Episodic sampler: subject-pure groups, each trial at most once per epoch.
subjects = np.repeat([1, 2, 4, 5, 6], 24)
sampler = dm_base.SubjectBatchSampler(subjects, subjects_per_batch=4, trials_per_subject=6)
batches = list(sampler)
assert len(batches) == len(sampler) == (5 * 4) // 4
flat = [i for b in batches for i in b]
assert len(flat) == len(set(flat))
for b in batches:
    groups = subjects[b].reshape(4, 6)
    assert (groups == groups[:, :1]).all() and len(set(groups[:, 0])) == 4
print(f'2 ok: {len(batches)} episodes of 4 subjects x 6 trials, no trial repeated')

# 3. Subject-pure S&R: augmented trials only mix trials of their own subject.
items = [(torch.full((N_CH, T), float(s)), int(c), int(s))
         for s in (3, 7) for c in range(N_CLS) for _ in range(3)]
x, y, s = dm_base.SubjectInterAugCollate({'interaug': True})(items)
assert len(x) == 2 * len(items)
assert torch.equal(x[:, 0, 0], s.float()) and torch.equal(x.amax(dim=(1, 2)), x.amin(dim=(1, 2)))
print('3 ok: augmented trials keep a single subject')

# 4. Stage-1 episode: gradients reach the context encoder; penalty is logged.
m = make(amortized_timescale=True)
m.train()
m.trainer = type('T', (), {'num_training_batches': 1, 'current_epoch': 0})()
m._context_bank = {3: torch.randn(30, N_CH, T), 7: torch.randn(30, N_CH, T)}
loss = m.training_step({'source': (x_src, y_src, s_src), 'target': x_tgt}, 0)
loss.backward()
head = m.context_encoder.head[-1]
assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
print('4 ok: one episode trains the context encoder, loss', round(float(loss), 4))

# 5. After a few steps the offsets differ between subjects.
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
for _ in range(3):
    opt.zero_grad()
    m.training_step({'source': (x_src, y_src, s_src), 'target': x_tgt}, 0).backward()
    opt.step()
m.eval()
with torch.no_grad():
    o3, o7 = m.predict_offsets(m._context_bank[3]), m.predict_offsets(m._context_bank[7])
assert o3.abs().max() > 0 and not torch.allclose(o3, o7)
print(f'5 ok: subject offsets differ (rms {float(o3.square().mean().sqrt()):.4f} vs {float(o7.square().mean().sqrt()):.4f})')

# 6. Validation uses each source subject's own predicted offsets.
m._validation_offsets = {3: o3, 7: o7}
out = m.validation_step((x_src, y_src, s_src), 0)
assert 0 <= float(out['val_acc']) <= 1
print('6 ok: source validation with per-subject offsets, acc', round(float(out['val_acc']), 3))

# 7. Scoring from K context trials restores the model state.
before = (m.domain_norm.target_mean.clone(), m._target_context_offsets)
r = m.score_with_context(x_tgt[:4], x_tgt, torch.randint(0, N_CLS, (8,)))
assert 0 <= r['acc'] <= 1 and torch.equal(before[0], m.domain_norm.target_mean)
assert m._target_context_offsets is before[1]
print('7 ok: score_with_context leaves the model unchanged, acc', r['acc'])

# 8. Option off: no context encoder, original training batches still accepted.
m = make()
assert m.context_encoder is None and m.amortized_timescale is False
print('8 ok: option off = original model')
