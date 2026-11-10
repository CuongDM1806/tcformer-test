"""Checks for the target time-scale offsets. Run from the repo root: python tests/test_target_timescale.py"""
import sys, copy, yaml, torch
sys.path.insert(0, '.')
from torch.utils.data import DataLoader, TensorDataset
from models.hada_tcformer import HADATCFormer

cfg = yaml.safe_load(open('configs/hada_tcformer.yaml'))['model_kwargs']
cfg = {k: v for k, v in cfg.items() if k != 'log_every_n_batches'}
N_CH, T, N_CLS = 22, 1000, 4

def make(**over):
    torch.manual_seed(0)
    kw = {**cfg, **over}
    m = HADATCFormer(n_channels=N_CH, n_classes=N_CLS, max_epochs=10, log_every_n_batches=0, **kw)
    m.print = lambda *a, **k: None   # no trainer attached
    return m

torch.manual_seed(1)
x_src, y_src = torch.randn(8, N_CH, T), torch.randint(0, N_CLS, (8,))
x_tgt = torch.randn(8, N_CH, T)

# 1. Off vs on with zero offsets: identical weights and outputs.
base, ts = make(), make(target_timescale=True)
missing = set(ts.state_dict()) - set(base.state_dict())
assert all(k.endswith('target_step_offset') for k in missing) and len(missing) == 5, missing
ts.load_state_dict({**ts.state_dict(), **base.state_dict()})
base.eval(); ts.eval()
with torch.no_grad():
    assert torch.allclose(base(x_tgt), ts(x_tgt), atol=1e-6)
print('1 ok: zero offsets reproduce the baseline model exactly')

# 2. Offsets act only on target rows (eval mode) and only in the target domain.
with torch.no_grad():
    for off in ts._timescale_offsets():
        off.normal_(0, 0.5)
    src_plain = base.model.extract_features(x_src)
    with ts._timescale_target(8):
        mixed = ts.model.extract_features(torch.cat((x_src, x_tgt)))
    assert torch.allclose(mixed[:8], src_plain, atol=1e-5), 'source rows changed'
    assert not torch.allclose(mixed[8:], base.model.extract_features(x_tgt), atol=1e-4)
    assert torch.allclose(ts(x_src, domain='source'), base(x_src, domain='source'), atol=1e-5)
    assert not torch.allclose(ts(x_tgt, domain='target'), base(x_tgt, domain='target'), atol=1e-4)
print('2 ok: offsets change target rows only; source domain untouched')

# 3. Stage-1 gradients: learned when target_timescale_stage1, frozen otherwise.
for stage1 in (True, False):
    m = make(target_timescale=True, target_timescale_stage1=stage1)
    m.train()
    m.log = lambda *a, **k: None
    m.trainer = type('T', (), {'num_training_batches': 1, 'current_epoch': 0})()
    loss = m.training_step({'source': (x_src, y_src), 'target': x_tgt}, 0)
    loss.backward()
    grads = [o.grad for o in m._timescale_offsets()]
    if stage1:
        assert all(g is not None and g.abs().sum() > 0 for g in grads), 'no offset gradient'
    else:
        assert all(not o.requires_grad and g is None for o, g in zip(m._timescale_offsets(), grads))
print('3 ok: Stage-1 trains the offsets only when enabled')

# 4. Optimizer: offsets in a separate group without weight decay.
m = make(target_timescale=True)
opt = m.configure_optimizers()
opt = opt[0][0] if isinstance(opt, tuple) else opt[0]
assert len(opt.param_groups) == 2 and opt.param_groups[1]['weight_decay'] == 0.0
assert opt.param_groups[0]['weight_decay'] == cfg['weight_decay'] > 0
assert sum(p.numel() for p in opt.param_groups[1]['params']) == 5 * 48
print('4 ok: offsets exempt from weight decay; others keep', cfg['weight_decay'])

# 5. IM-TTA variants update exactly the chosen parameters.
loader = DataLoader(TensorDataset(x_tgt, torch.zeros(8, dtype=torch.long)), batch_size=8)
for mode in ('batchnorm', 'timescale', 'both'):
    m = make(target_timescale=True, im_tta_parameters=mode)
    m.load_state_dict(ts.state_dict() | {k: torch.zeros_like(v) for k, v in ts.state_dict().items() if k.endswith('target_step_offset')}, strict=False)
    before_bn = copy.deepcopy([p.detach().clone() for n, p in m.model.named_parameters() if 'conv_block' in n and n.endswith('weight') and p.dim() == 1])
    before_off = [o.detach().clone() for o in m._timescale_offsets()]
    m.adapt_to_target(loader)
    bn_moved = any(not torch.equal(a, b.detach()) for a, b in zip(before_bn, [p for n, p in m.model.named_parameters() if 'conv_block' in n and n.endswith('weight') and p.dim() == 1]))
    off_moved = any(not torch.equal(a, b.detach()) for a, b in zip(before_off, m._timescale_offsets()))
    assert bn_moved == (mode != 'timescale'), (mode, bn_moved)
    assert off_moved == (mode != 'batchnorm'), (mode, off_moved)
    s = m.last_timescale_summary
    assert len(s['blocks']) == 5 and s['tau_ratio'] > 0
    if mode == 'batchnorm':
        assert abs(s['tau_ratio'] - 1.0) < 1e-9, s['tau_ratio']
    print(f"5 ok: TTA={mode:9s} bn_moved={bn_moved} offsets_moved={off_moved} tau_ratio={s['tau_ratio']:.4f} "
          f"max|offset|={max(b['offset_abs_mean'] for b in s['blocks']):.4f}")

# 6. Without the option nothing changes in the TTA log path.
m = make()
assert m.target_timescale is False and m.model.timescale_mixers() == []
try:
    make(im_tta_parameters='timescale'); raise SystemExit('expected ValueError')
except ValueError:
    pass
print('6 ok: option off = original model; timescale TTA without offsets is rejected')
