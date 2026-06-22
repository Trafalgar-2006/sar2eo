import sys, os, shutil, inspect, torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

print('=' * 55)
print('FINAL AUDIT - All 8 checks')
print('=' * 55)

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import L1Loss, GANLoss, FFTLoss, VGGPerceptualLoss
from utils.metrics import compute_metrics, _get_lpips
from utils.visualize import save_triplets

print('[1/8] Imports ... OK')

G = UNetGenerator(1, 3, 64); G.init_weights(); G.eval()
with torch.no_grad():
    out = G(torch.randn(2, 1, 256, 256))
assert out.shape == (2, 3, 256, 256)
assert out.min() >= -1.01 and out.max() <= 1.01
print('[2/8] Generator shape/range ... OK')

D = PatchGANDiscriminator(1, 3, 64, 3); D.init_weights(); D.eval()
with torch.no_grad():
    d = D(torch.randn(2, 1, 256, 256), torch.randn(2, 3, 256, 256))
assert d.shape == (2, 1, 30, 30)
print('[3/8] Discriminator shape ... OK')

G.train(); D.train()
sar = torch.randn(2, 1, 256, 256); real = torch.randn(2, 3, 256, 256)
fake = G(sar)
total = L1Loss()(fake, real)*100 + GANLoss()(D(sar, fake), True) + FFTLoss()(fake, real)*10 + VGGPerceptualLoss()(fake, real)*10
total.backward()
print(f'[4/8] Losses + grads ... OK  G_total={total.item():.2f}')

G.eval()
preds   = [torch.randn(3, 256, 256) for _ in range(2)]
targets = [torch.randn(3, 256, 256) for _ in range(2)]
results = compute_metrics(preds, targets)
assert all(not (v != v) for v in results.values()), 'NaN in metrics!'
lpips_val = results['lpips']
print(f'[5/8] Metrics (no NaN) ... OK  lpips={lpips_val:.4f}')

fn1 = _get_lpips('cpu'); fn2 = _get_lpips('cpu')
assert fn1 is fn2, 'Singleton broken'
print('[6/8] LPIPS singleton ... OK')

os.makedirs('outputs/audit', exist_ok=True)
save_triplets([], [], [], 'outputs/audit', 'e')   # n=0 guard
save_triplets([torch.randn(1,256,256)], [torch.randn(3,256,256)], [torch.randn(3,256,256)], 'outputs/audit', 's')  # n=1 axes
shutil.rmtree('outputs/audit', ignore_errors=True)
print('[7/8] Visualize n=0/n=1 edge cases ... OK')

import infer
sig = inspect.signature(infer.load_model)
assert sig.return_annotation == torch.nn.Module, f'Bad annotation: {sig.return_annotation}'
assert 'weights_only=False' in inspect.getsource(infer.load_model), 'Missing weights_only'
assert 'torch.amp.autocast' in inspect.getsource(infer.run_inference), 'Old autocast API'
# Check no Unicode arrows in print() calls
src = inspect.getsource(infer)
for line in src.split('\n'):
    if 'print(' in line and '\u2192' in line:
        raise AssertionError(f'Unicode arrow in print: {line.strip()}')
print('[8/8] infer.py annotations/API/Unicode ... OK')

print()
print('=' * 55)
print('ALL 8 CHECKS PASSED - READY TO PUSH')
print('=' * 55)
