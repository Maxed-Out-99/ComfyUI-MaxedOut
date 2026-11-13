"""Release checks using synthetic images only; no model weights or personal media."""
import ast
from pathlib import Path
import torch
import torch.nn.functional as F

def functions(path,names):
 tree=ast.parse(Path(path).read_text(encoding='utf-8'))
 return compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),path,'exec')
ns={'F':F,'torch':torch}
exec(functions('nodes/krea2_edit_core.py',{'fit_encode_image','fit_boost_mask'}),ns)
class VAE:
 def encode(self,image):
  self.image=image
  return image.movedim(-1,1)
for h,w,th,tw in [(100,100,96,88),(120,240,96,96),(240,120,96,96),(96,96,96,96)]:
 image=torch.zeros(1,h,w,3);image[:,h//4:3*h//4,w//4:3*w//4]=1
 mask=image[...,0].clone();vae=VAE()
 ns['fit_encode_image'](image,vae,th//8,tw//8,{},(0,),'fit')
 cropped=ns['fit_boost_mask'](mask,image,th,tw)
 aligned=F.interpolate(cropped[:,None],size=vae.image.shape[1:3],mode='bicubic',antialias=True).clamp(0,1)[:,0]
 torch.testing.assert_close(aligned,vae.image[...,0])
 assert ns['fit_boost_mask'](None,image,th,tw) is None
print('PASS: boost-mask crop matches encoded image for square, near-square, portrait and landscape references')
# The original published model class remains unchanged.
import subprocess
old=ast.parse(subprocess.check_output(['git','show','HEAD:nodes/krea.py']).decode());new=ast.parse(Path('nodes/krea.py').read_text())
get=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
assert ast.dump(get(old,'Krea2EditModelPatchMXD'))==ast.dump(get(new,'Krea2EditModelPatchMXD'))
for tree in [old,new]:
 c=get(tree,'Krea2EditGroundedEncodeMXD');tree.schema=ast.dump(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='INPUT_TYPES'))
assert old.schema==new.schema
print('PASS: published legacy model and legacy prompt input schema preserved')
