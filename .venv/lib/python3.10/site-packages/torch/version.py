from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.7.1+cu118'
debug = False
cuda: Optional[str] = '11.8'
git_version = 'e2d141dbde55c2a4370fac5165b0561b6af4798b'
hip: Optional[str] = None
xpu: Optional[str] = None
