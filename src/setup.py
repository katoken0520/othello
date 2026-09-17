import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# OSに応じた最適化オプションの設定
if sys.platform == "win32":
    # Windows (MSVC) 用のオプション
    # /O2: 最適化, /std:c++17: C++17規格, /utf-8: 文字コードの警告回避
    extra_compile_args = ['/O2', '/std:c++17', '/utf-8']
else:
    # Mac / Linux 用のオプション
    extra_compile_args = ['-O3', '-march=native', '-std=c++17']

ext_modules = [
    Extension(
        'engine', 
        sources=['src/cpp/othello_core.cpp'], 
        include_dirs=[pybind11.get_include()],
        language='c++',
        # ↓ extra_compile_args に '/arch:AVX2' などを追加
        extra_compile_args=['/std:c++17', '/O2', '/utf-8', '/arch:AVX2', '/fp:fast']
    ),
]

setup(
    name="engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)