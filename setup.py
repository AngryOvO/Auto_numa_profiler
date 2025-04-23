# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "log_parser",            # 이 모듈은 Python에서 log_parser 로 임포트됩니다.
        ["log_parser.cpp"],      # 소스 파일 목록
        cxx_std=17,              # C++17 사용 (std::filesystem 이용)
    ),
]

setup(
    name="log_parser",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
