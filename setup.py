from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "log_parser",               # 모듈 이름: 나중에 Python에서 import log_parser 로 사용됨
        ["log_parser.cpp"],         # C++ 소스 파일 목록
        cxx_std=17,                 # C++17 사용 (std::filesystem 때문에 필요)
    ),
]

setup(
    name="log_parser",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
