from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
    Extension(
        "evolution_rules",
        [
            "ga_ner/utils/cython/*.pyx",
            "ga_ner/utils/cython/evolutionary_operations.cpp",
        ],
        language="c++",
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="evolutinary-rules",
    ext_modules=cythonize(ext_modules, language_level="3"),
)
