from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Flags de compilação para otimização e padrão C++
compile_args = ["/O2", "/std:c++17"]

extensions = [
    Extension(
        "modelo_simplificado", 
        ["modelo_simplificado.pyx"], 
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=compile_args, 
    )
]

setup(
    name="Modelo Chagas Simplificado",
    # Converte .pyx para .c
    ext_modules=cythonize(extensions, language_level="3"), 
)

