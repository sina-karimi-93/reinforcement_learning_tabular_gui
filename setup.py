
from cx_Freeze import setup, Executable

files = ['./src/']
build_exe_opt = {
    # 'includes': ['tkinter'],
    # 'include_files': files,
    'excludes': ['Tkinter'],
    'packages': ['babel']}

setup(
    name='Reinforcement Learning Tabular GUI',
    version='1.1',
    description='Reinforcement Learning Tabular GUI',
    executables=[Executable('main.py', base='Win32GUI')],
    # options={'build_exe': build_exe_opt}
)
