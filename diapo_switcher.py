from flask import Flask
import pyautogui
import subprocess
import os
import sys

app = Flask( __name__ )

@app.route( "/switch-diapo", methods=["GET"] )
def switch():
    print( "switch forward" )
    pyautogui.press('space')
    return ('', 200)

@app.route( "/back-diapo", methods=["GET"] )
def back_switch():
    print( "switch backward" )
    pyautogui.press('backspace')
    return ('', 200)

HOOK_CONTENT = '''from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

hiddenimports = collect_submodules('mediapipe.tasks')
hiddenimports += ['mediapipe.tasks.c']
datas = collect_data_files('mediapipe')
binaries = collect_dynamic_libs('mediapipe')
'''

def write_temp_hook(hook_path='hook-mediapipe.py'):
    with open(hook_path, 'w', encoding='utf-8') as f:
        f.write(HOOK_CONTENT)

def build_exe():
    hook_name = 'hook-mediapipe.py'
    write_temp_hook(hook_name)
    cmd = [
        sys.executable, '-m', 'PyInstaller', '--onefile', '--noconfirm', '--clean',
        '--additional-hooks-dir=.', '--hidden-import=mediapipe.tasks.c',
        '--add-data', 'hand_landmarker.task;.', os.path.basename(__file__)
    ]
    subprocess.run(cmd)
    try:
        os.remove(hook_name)
    except OSError:
        pass

if __name__ == '__main__':
    if '--build-exe' in sys.argv:
        build_exe()
    else:
        app.run( "0.0.0.0", port=9952 )
