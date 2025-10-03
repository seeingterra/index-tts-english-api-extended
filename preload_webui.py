import os, sys, traceback
print('LAUNCHER starting')
try:
    import soxr, librosa
    print('PRELOAD OK:', soxr.__file__)
except Exception:
    traceback.print_exc()
    sys.exit(1)
# exec webui
os.execv(sys.executable, [sys.executable, os.path.join(os.getcwd(), 'webui.py'), '--port', '7860', '--host', '127.0.0.1'])
