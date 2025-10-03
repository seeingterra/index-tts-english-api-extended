import sys, traceback
try:
    import soxr, librosa
    print("soxr OK:", soxr.__file__)
    print("librosa OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
print("IMPORT-TEST-SUCCESS")
