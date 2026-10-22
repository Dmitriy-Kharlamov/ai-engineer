import sys

print("stdout: строка 1")
print("stderr: строка 1", file=sys.stderr)
print("stdout: строка 2")
print("stderr: строка 2", file=sys.stderr)
raise RuntimeError("noisy.py упал специально")
