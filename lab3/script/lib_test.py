import time
from ctypes import CDLL, c_char_p, c_int, c_float
from PIL import Image
import os
lib = CDLL("./libimgproc_cuda.so")
lib.process_image.argtypes = [c_char_p, c_char_p, c_char_p, c_int]
lib.process_image.restype = c_float
data = "data"
out = "out"
os.makedirs(out, exist_ok=True)
ms = [256, 512, 1024]
ks = [3, 5, 7]
filters = ["edge", "resize"]
def prepare(src, M, dst):
    img = Image.open(src).convert("RGB")
    img = img.resize((M,M))
    img.save(dst, "JPEG")
for img in os.listdir(data):
    if not img.lower().endswith(".jpg"):
        continue
    base = os.path.splitext(img)[0]
    for M in ms:
        tmp = f"{out}/{base}_M{M}.jpg"
        prepare(f"{data}/{img}", M, tmp)
        for K in ks:
            for f in filters:
                t0 = time.perf_counter()
                gpu_ms = lib.process_image(
                    tmp.encode(),
                    f"{out}/{base}_M{M}".encode(),
                    f.encode(),
                    K
                )
                t1 = time.perf_counter()
                print(f"{base},{M},{f},{K},{t1-t0:.6f},{gpu_ms:.3f}")
