import os
import subprocess
import time
from PIL import Image

data_dir = "data"
out_dir = "out"
exec = "./imgproc"  

ms = [256, 512, 1024]    
ns = [3, 5, 7]           
filters = ["edge", "resize", "normalize"]
os.makedirs(out_dir, exist_ok=True)
def prepare_image(src_path, m, dst_path):
    img = Image.open(src_path).convert("RGB")
    img = img.resize((m, m), Image.BILINEAR)
    img.save(dst_path, format="JPEG")

def run_one(infile, outprefix, filt, n, resize_scale=None):
    cmd = [exec, infile, outprefix, filt, str(n)]
    if resize_scale:
        cmd.append(str(resize_scale))
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t1 = time.time()
    elapsed = t1 - t0
    stdout = proc.stdout.decode().strip()
    stderr = proc.stderr.decode().strip()
    return elapsed, stdout, stderr

def main():
    images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not images:
        print("No images found in data")
        return
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        for M in ms:
            tmp_input = os.path.join(out_dir, f"{base}_M{M}.jpg")
            prepare_image(img_path, M, tmp_input)
            for N in ns:
                for filt in filters:
                    outprefix = os.path.join(out_dir, f"{base}_M{M}")
                    elapsed, out, err = run_one(tmp_input, outprefix, filt, N)
                    safe_out = out.replace('"', "'")
                    print(f"{base},{M},{N},{filt},{elapsed:.6f},\"{safe_out}\"")
if __name__ == '__main__':
    main()
