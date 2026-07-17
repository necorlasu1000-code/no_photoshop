# -*- coding: utf-8 -*-
"""릴리스 패키지 제작 — EXE 빌드 → 배포 zip 스테이징.
사용: python make_release.py  →  release/NoPhotoshop_win.zip 생성
이후 업로드: gh release create v<버전> "release/NoPhotoshop_win.zip" -t "<제목>" -n "<노트>"
(자동 업데이터는 최신 릴리스의 .zip 에셋을 받아 앱 종료 시 교체한다)
"""
import os, re, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

ver = re.search(r'APP_VER = "ver ([\d.]+)"',
                open("deband_app.py", encoding="utf-8").read()).group(1)
print(f"버전: {ver}")

print("PyInstaller 빌드 …")
r = subprocess.run([sys.executable, "-m", "PyInstaller", "No photoshop.spec",
                    "--noconfirm"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)

os.makedirs("release", exist_ok=True)
zpath = os.path.join("release", "NoPhotoshop_win.zip")
if os.path.exists(zpath):
    os.remove(zpath)
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join("dist", "No photoshop.exe"), "No photoshop.exe")
    z.write("depth_extract.py", "depth_extract.py")
    for root, dirs, files in os.walk("vda"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            p = os.path.join(root, f)
            z.write(p, p)
print(f"완성: {zpath} ({os.path.getsize(zpath)//1048576}MB)")
print(f"업로드: gh release create v{ver} \"{zpath}\" -t \"No photoshop v{ver}\"")
