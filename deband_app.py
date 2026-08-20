# -*- coding: utf-8 -*-
"""
No photoshop — 이미지 후처리 프로그램
탭 1: 밴딩 제거 / 탭 2: 실사 질감 / 탭 3: Remove GPT (AI 도트 제거)
- 탭 1·2: 드래그&드롭 첨부 → [▶ 변환 시작] → 완료 시 목록에서 제거, 자동 저장
- 탭 3: 이미지 1장 첨부 → 강도 선택(표준/강력) → 변환 → 전후 비교 슬라이더
        → [저장]을 눌러야 다른 이름으로 저장
- 외부 API/인터넷 불필요, 전부 로컬 연산
"""
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import queue
import threading
import time
import math
import traceback

import numpy as np
import cv2

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
VEXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
APP_TITLE = "no photoshop — 이미지 후처리"
APP_VER = "ver 1.8"

UPDATE_REPO = "necorlasu1000-code/no_photoshop"


def _ver_tuple(v):
    import re
    n = [int(x) for x in re.findall(r"\d+", str(v))[:3]]
    return tuple(n + [0] * (3 - len(n)))


def check_update_async(root):
    """사일런트 자동 업데이트 (EXE 실행일 때만): 시작 시 백그라운드로 최신
    GitHub 릴리스를 확인하고 zip을 받아두면, 교체 스크립트가 앱 종료를 기다렸다가
    조용히 파일을 교체한다 — 다음 실행부터 새 버전."""
    if not getattr(sys, "frozen", False):
        return

    def work():
        try:
            import json, urllib.request, tempfile, zipfile, re
            req = urllib.request.Request(
                f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest",
                headers={"User-Agent": "no-photoshop-updater"})
            with urllib.request.urlopen(req, timeout=10) as r:
                rel = json.load(r)
            tag = rel.get("tag_name", "")
            if _ver_tuple(tag) <= _ver_tuple(APP_VER):
                return
            asset = next((a for a in rel.get("assets", [])
                          if a.get("name", "").lower().endswith(".zip")), None)
            if not asset:
                return
            appdir = os.path.dirname(os.path.abspath(sys.argv[0]))
            tmp = tempfile.gettempdir()
            zpath = os.path.join(tmp, "nophotoshop_update.zip")
            urllib.request.urlretrieve(asset["browser_download_url"], zpath)
            updir = os.path.join(tmp, "nophotoshop_update")
            shutil.rmtree(updir, ignore_errors=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(updir)
            os.remove(zpath)
            if not os.path.exists(os.path.join(updir, "No photoshop.exe")):
                return                       # zip 구조 검증 실패 → 적용 안함
            bat = os.path.join(tmp, "nophotoshop_apply.bat")
            with open(bat, "w", encoding="utf-8") as f:
                f.write('@echo off\r\nchcp 65001 >nul\r\n'
                        ':wait\r\n'
                        'tasklist /FI "IMAGENAME eq No photoshop.exe" | find /I "No photoshop.exe" >nul\r\n'
                        'if not errorlevel 1 (timeout /t 2 /nobreak >nul & goto wait)\r\n'
                        f'xcopy /E /Y /Q "{updir}\\*" "{appdir}\\" >nul\r\n'
                        f'rmdir /S /Q "{updir}"\r\n'
                        'del "%~f0"\r\n')
            subprocess.Popen(["cmd", "/c", bat],
                             creationflags=0x08000008,   # DETACHED | NO_WINDOW
                             close_fds=True)
            try:
                root.after(0, lambda: root.title(
                    f"{APP_TITLE}   ({tag} 준비됨 — 껐다 켜면 적용)"))
            except Exception:
                pass
        except Exception:
            pass
    threading.Thread(target=work, daemon=True).start()


# v2 모노크롬(블랙&화이트) — 유색 액센트 없이 명도 대비로만 위계.
# "lime" 키는 호환을 위해 유지하되 값은 화이트 프라이머리.
COL = {
    "bg":     "#0b0b0c",
    "side":   "#111113",
    "card":   "#151517",
    "card2":  "#1e1e22",
    "border": "#26262a",
    "dash":   "#38383d",
    "lime":   "#fafafa",
    "lime2":  "#e6e6e9",
    "text":   "#f2f2f4",
    "sub":    "#a3a3ab",
    "dim":    "#68686f",
    "ok":     "#d9d9de",
    "err":    "#e07a6a",
}
FONT = "Malgun Gothic"
MONO = "Consolas"


def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def imread_u(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1].lower() or ".png"
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("인코딩 실패")
    buf.tofile(path)


# ---------------- 처리 1: 디밴딩 ----------------
def _deband_core(img, seed=12345, progress=None):
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    scales = [(6, 1.2, 4.0), (16, 1.5, 5.0), (40, 2.0, 6.0), (100, 2.5, 7.0)]
    for i, (sigma, t_low, t_high) in enumerate(scales):
        blur = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REPLICATE)
        diff = np.abs(f - blur).max(axis=2)
        k = max(3, int(sigma // 3) | 1)
        diff = cv2.dilate(diff, np.ones((k, k), np.uint8))
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=max(2, sigma / 4))
        t = np.clip((t_high - diff) / (t_high - t_low), 0.0, 1.0)
        wgt = (t * t * (3 - 2 * t))[..., None]
        f = f * (1 - wgt) + blur * wgt
        if progress:
            progress(int((i + 1) / (len(scales) + 1) * 100))
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    blur_m = cv2.GaussianBlur(f, (0, 0), sigmaX=12, borderType=cv2.BORDER_REPLICATE)
    dm = np.abs(f - blur_m).max(axis=2)
    dm = cv2.GaussianBlur(dm, (0, 0), sigmaX=6)
    flat = np.clip((6.0 - dm) / 6.0, 0.0, 1.0)[..., None]
    f = f + grain * (0.45 + 0.75 * flat)
    if progress:
        progress(100)
    return np.clip(np.round(f), 0, 255).astype(np.uint8)


def deband(img, progress=None):
    return _deband_core(img, seed=12345, progress=progress)


# ---------------- 처리 1b: 영상 디밴딩 ----------------
def get_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def deband_video(src, progress=None):
    """프레임별 디밴딩(프레임마다 그레인 시드 변경) → 10비트 HEVC 인코딩.
    오디오는 원본에서 복사."""
    ff = get_ffmpeg()
    if not ff:
        raise RuntimeError("FFmpeg을 찾을 수 없습니다")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    base, _ = os.path.splitext(src)
    dst = base + "_debanded.mp4"

    cmd = [ff, "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-",
           "-i", src,
           "-map", "0:v", "-map", "1:a?",
           "-c:a", "copy",
           "-c:v", "libx265", "-preset", "medium", "-crf", "16",
           "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
           dst]
    flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags)

    from multiprocessing.pool import ThreadPool
    workers = min(6, max(2, (os.cpu_count() or 4) - 2))

    def df(args):
        idx, fr = args
        return idx, _deband_core(fr, seed=12345 + idx)

    done = 0
    try:
        with ThreadPool(workers) as pool:
            idx = 0
            while True:
                batch = []
                while len(batch) < workers * 3:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    batch.append((idx, fr))
                    idx += 1
                if not batch:
                    break
                for _, out in pool.map(df, batch):
                    proc.stdin.write(out.tobytes())
                    done += 1
                if progress:
                    progress(min(99, int(done / n * 100)))
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg 인코딩 실패")
    if progress:
        progress(100)
    return dst


# ---------------- 처리 1c: 영상 스태킹 디밴딩 (진짜 10비트 복원) ----------------
def _deband_core_float(f):
    """엣지 게이트 다중스케일 스무딩 (float, 그레인/양자화 없음).
    스태킹이 못 채운 빠른 모션 영역의 잔여 밴딩 보강용."""
    scales = [(6, 1.2, 4.0), (16, 1.5, 5.0), (40, 2.0, 6.0), (100, 2.5, 7.0)]
    for sigma, t_low, t_high in scales:
        blur = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REPLICATE)
        diff = np.abs(f - blur).max(axis=2)
        k = max(3, int(sigma // 3) | 1)
        diff = cv2.dilate(diff, np.ones((k, k), np.uint8))
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=max(2, sigma / 4))
        t = np.clip((t_high - diff) / (t_high - t_low), 0.0, 1.0)
        wgt = (t * t * (3 - 2 * t))[..., None]           # smoothstep 게이트
        f = f * (1 - wgt) + blur * wgt
    return f


def stack_deband_video(src, progress=None, radius=12):
    """여러 프레임을 옵티컬 플로우로 정렬·모션마스킹 평균해 8비트에서 실제 ~10비트
    계조를 복원(천체사진 스태킹 원리) → 고주파 디테일 복원 → 잔여 디밴딩 →
    ProRes 422 HQ .mov(그레이딩 마스터) + HEVC .mp4(미리보기) 출력. 공간 디밴딩과
    달리 '가리기'가 아니라 실제 정보 복원. 영상 전용 — 프레임 3장 미만이면 폴백."""
    ff = get_ffmpeg()
    if not ff:
        raise RuntimeError("FFmpeg을 찾을 수 없습니다")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n < 3:
        return deband_video(src, progress=progress)   # 스태킹 불가 → 기존 방식
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    base, _ = os.path.splitext(src)
    dst = base + "_debanded.mov"          # 그레이딩 마스터 (ProRes 422 HQ, 4:2:2 10비트 인트라)
    prev = base + "_debanded.mp4"         # 가벼운 미리보기 (HEVC 10비트)
    flags = 0x08000000 if os.name == "nt" else 0

    def _enc(out, kind):
        if kind == "prores":
            vargs = ["-c:v", "prores_ks", "-profile:v", "3",
                     "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
        else:
            vargs = ["-c:v", "libx265", "-preset", "medium", "-crf", "18",
                     "-x265-params", "log-level=none",
                     "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"]
        cmd = [ff, "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr48le", "-s", f"{W}x{H}",
               "-r", f"{fps}", "-i", "-",
               "-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
               *vargs,
               "-color_primaries", "bt709", "-color_trc", "bt709",
               "-colorspace", "bt709", out]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=flags)
    proc = _enc(dst, "prores")
    proc_prev = _enc(prev, "hevc")

    SIGMA, DETAIL_SIGMA, DETAIL_W, DITHER = 6.0, 1.6, 0.6, 0.08
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    tls = threading.local()

    def get_dis():
        if not hasattr(tls, "dis"):
            tls.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return tls.dis

    def stack_one(i):
        dis = get_dis()
        cf = frames[i].astype(np.float32)
        cg = grays[i].astype(np.float32)
        acc = cf.copy()
        wsum = np.ones((H, W, 1), np.float32)
        for j in range(max(0, i - radius), min(n - 1, i + radius) + 1):
            if j == i:
                continue
            flow = dis.calc(grays[i], grays[j], None)
            mapx = gx + flow[..., 0]
            mapy = gy + flow[..., 1]
            warped = cv2.remap(frames[j], mapx, mapy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            res = np.abs(cv2.GaussianBlur(wg - cg, (5, 5), 0))
            w = np.exp(-(res / SIGMA) ** 2)[..., None]     # 모션 마스크
            acc += warped.astype(np.float32) * w
            wsum += w
        stacked = acc / wsum
        # 고주파(텍스처)는 원본에서 복원 — 밴딩은 저주파에만 삼
        low_s = cv2.GaussianBlur(stacked, (0, 0), DETAIL_SIGMA)
        low_c = cv2.GaussianBlur(cf, (0, 0), DETAIL_SIGMA)
        out = low_s + (cf - low_c) * DETAIL_W + (stacked - low_s) * (1.0 - DETAIL_W)
        out = _deband_core_float(out)                      # 잔여 밴딩 보강
        rng = np.random.default_rng(777 + i)
        d = (rng.random((H, W, 1), dtype=np.float32) -
             rng.random((H, W, 1), dtype=np.float32)) * DITHER   # TPDF 디더
        out = out + d
        return i, np.clip(out * 257.0 + 0.5, 0, 65535).astype(np.uint16)

    from multiprocessing.pool import ThreadPool
    workers = min(16, max(2, (os.cpu_count() or 4) - 2))
    done = 0
    try:
        with ThreadPool(workers) as pool:
            CH = workers * 2
            for s in range(0, n, CH):
                idxs = list(range(s, min(s + CH, n)))
                for i, out16 in pool.map(stack_one, idxs):
                    buf = out16.tobytes()
                    proc.stdin.write(buf)
                    proc_prev.stdin.write(buf)
                    done += 1
                if progress:
                    progress(min(99, int(done / n * 100)))
    finally:
        for p in (proc, proc_prev):
            try:
                p.stdin.close()
            except Exception:
                pass
        proc.wait(); proc_prev.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg 인코딩 실패")
    if progress:
        progress(100)
    return dst


# ---------------- 처리 1d: 깊이 기반 아웃포커스 (포커싱 탭) ----------------
DOF_K = 8        # 깊이 슬라이스 수
DOF_HBOOST = 2.5 # 하이라이트 보케 부스트 최대 강도 (hl 슬라이더로 스케일)
_DOF_DISC = {}   # 디스크 커널 캐시


def _disc_kernel(r, shape="disc"):
    """조리개 커널 — 실제 렌즈의 착란원. 4× 슈퍼샘플로 0.25px 분수 반지름 +
    안티앨리어스. shape: disc(원형) | hex(6날) | oct(8날) | ana(아나모픽 타원)."""
    q = max(1.0, round(float(r) * 4.0) / 4.0)
    key = (shape, q)
    k = _DOF_DISC.get(key)
    if k is None:
        ri = int(np.ceil(q))
        ss = 4
        n = (2 * ri + 1) * ss
        a = np.zeros((n, n), np.float32)
        c = ri * ss + ss // 2
        R = q * ss
        if shape == "ana":
            cv2.ellipse(a, (c, c), (int(round(R * 0.72)), int(round(R))),
                        0, 0, 360, 1.0, -1)
        elif shape in ("hex", "oct"):
            nb = 6 if shape == "hex" else 8
            rot = math.pi / nb                     # 날 살짝 회전
            pts = np.array([[c + R * math.cos(rot + 2 * math.pi * t / nb),
                             c + R * math.sin(rot + 2 * math.pi * t / nb)]
                            for t in range(nb)], np.int32)
            cv2.fillConvexPoly(a, pts, 1.0)
        else:
            cv2.circle(a, (c, c), int(round(R)), 1.0, -1)
        a = cv2.resize(a, (2 * ri + 1, 2 * ri + 1), interpolation=cv2.INTER_AREA)
        k = a / a.sum()
        _DOF_DISC[key] = k
    return k


def _dof_blur(x, r, fast=False, shape="disc"):
    """디스크(착란원) 블러 — 가우시안의 '안개'가 아니라 렌즈처럼 '겹쳐 번지는' 흐림.
    큰 반지름은 다운스케일+디스크+업스케일 근사. fast=프리뷰용(더 공격적 근사)."""
    r = float(r)
    if r < 0.5:
        return x
    if r < 1.5:
        return cv2.GaussianBlur(x, (0, 0), r * 0.7)
    direct = 3.0 if fast else 12.0
    if r <= direct:
        return cv2.filter2D(x, -1, _disc_kernel(r, shape), borderType=cv2.BORDER_REPLICATE)
    h, w = x.shape[:2]
    f = min(8, max(2, int(r / (3 if fast else 8))))
    sm = cv2.resize(x, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    sm = cv2.filter2D(sm, -1, _disc_kernel(r / f, shape), borderType=cv2.BORDER_REPLICATE)
    up = cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)
    if f >= 3:                       # 업스케일 격자(계단 픽셀) 제거
        up = cv2.GaussianBlur(up, (0, 0), f * 0.5)
    return up


def dof_dpack(depth01):
    """뎁스 입력 정규화 — refine-edge 팩 (dhard, dnear, dfar, matte) 그대로,
    단일 뎁스맵이면 매트 없는 팩으로 감싼다."""
    if isinstance(depth01, (tuple, list)):
        return depth01
    return (depth01, depth01, depth01, None)


def dof_slices(img01, depth01, hl=0.0):
    """깊이 슬라이스 분해 — 초점·조리개와 무관해서 프레임별 캐시 가능.
    depth01: 단일 뎁스맵 또는 refine-edge 팩 (dhard, dnear, dfar, matte 0..1).
    매트가 있으면 경계 밴드 픽셀(털 가닥 등)을 M 비율로 전경/배경 깊이에
    이중 소속시킨다 — 곡선 마스크가 아니라 refine-edge식 소프트 매팅.
    hl: 보케 하이라이트 0..1. 0이면 감마 공간 그대로 블러(색 100% 보존).
    반환: (슬라이스 리스트[(중심깊이, 대표깊이, 색*α, 부스트색*α, α)] far→near,
           경계존 맵, 작업공간 원본, 감마 플래그)."""
    dhard, dnear, dfar, M = dof_dpack(depth01)
    if hl > 0.0:
        lin = img01 * img01                              # 감마2 근사 리니어
        lum = lin.max(axis=2)
        hb = (1.0 + DOF_HBOOST * hl *
              np.clip((lum - 0.55) / 0.45, 0.0, 1.0) ** 2)[..., None]
        linb = lin * hb
        gam = True
    else:
        lin = img01                                      # 색 보존: 표시 공간 그대로
        linb = lin
        gam = False
    dn = np.clip(dnear.astype(np.float32), 0.5 / DOF_K, 1 - 0.5 / DOF_K)
    if M is None:
        df, Mw, Mi = dn, None, None
    else:
        df = np.clip(dfar.astype(np.float32), 0.5 / DOF_K, 1 - 0.5 / DOF_K)
        Mw = M.astype(np.float32)
        # 매트 리맵: 거의 순수한 배경 픽셀(Mi>0.7)만 배경 소속을 유지하고 혼합
        # 링은 피사체를 따라가게 — 후경 초점 시 혼합색의 배경 몫이 선명하게
        # 렌더되어 남던 '원본 외곽 잔상'이 피사체와 함께 뭉개진다 (test4 검증).
        _mi = 1.0 - Mw
        Mw = 1.0 - _mi * np.clip((_mi - 0.4) / 0.3, 0.0, 1.0)
        Mi = 1.0 - Mw
    # 적응형 슬라이스: 중심을 균일 격자가 아닌 '콘텐츠 깊이 분위수'에 배치 —
    # 피사체가 몰린 깊이 구간에 슬라이스가 집중되어 몸 안 부위(머리/팔/해머)가
    # 서로 다른 슬라이스로 분리, 실제 렌즈처럼 초점면이 공간을 훑는 느낌이 생긴다.
    cs = np.quantile(dn, np.linspace(0.001, 0.999, DOF_K)).astype(np.float64)
    for _i in range(1, DOF_K):
        cs[_i] = max(cs[_i], cs[_i - 1] + 0.015)
    cs = np.clip(cs, 0.0, 1.0)
    _gL = [1e9] + [max(cs[_i] - cs[_i - 1], 1e-6) for _i in range(1, DOF_K)]
    _gR = [max(cs[_i + 1] - cs[_i], 1e-6) for _i in range(DOF_K - 1)] + [1e9]

    def _w(dv, k):
        lo = np.clip((cs[k] - dv) / _gL[k], 0.0, None)
        hi = np.clip((dv - cs[k]) / _gR[k], 0.0, None)
        return np.clip(1.0 - np.maximum(lo, hi), 0.0, 1.0)
    out = []
    for k in range(DOF_K):
        c = float(cs[k])
        if M is None:
            m = _w(dn, k)
            num = None
        else:
            m1 = Mw * _w(dn, k)
            m2 = Mi * _w(df, k)
            m = m1 + m2
            num = m1 * dn + m2 * df
        ms = float(m.sum())
        if ms < 1e-3:
            continue
        # 블러 반지름용 대표 깊이 = 내용물의 실제 평균 깊이 (슬라이스 중심이 아니라).
        # 중심값을 쓰면 초점이 각 슬라이스 중심을 지날 때마다 배경이 번갈아
        # '완전 선명'으로 튀어 엣지 밝기가 맥동(깜빡임)한다.
        dm = float(((m * dn) if num is None else num).sum() / ms)
        m3 = m[..., None]
        pm = lin * m3
        out.append((c, dm, pm, pm if not gam else linb * m3, m))
    k5 = np.ones((5, 5), np.uint8)
    e = cv2.dilate(dhard, k5) - cv2.erode(dhard, k5)
    edge = cv2.GaussianBlur(np.clip((e - 0.05) / 0.10, 0.0, 1.0), (0, 0), 1.5)
    return out, edge, lin, gam


def dof_apply(img01, depth01, focus, aperture_px, slices=None, fast=False, hl=0.0,
              shape="disc", ca=0.0, breath=0.0):
    """뎁스 기반 피사계 심도 — 레이어드 합성(Nuke ZDefocus 계열) + 디스크 보케.
    2패스: ①초점 이하(원경~초점면) 슬라이스 합성 → 초점면 원본 복원(경계존 보호)
    → ②전경 슬라이스를 그 위에 오버 합성. 복원이 전경 스필보다 먼저라 흐린
    전경이 선명한 배경 위로 온전히 반투명하게 번짐(스필 감쇠 버그 없음).
    hl=0이면 색 100% 보존, >0이면 리니어+하이라이트 부스트(보케 볼).
    img01: float32 HxWx3 (0-1, BGR) · depth01: 0-1 (1=가까움) · aperture_px: 최대 반지름."""
    if aperture_px < 0.6:
        return img01
    if slices is None:
        slices = dof_slices(img01, depth01, hl)
    sl, edge, lin, gam = slices

    def blur_slice(dm, pm, pmb, m):
        s = abs(dm - focus) * aperture_px                # 대표 깊이 기반 — 데드존 없음
        if s < 0.5:
            return pm, m
        t = min(1.0, max(0.0, (s - 1.0) / 3.0))          # 흐린 슬라이스만 부스트 (연속 램프)
        src = pm if (not gam or t <= 0) else (pmb if t >= 1 else pm + (pmb - pm) * t)
        if ca > 0.0 and s >= 2.0:
            # 축상 색수차: 후경 보케=그린이 크게, 전경 보케=마젠타(R/B)가 크게
            k = 0.10 * ca
            if dm < focus:
                rG, rRB = s * (1.0 + k), s * (1.0 - 0.5 * k)
            else:
                rG, rRB = s * (1.0 - 0.5 * k), s * (1.0 + k)
            b_, g_, r_ = cv2.split(src)
            bpm = cv2.merge([_dof_blur(b_, rRB, fast, shape),
                             _dof_blur(g_, rG, fast, shape),
                             _dof_blur(r_, rRB, fast, shape)])
            return bpm, _dof_blur(m, s, fast, shape)
        return _dof_blur(src, s, fast, shape), _dof_blur(m, s, fast, shape)

    acc = cov = accN = covN = None
    for c, dm, pm, pmb, m in sl:
        # 소프트 패스 분할: 초점보다 한 슬라이스 이상 가까우면 전경(nw=1),
        # 경계 슬라이스는 비율로 양쪽에 — 초점 이동 시 합성 방식이 뚝 안 바뀜
        nw = min(1.0, max(0.0, (dm - focus) * DOF_K - 0.5))
        bpm, ba = blur_slice(dm, pm, pmb, m)
        if nw < 1.0:                 # ① 원경~초점면: 오버 합성 후 정규화
            f = 1.0 - nw
            bp, bb = (bpm, ba) if nw <= 0.0 else (bpm * f, ba * f)
            if acc is None:
                acc, cov = bp.copy(), bb.copy()
            else:
                ia = 1.0 - bb
                np.multiply(acc, ia[..., None], out=acc)
                np.add(acc, bp, out=acc)
                np.multiply(cov, ia, out=cov)
                np.add(cov, bb, out=cov)
        if nw > 0.0:                 # ② 전경: 가산 파티션 (내부 색 보존)
            bp, bb = (bpm, ba) if nw >= 1.0 else (bpm * nw, ba * nw)
            if accN is None:
                accN, covN = bp.copy(), bb.copy()
            else:
                np.add(accN, bp, out=accN)
                np.add(covN, bb, out=covN)
    # 배경(패스1)을 '완전 커버리지'로 만든 뒤 전경을 얹는다. 커버리지 구멍을
    # 나눗셈으로 때우면 그 나눗셈이 증폭기가 되어 초점 이동 시 밝기 맥동(깜빡임)이
    # 생기므로, 구멍(전경에 가려진 배경)은 주변 배경색 외삽으로 채운다.
    if acc is None:
        acc = lin.copy()
    else:
        colorB = acc / np.maximum(cov, 1e-6)[..., None]
        np.clip(colorB, 0.0, 1.0, out=colorB)
        covB = np.clip(cov * 1.3, 0.0, 1.0)   # 인접 슬라이스 오버합성 결손(≈0.77) 보정
        acc = colorB * covB[..., None]
        r_fill = max(3.0, aperture_px * 0.5)
        fill = (_dof_blur(acc, r_fill, True) /
                np.maximum(_dof_blur(covB, r_fill, True), 1e-4)[..., None])
        np.clip(fill, 0.0, 1.0, out=fill)
        acc += fill * (1.0 - covB)[..., None]  # 가려진 배경 = 주변 배경 외삽
    dh_, dnr_, dfr_, M_ = dof_dpack(depth01)
    if M_ is None:
        # 단일 뎁스: 경계 램프의 중간 깊이가 초점으로 오인될 수 있어 경계존 가드 필요
        wd = np.clip(1.0 - np.abs(dnr_ - focus) * aperture_px / 1.6, 0.0, 1.0) * (1.0 - edge)
    else:
        # 매트 팩: 근경(피사체) 쪽 초점 근접도 × M 만으로 복원. 초점 맞은 피사체
        # 실루엣은 (시간적으로 안정적인) 원본으로 고정되어 경계 깜빡임이 없고,
        # 원경이 초점일 때는 복원하지 않는다 — 원본에는 선명한 피사체 엣지가 섞여
        # 있어서 원경 가중으로 복원하면 흐린 피사체 위에 실루엣 선이 재주입된다.
        # (원경의 선명함은 패스1 합성이 이미 올바르게 렌더함)
        wd = M_ * np.clip(1.0 - np.abs(dnr_ - focus) * aperture_px / 1.6, 0.0, 1.0)
    w = wd[..., None]                # 초점면 원본 복원
    np.multiply(acc, 1.0 - w, out=acc)
    acc += lin * w
    if accN is not None:             # 전경을 복원된 배경 위로: out = FG + BG*(1-α)
        sc = np.maximum(covN, 1.0)   # 겹친 스필만 눌러줌 (파티션 합=1은 그대로)
        np.divide(accN, sc[..., None], out=accN)
        np.minimum(covN, 1.0, out=covN)
        np.multiply(acc, (1.0 - covN)[..., None], out=acc)
        np.add(acc, accN, out=acc)
    np.clip(acc, 0.0, 1.0, out=acc)
    if gam:
        np.sqrt(acc, out=acc)
    # 오클루전 엣지 일관성 패스: 실루엣 밴드의 선명도 상한 = 피사체(근경) 블러.
    # 매트 밖까지 새어 있는 피사체 색(AA 헤일로·림 글로우)이 배경 초점에서
    # 선명한 외곽선으로 남는 것을, 색 추정 없이 밴드 재블러로 봉쇄한다.
    # (밴드·게이트 모두 넓게 스무딩 — 급한 전이는 그 자체가 새 경계를 만든다)
    if M_ is not None:
        tz = np.maximum(((M_ > 0.02) & (M_ < 0.98)).astype(np.float32), edge)
        band = np.clip(cv2.GaussianBlur(tz, (0, 0), 6.0) * 1.6, 0.0, 0.9)
        rr = np.abs(dnr_ - focus) * aperture_px
        rgate = cv2.GaussianBlur(np.clip((rr - 1.0) / 3.0, 0.0, 1.0), (0, 0), 5.0)
        w3 = band * rgate
        if float(w3.max()) > 0.05:
            rmed = float(np.median(rr[w3 > 0.05]))
            if rmed >= 1.0:
                bl = _dof_blur(acc, rmed, fast)
                w3 = w3[..., None]
                acc = acc * (1.0 - w3) + bl * w3
    if breath > 0.0:
        # 포커스 브리딩: 근접 초점일수록 미세 줌 (최대 ±1.5%) — 렌즈 손맛
        sc = 1.0 + 0.015 * breath * (focus - 0.5) * 2.0
        h_, w_ = acc.shape[:2]
        Mw_ = cv2.getRotationMatrix2D((w_ / 2.0, h_ / 2.0), 0.0, sc)
        acc = cv2.warpAffine(acc, Mw_, (w_, h_), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return acc


def focus_render_video(frames, depth16, focus_curve, aperture_frac, fps, src,
                       progress=None, hl=0.0, shape="disc", ca=0.0, breath=0.0):
    """녹화된 포커스 커브(프레임별 초점 뎁스)로 전체 영상에 DoF를 구워
    ProRes .mov(마스터) + HEVC .mp4(미리보기) 출력."""
    ff = get_ffmpeg()
    if not ff:
        raise RuntimeError("FFmpeg을 찾을 수 없습니다")
    H, W = frames[0].shape[:2]
    N = len(frames)
    base, _ = os.path.splitext(src)
    dst = base + "_focus.mov"
    prev = base + "_focus.mp4"
    flags = 0x08000000 if os.name == "nt" else 0

    def _enc(out, kind):
        if kind == "prores":
            vargs = ["-c:v", "prores_ks", "-profile:v", "3",
                     "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
        else:
            vargs = ["-c:v", "libx265", "-preset", "medium", "-crf", "18",
                     "-x265-params", "log-level=none", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"]
        cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr48le", "-s", f"{W}x{H}",
               "-r", f"{fps}", "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
               *vargs, "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, creationflags=flags)
    p = _enc(dst, "prores"); pp = _enc(prev, "hevc")
    d16, dn16, df16, mt8 = depth16 if isinstance(depth16, tuple) else (depth16, None, None, None)
    ap_arr = hasattr(aperture_frac, "__len__")

    def _one(i):
        img = frames[i].astype(np.float32) / 255.0
        if mt8 is None:
            dep = d16[i].astype(np.float32) / 65535.0
        else:
            dep = (d16[i].astype(np.float32) / 65535.0,
                   dn16[i].astype(np.float32) / 65535.0,
                   df16[i].astype(np.float32) / 65535.0,
                   mt8[i].astype(np.float32) / 255.0)
        af = float(aperture_frac[i]) if ap_arr else float(aperture_frac)
        out = dof_apply(img, dep, float(focus_curve[i]), af * W, hl=hl,
                        shape=shape, ca=ca, breath=breath)
        return np.clip(out * 65535.0 + 0.5, 0, 65535).astype(np.uint16).tobytes()

    from multiprocessing.pool import ThreadPool
    workers = min(8, max(2, (os.cpu_count() or 4) - 2))
    try:
        done = 0
        with ThreadPool(workers) as pool:
            CH = workers * 2
            for s0 in range(0, N, CH):
                for buf in pool.map(_one, range(s0, min(s0 + CH, N))):
                    p.stdin.write(buf); pp.stdin.write(buf)
                    done += 1
                if progress:
                    progress(min(99, int(done / N * 100)))
    finally:
        for q in (p, pp):
            try:
                q.stdin.close()
            except Exception:
                pass
        p.wait(); pp.wait()
    if p.returncode != 0:
        raise RuntimeError("FFmpeg 인코딩 실패")
    if progress:
        progress(100)
    return dst


# ---------------- 처리 2: 실사 질감 (효과별 강도 조절) ----------------
PR_DEFAULTS = {"lift": 1.0, "rolloff": 1.0, "bloom": 1.0,
               "ca": 1.0, "vignette": 1.0, "grain": 1.0}


def photoreal_params(img, p=None, progress=None):
    """효과별 강도(0.0=끔, 1.0=기본, 2.0=2배)를 받는 실사 질감 파이프라인."""
    if p is None:
        p = PR_DEFAULTS
    def g(k):
        return float(p.get(k, 1.0))
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    sc = max(0.5, w / 2752.0)

    # 1) 블랙 리프트
    s = g("lift")
    if s > 0:
        lift = 0.022 * s
        f = lift + f * (1.0 - lift)

    # 2) 하이라이트 롤오프 (+미드톤 S커브)
    s = g("rolloff")
    if s > 0:
        knee = 0.78 - 0.06 * max(0.0, s - 1.0)
        hi = f > knee
        rolled = np.where(hi, knee + (1 - knee) * np.tanh((f - knee) / (1 - knee)), f)
        f = np.clip(f + (rolled - f) * min(s, 1.0), 0, 1)
        f = np.clip(f + 0.06 * min(s, 2.0) * np.sin((f - 0.5) * np.pi) * f * (1 - f), 0, 1)
    if progress:
        progress(25)

    # 3) 렌즈 블룸
    s = g("bloom")
    if s > 0:
        luma = f @ np.float32([0.114, 0.587, 0.299])
        bright = np.clip((luma - 0.72) / 0.28, 0, 1) ** 1.5
        bloom_src = f * bright[..., None]
        bloom = (cv2.GaussianBlur(bloom_src, (0, 0), 18 * sc) * 0.5 +
                 cv2.GaussianBlur(bloom_src, (0, 0), 60 * sc) * 0.5)
        f = 1 - (1 - f) * (1 - np.clip(bloom * 0.45 * s, 0, 1))
    if progress:
        progress(55)

    # 4) 색수차
    s = g("ca")
    if s > 0:
        def radial_scale(ch, sv):
            M = np.float32([[sv, 0, (1 - sv) * w / 2], [0, sv, (1 - sv) * h / 2]])
            return cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        shift = 0.0012 * s
        b, gg, r = cv2.split(f)
        r = radial_scale(r, 1.0 + shift)
        b = radial_scale(b, 1.0 - shift)
        f = cv2.merge([b, gg, r])
    if progress:
        progress(70)

    # 5) 비네팅
    s = g("vignette")
    if s > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
        vig = 1.0 - (0.16 * s) * np.clip((nx * nx + ny * ny) / 2.0, 0, 1) ** 1.4
        f = f * np.clip(vig, 0, 1)[..., None]
    if progress:
        progress(85)

    # 6) 필름 그레인
    s = g("grain")
    if s > 0:
        rng = np.random.default_rng(7)
        n_luma = rng.normal(0, 1, (h, w, 1)).astype(np.float32)
        n_chroma = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
        grain = cv2.GaussianBlur(n_luma * 0.9 + n_chroma * 0.1, (0, 0),
                                 0.6 * max(1.0, sc)) * 1.8
        luma2 = np.clip(f @ np.float32([0.114, 0.587, 0.299]), 0, 1)
        f = f + grain * s * (0.012 + 0.020 * (1 - luma2) ** 1.2)[..., None]
    if progress:
        progress(100)
    return np.clip(np.round(f * 255), 0, 255).astype(np.uint8)


def photoreal(img, progress=None):
    return photoreal_params(img, PR_DEFAULTS, progress=progress)


# ---------------- 처리 3: Remove GPT (AI 도트 제거) ----------------
def _dot_mask(f, k, t, max_area, max_dim, protect_warm):
    """top-hat 기반 점 마스크와 배경(오프닝) 반환."""
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    luma = f @ np.float32([0.114, 0.587, 0.299])
    bg = cv2.morphologyEx(f, cv2.MORPH_OPEN, kern)
    tophat = luma - (bg @ np.float32([0.114, 0.587, 0.299]))
    m = (tophat > t).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if protect_warm:
        flat = labels.ravel()
        bsum = np.bincount(flat, weights=f[..., 0].ravel(), minlength=n)
        rsum = np.bincount(flat, weights=f[..., 2].ravel(), minlength=n)
        cnt = np.bincount(flat, minlength=n).astype(np.float64)
        cnt[cnt == 0] = 1
        mean_b, mean_r = bsum / cnt, rsum / cnt

    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area > max_area or max(bw, bh) > max_dim:
            continue
        if protect_warm and mean_r[i] > mean_b[i] + 12:
            continue
        keep[i] = 1
    return keep[labels], bg, m


def remove_gpt(img, strong=False, progress=None):
    """AI 도트(스티플) 제거.
    표준: 따뜻한 색(금색 디테일·불빛) 보호, 점 밀집 면만 융합
    강력: 모든 점 무차별 제거 + 엣지 보호 융합"""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    density_acc = np.zeros((h, w), np.float32)

    if strong:
        iters, passes = 3, [(5, 5, 260, 18), (9, 7, 260, 18), (13, 9, 260, 18)]
        protect = False
    else:
        iters, passes = 1, [(5, 6, 40, 8), (9, 10, 80, 11)]
        protect = True

    step = 0
    total_steps = iters * len(passes) + 1
    for it in range(iters):
        for k, t, ma, md in passes:
            dots, bg, raw = _dot_mask(f, k, t, ma, md, protect)
            if it == 0:
                density_acc += raw.astype(np.float32)
            a = cv2.GaussianBlur(
                cv2.dilate(dots, np.ones((3, 3), np.uint8)).astype(np.float32),
                (0, 0), 1.0)[..., None]
            f = f * (1 - a) + bg * a
            step += 1
            if progress:
                progress(int(step / total_steps * 100))

    dens = cv2.GaussianBlur(density_acc, (0, 0), 9)
    dens = np.clip((dens - 0.2 if strong else dens - 0.25) / 0.6, 0, 1)
    u8 = np.clip(np.round(f), 0, 255).astype(np.uint8)
    med = cv2.medianBlur(u8, 3).astype(np.float32)
    if strong:
        gray = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3)
        edge = np.clip(edge / 120.0, 0, 1)
        alpha = (dens * (1 - edge) * 0.8)[..., None]
    else:
        alpha = (dens * 0.7)[..., None]
    f = f * (1 - alpha) + med * alpha
    if progress:
        progress(100)
    return np.clip(np.round(f), 0, 255).astype(np.uint8)


# ---------------- 처리 4: AI 우글거림(sizzle) 수정 ----------------
SIZZLE_BIAS = {"bubble": (0.0, -1.1), "dust": (0.0, 0.0),
               "snow": (0.0, 0.9), "ember": (0.0, -0.7)}


def fix_sizzle_video(src, ptype="bubble", k_blend=0.5, k_blur=0.4,
                     k_dens=1.0, k_grain=0.8, progress=None):
    """모션보상 잔차로 '끓는 영역'을 탐지해 시간 블렌딩 + 연속 모션블러 +
    장면 적응형 파티클 레이어 + 그레인. 슬라이딩 윈도우 스트리밍(메모리 안전)."""
    ff = get_ffmpeg()
    if not ff:
        raise RuntimeError("FFmpeg을 찾을 수 없습니다")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_est = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    base, _ = os.path.splitext(src)
    dst = base + "_particlefix.mp4"

    S = 0.25
    sw, sh = max(8, int(W * S)), max(8, int(H * S))
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32),
                         np.arange(sh, dtype=np.float32))
    FX, FY = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    rng = np.random.default_rng(99)

    # auto 타입: 앞부분 프레임으로 바이어스 추정
    if ptype == "auto":
        g_prev, acc, wsum = None, np.zeros(2), 1e-6
        for _ in range(25):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
            if g_prev is not None:
                fl = cv2.calcOpticalFlowFarneback(g_prev, g, None,
                                                  0.5, 3, 21, 3, 5, 1.2, 0)
                mapx, mapy = gx - fl[..., 0], gy - fl[..., 1]
                wp = cv2.remap(g_prev, mapx, mapy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
                m = np.clip((cv2.GaussianBlur(
                    np.abs(g.astype(np.float32) - wp.astype(np.float32)),
                    (0, 0), 4) - 6.0) / 16.0, 0, 1)
                ws = float(m.sum()) + 1e-6
                acc += np.array([float((fl[..., 0] * m).sum()),
                                 float((fl[..., 1] * m).sum())]) / ws
                wsum += 1
            g_prev = g
        v = acc / wsum
        BIAS = (float(v[0]) * 0.3 / S, float(v[1]) * 0.3 / S - 0.3)
        cap.release()
        cap = cv2.VideoCapture(src)
    else:
        BIAS = SIZZLE_BIAS.get(ptype, (0.0, -1.1))

    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{fps}", "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?",
           "-c:a", "copy", "-c:v", "libx265", "-preset", "medium", "-crf", "16",
           "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", dst]
    flags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags)

    frames, grays, flows, raw_sz = {}, {}, {}, {}
    read_idx = [0]
    eof = [False]
    n_total = [None]

    def read_until(idx):
        while not eof[0] and read_idx[0] <= idx:
            ok, fr = cap.read()
            j = read_idx[0]
            if not ok:
                eof[0] = True
                n_total[0] = j
                break
            frames[j] = fr
            grays[j] = cv2.cvtColor(cv2.resize(fr, (sw, sh),
                                               interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2GRAY)
            if j >= 1:
                flows[j - 1] = cv2.calcOpticalFlowFarneback(
                    grays[j - 1], grays[j], None, 0.5, 3, 21, 3, 5, 1.2, 0)
                fl = flows[j - 1]
                wp = cv2.remap(grays[j - 1], gx - fl[..., 0], gy - fl[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                raw_sz[j] = cv2.GaussianBlur(
                    np.abs(grays[j].astype(np.float32) - wp.astype(np.float32)),
                    (0, 0), 4)
            read_idx[0] += 1

    def sizzle_at(i):
        last = (n_total[0] - 1) if n_total[0] else i + 2
        acc = None
        for j in range(i - 2, i + 3):
            jj = max(1, min(last, j))
            r = raw_sz.get(jj)
            if r is None:
                r = raw_sz.get(max(raw_sz) if raw_sz else 1,
                               np.zeros((sh, sw), np.float32))
            acc = r.copy() if acc is None else acc + r
        return np.clip((acc / 5.0 - 6.0) / 16.0, 0, 1)

    def full_flow(i):
        last_f = max(flows) if flows else 0
        fl = flows.get(min(i, last_f), np.zeros((sh, sw, 2), np.float32))
        return cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR) / S

    particles = []
    TARGET = int(260 * k_dens)

    def spawn(i, mask_s, frame, k):
        p = mask_s.ravel().astype(np.float64)
        tot = p.sum()
        if tot < 1e-3:
            return
        p /= tot
        idx = rng.choice(len(p), size=k, p=p)
        ys, xs = np.unravel_index(idx, mask_s.shape)
        last_f = max(flows) if flows else 0
        fl = flows.get(min(i, last_f), np.zeros((sh, sw, 2), np.float32))
        for y, x in zip(ys, xs):
            px = min(W - 2.0, max(1.0, (x + rng.normal(0, 1.5)) / S))
            py = min(H - 2.0, max(1.0, (y + rng.normal(0, 1.5)) / S))
            x0, y0 = int(max(0, px - 10)), int(max(0, py - 10))
            patch = frame[y0:y0 + 21, x0:x0 + 21].reshape(-1, 3)
            col = np.clip(np.percentile(patch, 92, axis=0).astype(np.float32)
                          * 1.15 + 18, 0, 255)
            v = fl[int(y), int(x)] / S * 0.55
            particles.append({
                "pos": np.array([px, py], np.float32),
                "prev": np.array([px, py], np.float32),
                "vel": np.array([v[0] + BIAS[0] + rng.normal(0, 0.5),
                                 v[1] + BIAS[1] + rng.normal(0, 0.5)], np.float32),
                "size": float(np.clip(rng.lognormal(0.45, 0.55), 0.8, 7.0)),
                "age": 0, "max_age": int(rng.uniform(20, 70)),
                "bright": float(rng.uniform(0.35, 0.95)), "col": col,
                "phase": float(rng.uniform(0, 6.28)),
                "freq": float(rng.uniform(0.25, 0.6)),
                "blur": float(rng.uniform(0, 1) ** 2 * 2.5)})

    def render_particles(overlay):
        for pt in particles:
            env = min(1.0, pt["age"] / 5.0, (pt["max_age"] - pt["age"]) / 8.0)
            if env <= 0:
                continue
            a = pt["bright"] * env
            r = pt["size"] * (1.0 + pt["blur"])
            dist = float(np.linalg.norm(pt["pos"] - pt["prev"]))
            ksm = int(np.clip(dist / max(1.0, r * 0.7) + 1, 1, 8))
            for t in np.linspace(0.0, 1.0, ksm):
                p = pt["prev"] * (1 - float(t)) + pt["pos"] * float(t)
                x, y = float(p[0]), float(p[1])
                R = int(max(2, r * 3))
                x0, y0, x1, y1 = int(x) - R, int(y) - R, int(x) + R + 1, int(y) + R + 1
                if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                    continue
                cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
                d2 = ((xx - x) ** 2 + (yy - y) ** 2) / (r * r + 1e-6)
                g = np.exp(-d2 * 1.6) * (a / ksm / (1.0 + pt["blur"]))
                overlay[cy0:cy1, cx0:cx1] += g[..., None] * (pt["col"] / 255.0)

    taps = np.linspace(-0.5, 0.5, 9)
    gw_t = np.exp(-(taps ** 2) / (2 * 0.25 ** 2))
    gw_t /= gw_t.sum()

    i = 0
    try:
        while True:
            read_until(i + 2)
            if n_total[0] is not None and i >= n_total[0]:
                break
            if i not in frames:
                break
            cur = frames[i].astype(np.float32)
            M_s = sizzle_at(i)
            M = cv2.resize(M_s, (W, H), interpolation=cv2.INTER_LINEAR)
            last = n_total[0] - 1 if n_total[0] else None

            if k_blend > 0 and i > 0 and (last is None or i < last) and (i + 1) in frames:
                fl_p = full_flow(i - 1)
                wp = cv2.remap(frames[i - 1].astype(np.float32),
                               FX - fl_p[..., 0], FY - fl_p[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                fl_n = full_flow(i)
                wn = cv2.remap(frames[i + 1].astype(np.float32),
                               FX + fl_n[..., 0], FY + fl_n[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                a = (np.clip(M, 0, 1) * 0.45 * k_blend)[..., None]
                cur = cur * (1 - a) + (wp + wn) * 0.5 * a

            if k_blur > 0:
                fl = full_flow(i)
                speed = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
                scale = np.minimum(1.0, 10.0 / np.maximum(speed, 1e-3)).astype(np.float32)
                fcx = (fl[..., 0] * scale).astype(np.float32)
                fcy = (fl[..., 1] * scale).astype(np.float32)
                wgt = np.clip((speed - 1.5) / 10.0, 0, 1) * 0.8 * k_blur
                if wgt.max() > 0.02:
                    acc = np.zeros_like(cur)
                    for t, sw_ in zip(taps, gw_t):
                        t, sw_ = float(t), float(sw_)
                        if abs(t) < 1e-6:
                            acc += cur * sw_
                        else:
                            acc += cv2.remap(cur,
                                             (FX + fcx * t).astype(np.float32),
                                             (FY + fcy * t).astype(np.float32),
                                             cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE) * sw_
                    cur = cur * (1 - wgt[..., None]) + acc * wgt[..., None]

            if k_dens > 0:
                if len(particles) < TARGET:
                    spawn(i, M_s, frames[i], min(40, TARGET - len(particles)))
                fl = full_flow(i)
                keep = []
                for pt in particles:
                    pt["prev"] = pt["pos"].copy()
                    x = int(np.clip(pt["pos"][0], 0, W - 1))
                    y = int(np.clip(pt["pos"][1], 0, H - 1))
                    local = fl[y, x] * 0.25
                    perp = np.array([-pt["vel"][1], pt["vel"][0]], np.float32)
                    nv = np.linalg.norm(perp) + 1e-6
                    wob = perp / nv * np.sin(pt["age"] * pt["freq"] + pt["phase"]) * 0.35
                    pt["vel"] = (pt["vel"] * 0.96
                                 + np.array(BIAS, np.float32) * 0.05
                                 + local * 0.1 + wob)
                    pt["pos"] += pt["vel"]
                    pt["age"] += 1
                    if (pt["age"] < pt["max_age"] and -20 < pt["pos"][0] < W + 20
                            and -20 < pt["pos"][1] < H + 20):
                        keep.append(pt)
                particles[:] = keep
                overlay = np.zeros((H, W, 3), np.float32)
                render_particles(overlay)
                cur01 = cur / 255.0
                cur = (1 - (1 - cur01) * (1 - np.clip(overlay, 0, 1))) * 255.0

            if k_grain > 0:
                g = rng.normal(0, 1, (H, W, 1)).astype(np.float32)
                cur = cur + g * (M[..., None] * 1.6 * k_grain)

            proc.stdin.write(np.clip(np.round(cur), 0, 255).astype(np.uint8).tobytes())

            # 윈도우 청소
            for d in (frames, grays):
                d.pop(i - 2, None)
            flows.pop(i - 3, None)
            raw_sz.pop(i - 3, None)

            if progress and i % 3 == 0:
                progress(min(99, int(i / max(1, n_est) * 100)))
            i += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg 인코딩 실패")
    if progress:
        progress(100)
    return dst


# ---------------- 처리 5: 디테일 보강 (AI 풀샷 품질 복원) ----------------
def detail_boost_video(src, k_detail=1.0, k_lock=1.0, k_blur=1.0,
                       k_smooth=1.0, k_grain=1.0, progress=None):
    """① 카메라 경로 스무딩(바닥 기준 측정 + 관성 클램프 — 속도 출렁임 제거)
       ② 텍스처 게이트 디테일 인핸스(매끈한 하늘/구름/안개는 자동 보호)
       ③ 텍스처 락킹(고주파를 광학 흐름 따라 이월 — 디테일 스위밍 억제)
       ④ 180도 셔터 모션블러(끊김 완화) ⑤ 루마 가중 그레인.
       2패스 스트리밍(메모리 안전), 10-bit HEVC 저장."""
    ff = get_ffmpeg()
    if not ff:
        raise RuntimeError("FFmpeg을 찾을 수 없습니다")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_est = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    base, _ = os.path.splitext(src)
    dst = base + "_detail.mp4"

    CLARITY = [(2.0, 0.55 * k_detail), (6.0, 0.45 * k_detail),
               (16.0, 0.30 * k_detail)]
    SHARP = 0.9 * k_detail
    LOCK = min(0.85, 0.62 * k_lock)
    LOCK_SIG = 2.5
    SHUTTER = 0.5 * k_blur
    TAPS = 9
    GRAIN_AMP = 2.2 * k_grain

    # ---- 패스 1: 카메라 경로 분석 ----
    corr, zoom = None, 1.0
    if k_smooth > 0.02:
        raw, prev_gb, winb, i = [], None, None, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None,
                           fx=0.5, fy=0.5).astype(np.float32)
            gb = np.ascontiguousarray(g[int(g.shape[0] * 0.60):, :])
            if prev_gb is not None:
                if winb is None:
                    winb = cv2.createHanningWindow((gb.shape[1], gb.shape[0]),
                                                   cv2.CV_32F)
                (dx, dy), resp = cv2.phaseCorrelate(prev_gb, gb, winb)
                raw.append([dx * 2, dy * 2, resp])
            prev_gb = gb
            i += 1
            if progress and i % 10 == 0:
                progress(min(18, int(i / max(1, n_est) * 18)))
        if len(raw) >= 8:
            raw = np.array(raw, np.float64)
            ACC = 0.35           # 관성: 프레임당 허용 가속(px/f^2)
            v = np.zeros((len(raw), 2))
            cur_v = raw[0, :2].copy()
            for j in range(len(raw)):
                trust = 1.0 if raw[j, 2] > 0.25 else 0.0
                cur_v = cur_v + np.clip((raw[j, :2] - cur_v) * trust, -ACC, ACC)
                v[j] = cur_v
            P_raw = np.vstack([np.zeros((1, 2)), np.cumsum(raw[:, :2], axis=0)])
            P_des = np.vstack([np.zeros((1, 2)), np.cumsum(v, axis=0)])
            # 장주기 드리프트는 원궤적 따라가고 단주기 출렁임만 교정
            drift = np.stack([cv2.GaussianBlur(
                (P_raw - P_des)[:, k:k + 1].astype(np.float32), (1, 0),
                sigmaX=0, sigmaY=24.0).ravel() for k in (0, 1)], 1)
            corr = ((P_des + drift) - P_raw) * float(min(1.0, k_smooth))
            mc = float(np.abs(corr).max())
            zoom = max(1.015, 1.0 + 2.2 * mc / min(W, H))
        cap.release()
        cap = cv2.VideoCapture(src)

    # ---- 패스 2: 보정 렌더 ----
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr48le", "-s", f"{W}x{H}",
           "-r", f"{fps}", "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?",
           "-c:a", "copy", "-c:v", "libx265", "-preset", "medium", "-crf", "16",
           "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", dst]
    flags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags)

    GY, GX = np.mgrid[0:H, 0:W].astype(np.float32)

    def warp_path(fr, idx):
        if corr is None or fr is None:
            return fr
        j = min(idx, len(corr) - 1)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, zoom)
        M[0, 2] += float(corr[j, 0]) * zoom
        M[1, 2] += float(corr[j, 1]) * zoom
        return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT101)

    def enhance(img8):
        f = img8.astype(np.float32)
        if k_detail <= 0.02:
            return f
        g = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # 텍스처 게이트: 원래 매끈한 영역(하늘/구름/안개)은 인핸스 제외
        t = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (0, 0), 6)
        tex = np.clip((t - 1.5) / 4.5, 0, 1)
        tex = cv2.GaussianBlur(tex, (0, 0), 18)
        tex = (tex * 0.95 + 0.05).astype(np.float32)
        out = f.copy()
        for sig, amt in CLARITY:
            blur = cv2.GaussianBlur(out, (0, 0), sig)
            luma = cv2.GaussianBlur(g, (0, 0), sig) / 255.0
            prot = (1.0 - np.abs(luma - 0.5) * 1.6).clip(0.25, 1.0)
            out = out + (out - blur) * (amt * (prot * tex)[:, :, None])
        e = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (0, 0), 3)
        wk = ((1.0 - np.clip(e / 25.0, 0, 1)) * 0.8 + 0.2) * tex
        blur2 = cv2.GaussianBlur(out, (0, 0), 1.1)
        out = out + (out - blur2) * (SHARP * wk[:, :, None])
        return np.clip(out, 0, 255)

    rng = np.random.default_rng(20260612)
    lock_acc = prev_gs = prev_lo_gray = prev_flow = None
    idx = 0
    try:
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError("영상을 읽을 수 없습니다")
        cur = warp_path(fr, 0)
        ok, fr = cap.read()
        nxt = warp_path(fr, 1) if ok else None
        while cur is not None:
            g8 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            gs = cv2.resize(g8, None, fx=0.5, fy=0.5)
            enh = enhance(cur)
            # 텍스처 락킹: 고주파를 직전 프레임에서 흐름 따라 이월
            if LOCK > 0.02:
                lo = cv2.GaussianBlur(enh, (0, 0), LOCK_SIG)
                hi = enh - lo
                if lock_acc is not None:
                    flb = cv2.calcOpticalFlowFarneback(gs, prev_gs, None,
                                                       0.5, 3, 25, 3, 5, 1.2, 0)
                    flb = cv2.resize(flb, (W, H)) * 2.0
                    mxb = (GX + flb[:, :, 0]).astype(np.float32)
                    myb = (GY + flb[:, :, 1]).astype(np.float32)
                    warped = cv2.remap(lock_acc, mxb, myb, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT101)
                    gpw = cv2.remap(prev_lo_gray, mxb, myb, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT101)
                    dlow = np.abs(cv2.GaussianBlur(g8.astype(np.float32),
                                                   (0, 0), LOCK_SIG) - gpw)
                    conf = (1.0 - np.clip((dlow - 4.0) / 12.0, 0, 1)).astype(np.float32)
                    kk = (LOCK * conf)[:, :, None]
                    hi = hi * (1 - kk) + warped * kk
                lock_acc = hi
                prev_lo_gray = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), LOCK_SIG)
                enh = np.clip(lo + hi, 0, 255)
            # 180도 셔터 모션블러
            if SHUTTER > 0.02:
                if nxt is not None:
                    g2s = cv2.resize(cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY), None,
                                     fx=0.5, fy=0.5)
                    fl = cv2.calcOpticalFlowFarneback(gs, g2s, None,
                                                      0.5, 3, 25, 3, 5, 1.2, 0)
                    fl = cv2.resize(fl, (W, H)) * 2.0
                    fl[:, :, 0] = cv2.GaussianBlur(fl[:, :, 0], (0, 0), 5)
                    fl[:, :, 1] = cv2.GaussianBlur(fl[:, :, 1], (0, 0), 5)
                    prev_flow = fl
                fl = prev_flow
                if fl is not None:
                    acc = np.zeros((H, W, 3), np.float32)
                    for t in np.linspace(-SHUTTER / 2, SHUTTER / 2, TAPS):
                        t = float(t)
                        if abs(t) < 1e-6:
                            acc += enh
                        else:
                            acc += cv2.remap(
                                enh, (GX + fl[:, :, 0] * t).astype(np.float32),
                                (GY + fl[:, :, 1] * t).astype(np.float32),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                    enh = acc / TAPS
            # 그레인 (블러 뒤 — 센서 노이즈는 잔상이 없음)
            if GRAIN_AMP > 0.02:
                gl = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), 4) / 255.0
                grain = cv2.GaussianBlur(rng.normal(0, 1.0, (H, W)).astype(np.float32),
                                         (0, 0), 0.6)
                gw = (1.0 - np.abs(gl - 0.45) * 1.5).clip(0.15, 1.0) * GRAIN_AMP
                enh = enh + (grain * gw)[:, :, None]
            proc.stdin.write((np.clip(enh, 0, 255) * 257.0).astype(np.uint16).tobytes())
            prev_gs = gs
            idx += 1
            if progress and idx % 3 == 0:
                progress(min(99, 20 + int(idx / max(1, n_est) * 79)))
            cur = nxt
            if cur is not None:
                ok, fr = cap.read()
                nxt = warp_path(fr, idx + 1) if ok else None
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg 인코딩 실패")
    if progress:
        progress(100)
    return dst


def process_file(src, fn, suffix, progress=None):
    img = imread_u(src)
    if img is None:
        raise RuntimeError("이미지를 읽을 수 없습니다")
    result = fn(img, progress=progress)
    base, _ = os.path.splitext(src)
    dst = base + suffix + ".png"
    imwrite_u(dst, result)
    return dst


# ---------------- GUI ----------------
def run_gui(initial_files):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    # 고DPI 선명 렌더링: 기본 tkinter는 96DPI 가정이라 OS가 창을 비트맵 확대해
    # 글자가 뿌옇게 보인다. DPI 인지를 선언해 실제 픽셀로 렌더하고, 대신
    # 고정 픽셀 치수들을 ui_scale 배로 키워 물리 크기를 유지한다.
    ui_scale = 1.0
    if os.name == "nt":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)   # 시스템 DPI 인지
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            try:
                ui_scale = ctypes.windll.user32.GetDpiForSystem() / 96.0
            except Exception:
                dc = ctypes.windll.user32.GetDC(0)
                ui_scale = ctypes.windll.gdi32.GetDeviceCaps(dc, 88) / 96.0
                ctypes.windll.user32.ReleaseDC(0, dc)
        except Exception:
            ui_scale = 1.0

    def UIS(v):
        """96DPI 기준 픽셀값 → 현재 DPI 픽셀값"""
        return int(round(v * ui_scale))

    dnd_ok = False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        dnd_ok = True
    except Exception:
        root = tk.Tk()

    root.title(APP_TITLE)
    try:                                    # 폰트(pt)가 실제 DPI 기준으로 환산되도록
        root.tk.call("tk", "scaling", ui_scale * 96.0 / 72.0)
    except Exception:
        pass

    global FONT                             # Pretendard 등 설치돼 있으면 우선 사용
    try:
        import tkinter.font as tkfont
        _fams = set(tkfont.families(root))
        for _f in ("Pretendard", "Pretendard Variable", "SUIT"):
            if _f in _fams:
                FONT = _f
                break
    except Exception:
        pass
    root.geometry(f"{UIS(1020)}x{UIS(780)}")
    root.minsize(UIS(780), UIS(640))
    root.configure(bg=COL["bg"])

    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(ctypes.c_int(1)), 4) == 0:
                break
    except Exception:
        pass

    try:
        ipath = resource_path("no photoshop.png")
        if os.path.isfile(ipath):
            _icon = tk.PhotoImage(file=ipath)
            root.iconphoto(True, _icon)
    except Exception:
        pass

    def rrect(cv, x0, y0, x1, y1, r, **kw):
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
               x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return cv.create_polygon(pts, smooth=True, **kw)

    # ---------- 좌측 사이드바 (v2 모노크롬 레이아웃) ----------
    shell = tk.Frame(root, bg=COL["bg"])
    shell.pack(fill="both", expand=True)

    side = tk.Frame(shell, bg=COL["side"], width=UIS(206))
    side.pack(side="left", fill="y")
    side.pack_propagate(False)
    tk.Frame(shell, bg=COL["border"], width=1).pack(side="left", fill="y")

    body = tk.Frame(shell, bg=COL["bg"])
    body.pack(side="right", fill="both", expand=True)

    pages = {}
    current = {"idx": 0}
    TABS = ["밴딩 제거", "실사 질감", "Remove GPT", "우글거림 수정", "디테일 보강", "포커싱",
            "배경 제거", "가상 카메라 β"]
    MAIN_TABS = [0, 5, 6, 7]                # 메인 노출 + TEST: 가상 카메라
    LEGACY_TABS = [1, 2, 3, 4]              # 사이드바 하단 흐린 그룹

    tk.Label(side, text="no photoshop", bg=COL["side"], fg=COL["text"],
             font=(FONT, 13, "bold")).pack(anchor="w", padx=UIS(20), pady=(UIS(20), 0))
    tk.Label(side, text="이미지 후처리", bg=COL["side"], fg=COL["dim"],
             font=(FONT, 9)).pack(anchor="w", padx=UIS(20), pady=(UIS(1), UIS(16)))

    nav_draws = []

    def nav_item(idx, small=False):
        h = UIS(30) if small else UIS(38)
        cv = tk.Canvas(side, height=h, bg=COL["side"], highlightthickness=0,
                       cursor="hand2")
        cv.pack(fill="x", padx=UIS(10), pady=UIS(1))
        st = {"hover": False}

        def draw(_=None):
            cv.delete("all")
            w = cv.winfo_width() or UIS(186)
            active = (current["idx"] == idx)
            if active:
                rrect(cv, 0, 0, w, h, UIS(9), fill=COL["card2"], outline="")
            elif st["hover"]:
                rrect(cv, 0, 0, w, h, UIS(9), fill=COL["card"], outline="")
            if active:
                fg, fw = COL["text"], "bold"
            elif st["hover"]:
                fg, fw = COL["text"], "normal"
            else:
                fg, fw = (COL["dim"] if small else COL["sub"]), "normal"
            cv.create_text(UIS(14), h / 2, text=TABS[idx], anchor="w", fill=fg,
                           font=(FONT, 9 if small else 11, fw))
        cv.bind("<Configure>", draw)
        cv.bind("<Enter>", lambda e: (st.update(hover=True), draw()))
        cv.bind("<Leave>", lambda e: (st.update(hover=False), draw()))
        cv.bind("<Button-1>", lambda e: select_tab(idx))
        nav_draws.append(draw)

    for _i in MAIN_TABS:
        nav_item(_i)

    tk.Label(side, text="레 거 시", bg=COL["side"], fg=COL["dim"],
             font=(FONT, 8, "bold")).pack(anchor="w", padx=UIS(24), pady=(UIS(20), UIS(3)))
    for _i in LEGACY_TABS:
        nav_item(_i, small=True)

    tk.Label(side, text=APP_VER, bg=COL["side"], fg=COL["dim"],
             font=(MONO, 9)).pack(side="bottom", anchor="w", padx=UIS(20), pady=UIS(16))

    def select_tab(idx):
        if idx != current["idx"]:
            current["idx"] = idx
            pages[idx].tkraise()
        for d in nav_draws:
            d()

    def canvas_button(parent, text, width, height, primary=True):
        width, height = UIS(width), UIS(height)
        cv = tk.Canvas(parent, width=width, height=height, bg=COL["bg"],
                       highlightthickness=0, cursor="hand2")
        state = {"hover": False, "enabled": True, "text": text}

        def draw():
            cv.delete("all")
            if primary:
                fill = COL["lime2"] if state["hover"] else COL["lime"]
                fg = "#101204"
                if not state["enabled"]:
                    fill, fg = COL["card2"], COL["dim"]
                rrect(cv, 1, 1, width - 1, height - 1, UIS(12), fill=fill, outline="")
            else:
                fill = COL["card2"] if state["hover"] else COL["card"]
                fg = COL["text"] if state["enabled"] else COL["dim"]
                rrect(cv, 1, 1, width - 1, height - 1, UIS(12), fill=fill,
                      outline=COL["border"])
            cv.create_text(width / 2, height / 2, text=state["text"], fill=fg,
                           font=(FONT, 11, "bold"))
        draw()
        cv.bind("<Enter>", lambda e: (state.update(hover=True), draw()))
        cv.bind("<Leave>", lambda e: (state.update(hover=False), draw()))

        def set_enabled(v):
            state["enabled"] = v
            draw()
        def set_text(t):
            state["text"] = t
            draw()
        cv.set_enabled = set_enabled
        cv.set_text = set_text
        cv.is_enabled = lambda: state["enabled"]
        return cv

    def progress_bar(parent, height=8):
        cv = tk.Canvas(parent, height=height, bg=COL["bg"], highlightthickness=0)
        st = {"frac": 0.0}

        def draw(_=None):
            cv.delete("all")
            w = cv.winfo_width() or 200
            rrect(cv, 0, 0, w, height, height // 2, fill=COL["card2"], outline="")
            fw = int(w * max(0.0, min(1.0, st["frac"])))
            if fw > height:
                rrect(cv, 0, 0, fw, height, height // 2, fill=COL["lime"], outline="")
        cv.bind("<Configure>", draw)

        def set_frac(f):
            st["frac"] = f
            draw()
        cv.set_frac = set_frac
        return cv

    # ---------- 탭 1·2 공용 페이지 ----------
    def build_page(hint_main, hint_file, fn, suffix, media=False):
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)

        mode = {"video": False}   # media=True일 때 IMG/VIDEO 선택

        if media:
            mode_row = tk.Frame(page, bg=COL["bg"])
            mode_row.pack(fill="x", padx=24, pady=(10, 0))
            mode_cv = tk.Canvas(mode_row, width=UIS(176), height=UIS(40), bg=COL["bg"],
                                highlightthickness=0, cursor="hand2")
            mode_cv.pack(side="left")
            MW, MH, MP = UIS(82), UIS(30), UIS(4)

            def draw_mode():
                mode_cv.delete("all")
                rrect(mode_cv, 0, 1, MW * 2 + MP * 2, 1 + MH + MP * 2, UIS(11),
                      fill=COL["card"], outline=COL["border"])
                for i, name in enumerate(["IMG", "VIDEO"]):
                    x0 = MP + i * MW
                    active = (i == 1) == mode["video"]
                    if active:
                        rrect(mode_cv, x0, 1 + MP, x0 + MW, 1 + MP + MH, UIS(9),
                              fill=COL["lime"], outline="")
                        fg = "#101204"
                    else:
                        fg = COL["sub"]
                    mode_cv.create_text(x0 + MW / 2, 1 + MP + MH / 2, text=name,
                                        fill=fg, font=(FONT, 9, "bold"))
            draw_mode()

            def on_mode_click(e):
                mode["video"] = e.x >= MP + MW
                draw_mode()
            mode_cv.bind("<Button-1>", on_mode_click)
            tk.Label(mode_row,
                     text="IMG = 이미지 일괄 처리  ·  VIDEO = 영상(mp4 등) 프레임별 처리 후 10bit 인코딩",
                     bg=COL["bg"], fg=COL["dim"], font=(FONT, 9)).pack(side="left", padx=12)

        hint_row = tk.Frame(page, bg=COL["bg"])
        hint_row.pack(fill="x", padx=24, pady=(12, 6))
        tk.Label(hint_row, text=hint_main, bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 10)).pack(side="left")
        tk.Label(hint_row, text=hint_file, bg=COL["bg"], fg=COL["dim"],
                 font=(MONO, 10)).pack(side="left")
        tk.Label(hint_row, text=" 로 저장돼요.", bg=COL["bg"], fg=COL["dim"],
                 font=(FONT, 10)).pack(side="left")

        drop = tk.Canvas(page, bg=COL["bg"], highlightthickness=0, height=UIS(250))
        drop.pack(fill="both", expand=True, padx=24, pady=(6, 0))
        pick_btn_box = {"bbox": (0, 0, 0, 0)}

        def draw_drop(_=None):
            drop.delete("all")
            w = drop.winfo_width() or 900
            h = drop.winfo_height() or 250
            rrect(drop, 2, 2, w - 2, h - 2, 18, fill=COL["bg"],
                  outline=COL["dash"], width=2)
            cx, cy = w / 2, h / 2
            iy = cy - 58
            for pts in [((cx, iy - 14), (cx, iy + 12)),
                        ((cx - 10, iy + 2), (cx, iy + 13)),
                        ((cx + 10, iy + 2), (cx, iy + 13)),
                        ((cx - 18, iy + 22), (cx + 18, iy + 22))]:
                drop.create_line(*pts[0], *pts[1], fill=COL["lime"], width=4,
                                 capstyle="round")
            head = "이미지를 여기에 던져 넣으세요" if dnd_ok else "파일 선택 버튼으로 이미지를 추가하세요"
            drop.create_text(cx, cy - 6, text=head, fill=COL["text"],
                             font=(FONT, 14, "bold"))
            sub = "drag & drop — 그게 전부예요" if dnd_ok else "여러 장 한꺼번에 선택할 수 있어요"
            drop.create_text(cx, cy + 22, text=sub, fill=COL["sub"], font=(FONT, 10))
            bw, bh = 210, 42
            bx0, by0 = cx - bw / 2, cy + 44
            rrect(drop, bx0, by0, bx0 + bw, by0 + bh, 10, fill=COL["bg"],
                  outline=COL["dash"], width=1)
            drop.create_text(cx, by0 + bh / 2, text="파일 선택 · 여러 장 가능",
                             fill=COL["text"], font=(FONT, 10, "bold"))
            pick_btn_box["bbox"] = (bx0, by0, bx0 + bw, by0 + bh)

        drop.bind("<Configure>", draw_drop)

        att_row = tk.Frame(page, bg=COL["bg"])
        att_row.pack(fill="x", padx=24, pady=(10, 0))
        att_label = tk.Label(att_row, text="첨부 0장", bg=COL["bg"],
                             fg=COL["sub"], font=(FONT, 10), width=8, anchor="w")
        att_label.pack(side="left", anchor="n", pady=14)
        thumbs_cv = tk.Canvas(att_row, bg=COL["bg"], highlightthickness=0, height=72)
        thumbs_cv.pack(side="left", fill="x", expand=True)
        thumbs_in = tk.Frame(thumbs_cv, bg=COL["bg"])
        thumbs_cv.create_window((0, 0), window=thumbs_in, anchor="nw")
        thumbs_in.bind("<Configure>", lambda e: thumbs_cv.configure(
            scrollregion=thumbs_cv.bbox("all") or (0, 0, 0, 0)))
        thumbs_cv.bind("<MouseWheel>",
                       lambda e: thumbs_cv.xview_scroll(-1 if e.delta > 0 else 1, "units"))

        log_card = tk.Frame(page, bg=COL["card"], highlightbackground=COL["border"],
                            highlightthickness=1)
        log_card.pack(fill="both", expand=True, padx=24, pady=(10, 0))
        log_head = tk.Frame(log_card, bg=COL["card"])
        log_head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(log_head, text="처리 로그", bg=COL["card"], fg=COL["text"],
                 font=(FONT, 10, "bold")).pack(side="left")
        ok_label = tk.Label(log_head, text="", bg=COL["card"], fg=COL["lime"],
                            font=(FONT, 10, "bold"))
        ok_label.pack(side="right")
        log = tk.Text(log_card, bg=COL["card"], fg=COL["text"], bd=0,
                      highlightthickness=0, height=6, state="disabled",
                      font=(MONO, 10))
        log.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        for tag, c in [("ok", COL["ok"]), ("name", COL["text"]),
                       ("out", COL["dim"]), ("err", COL["err"])]:
            log.tag_configure(tag, foreground=c)

        bottom = tk.Frame(page, bg=COL["bg"])
        bottom.pack(fill="x", padx=24, pady=(8, 16))
        start_btn = canvas_button(bottom, "▶  변환 시작", 170, 48, primary=True)
        start_btn.pack(side="right")
        status = tk.Label(bottom, text="대기 중", bg=COL["bg"], fg=COL["sub"],
                          font=(FONT, 10), anchor="e")
        status.pack(side="right", padx=(0, 14), pady=10)
        pbar = progress_bar(bottom)
        pbar.pack(side="left", fill="x", expand=True, padx=(0, 14), pady=20)

        def set_status(msg):
            status.configure(text=msg)

        def append_log(parts):
            log.configure(state="normal")
            for text, tag in parts:
                log.insert("end", text, tag)
            log.see("end")
            log.configure(state="disabled")

        jobs = queue.Queue()
        counter = {"done": 0, "fail": 0}
        batch = {"total": 0, "done": 0}
        seen = set()
        pending = []
        thumbs = {}
        timing = {"avg": 6.0}

        def thumb_photo(p):
            try:
                if p.lower().endswith(VEXTS):
                    cap_t = cv2.VideoCapture(p)
                    ok_t, im = cap_t.read()
                    cap_t.release()
                    if not ok_t:
                        return None
                else:
                    im = imread_u(p)
                tw, th = 84, 60
                ih, iw = im.shape[:2]
                s = max(tw / iw, th / ih)
                im = cv2.resize(im, (max(1, int(iw * s)), max(1, int(ih * s))),
                                interpolation=cv2.INTER_AREA)
                ih, iw = im.shape[:2]
                x0, y0 = (iw - tw) // 2, (ih - th) // 2
                im = im[y0:y0 + th, x0:x0 + tw]
                ok, buf = cv2.imencode(".png", im)
                return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
            except Exception:
                return None

        def update_att_label():
            att_label.configure(text=f"첨부 {len(thumbs)}장")

        def update_pending_status():
            if pending:
                est = max(1, int(round(timing["avg"] * len(pending))))
                set_status(f"대기 {len(pending)}장 · 예상 ~{est}초")
            else:
                set_status("대기 중")

        def add_thumb(p):
            holder = tk.Frame(thumbs_in, bg=COL["card2"],
                              highlightbackground=COL["border"], highlightthickness=1)
            holder.pack(side="left", padx=(0, 10))
            ph = thumb_photo(p)
            if ph is not None:
                tk.Label(holder, image=ph, bg=COL["card2"], bd=0).pack()
            else:
                tk.Label(holder, text="?", width=10, height=4,
                         bg=COL["card2"], fg=COL["dim"]).pack()
            name = os.path.basename(p)
            if len(name) > 14:
                name = name[:11] + "…"
            tk.Label(holder, text=name, bg=COL["card2"], fg=COL["dim"],
                     font=(MONO, 7)).pack(fill="x")
            thumbs[p] = (holder, ph)
            update_att_label()

        def remove_thumb(p):
            item = thumbs.pop(p, None)
            if item:
                item[0].destroy()
            seen.discard(os.path.normcase(os.path.abspath(p)))
            update_att_label()

        def worker():
            while True:
                p = jobs.get()
                name = os.path.basename(p)
                t0 = time.time()
                try:
                    def prog(v, name=name):
                        root.after(0, set_status,
                                   f"처리 중: {name}  {v}% · 대기 {jobs.qsize()}장")
                        frac = (batch["done"] + v / 100.0) / max(1, batch["total"])
                        root.after(0, pbar.set_frac, frac)
                    root.after(0, set_status, f"처리 중: {name} · 대기 {jobs.qsize()}장")
                    if p.lower().endswith(VEXTS):
                        dst = stack_deband_video(p, progress=prog)
                    else:
                        dst = process_file(p, fn, suffix, progress=prog)
                    counter["done"] += 1
                    timing["avg"] = timing["avg"] * 0.5 + (time.time() - t0) * 0.5
                    root.after(0, append_log, [("✓ ", "ok"), (name + "\n", "name"),
                                               ("   →  " + os.path.basename(dst) + "\n", "out")])
                except Exception as e:
                    counter["fail"] += 1
                    root.after(0, append_log, [("✗ ", "err"),
                                               (f"{name} 실패: {e}\n", "err")])
                finally:
                    batch["done"] += 1
                    root.after(0, pbar.set_frac,
                               batch["done"] / max(1, batch["total"]))
                    root.after(0, remove_thumb, p)
                    jobs.task_done()
                    msg = f"● 성공 {counter['done']}장"
                    if counter["fail"]:
                        msg += f" · 실패 {counter['fail']}장"
                    root.after(0, ok_label.configure, {"text": msg})
                    if jobs.empty():
                        def finish():
                            set_status("완료")
                            pbar.set_frac(0.0)
                            batch["total"] = batch["done"] = 0
                            start_btn.set_text("▶  변환 시작")
                            start_btn.set_enabled(True)
                        root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

        def accepted_exts():
            if media and mode["video"]:
                return VEXTS
            return EXTS

        def add_files(files):
            ok_exts = accepted_exts()
            files = [f for f in files if f.lower().endswith(ok_exts) and os.path.isfile(f)]
            n_new = 0
            for p in files:
                key = os.path.normcase(os.path.abspath(p))
                if key in seen:
                    continue
                seen.add(key)
                n_new += 1
                pending.append(p)
                add_thumb(p)
            if n_new:
                update_pending_status()
            elif files == []:
                set_status("현재 모드와 맞지 않는 파일이에요 — 상단에서 IMG/VIDEO를 확인하세요"
                           if media else "지원하지 않는 파일 형식이에요")

        def start_convert(_=None):
            if not start_btn.is_enabled():
                return
            if not pending:
                set_status("첨부된 대기 파일이 없어요 — 먼저 던져 넣으세요")
                return
            batch["total"] += len(pending)
            for p in pending:
                jobs.put(p)
            pending.clear()
            start_btn.set_text("⏳  변환 중…")
            start_btn.set_enabled(False)
            pbar.set_frac(0.0)

        def pick_files():
            if media and mode["video"]:
                ft = [("영상", "*.mp4 *.mov *.avi *.mkv *.webm"), ("모든 파일", "*.*")]
                title = "영상 선택"
            else:
                ft = [("이미지", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                      ("모든 파일", "*.*")]
                title = "이미지 선택"
            files = filedialog.askopenfilenames(title=title, filetypes=ft)
            add_files(list(files))

        def on_drop_click(e):
            x0, y0, x1, y1 = pick_btn_box["bbox"]
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                pick_files()
        drop.bind("<Button-1>", on_drop_click)
        start_btn.bind("<Button-1>", start_convert)

        if dnd_ok:
            def on_drop(event):
                try:
                    paths = root.tk.splitlist(event.data)
                except Exception:
                    paths = [event.data]
                add_files([p.strip("{}") for p in paths])
            for widget in (page, drop, log, thumbs_cv):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", on_drop)
                except Exception:
                    pass

        page.add_files = add_files
        return page

    # ---------- 탭 3: Remove GPT ----------
    def build_removegpt_page():
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)

        hint_row = tk.Frame(page, bg=COL["bg"])
        hint_row.pack(fill="x", padx=24, pady=(12, 6))
        tk.Label(hint_row,
                 text="AI 이미지 특유의 도트(점가루)를 지워요 — 변환 후 슬라이더로 원본과 비교하고, 마음에 들면 저장하세요.",
                 bg=COL["bg"], fg=COL["sub"], font=(FONT, 10)).pack(side="left")

        # 컨트롤 줄: 파일 선택 / 강도 / 변환 / 저장
        ctrl = tk.Frame(page, bg=COL["bg"])
        ctrl.pack(fill="x", padx=24, pady=(0, 6))
        pick_btn = canvas_button(ctrl, "이미지 선택", 140, 44, primary=False)
        pick_btn.pack(side="left")

        tk.Label(ctrl, text="강도", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 10)).pack(side="left", padx=(16, 6))
        strength = tk.IntVar(value=0)   # 0=표준, 1=강력

        def mk_radio(text, val):
            rb = tk.Radiobutton(ctrl, text=text, variable=strength, value=val,
                                indicatoron=0, bd=0, font=(FONT, 10, "bold"),
                                bg=COL["card"], fg=COL["sub"],
                                selectcolor=COL["lime"], activebackground=COL["card2"],
                                activeforeground=COL["text"], width=14, pady=10,
                                cursor="hand2", highlightthickness=1,
                                highlightbackground=COL["border"])
            rb.pack(side="left", padx=(0, 6))
            return rb
        mk_radio("표준 (점만 정밀)", 0)
        mk_radio("강력 (전부 제거)", 1)
        if "--strong" in sys.argv:   # UI 테스트용
            strength.set(1)

        save_btn = canvas_button(ctrl, "💾  저장", 120, 44, primary=False)
        save_btn.pack(side="right")
        save_btn.set_enabled(False)
        run_btn = canvas_button(ctrl, "▶  변환 시작", 150, 44, primary=True)
        run_btn.pack(side="right", padx=(0, 8))

        status = tk.Label(page, text="이미지를 첨부하세요", bg=COL["bg"],
                          fg=COL["sub"], font=(FONT, 10), anchor="w")
        status.pack(fill="x", padx=24)
        pbar = progress_bar(page)
        pbar.pack(fill="x", padx=24, pady=(4, 0))

        view = tk.Canvas(page, bg=COL["bg"], highlightthickness=0)
        view.pack(fill="both", expand=True, padx=24, pady=(6, 16))

        st = {"path": None, "orig": None, "proc": None,
              "disp_o": None, "disp_p": None, "photo": None,
              "split": 0.5, "busy": False, "geo": (0, 0, 0, 0)}

        def set_status(msg, color=None):
            status.configure(text=msg, fg=color or COL["sub"])

        def fit_image(im, cw, ch):
            ih, iw = im.shape[:2]
            s = min(cw / iw, ch / ih)
            nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
            return cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

        def np_to_photo(arr):
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            hh, ww = rgb.shape[:2]
            header = f"P6 {ww} {hh} 255 ".encode()
            return tk.PhotoImage(data=header + rgb.tobytes())

        def redraw(_=None):
            view.delete("all")
            cw = max(2, view.winfo_width())
            ch = max(2, view.winfo_height())
            if st["orig"] is None:
                # 드롭 안내
                rrect(view, 2, 2, cw - 2, ch - 2, 18, fill=COL["bg"],
                      outline=COL["dash"], width=2)
                cx, cy = cw / 2, ch / 2
                iy = cy - 46
                for pts in [((cx, iy - 14), (cx, iy + 12)),
                            ((cx - 10, iy + 2), (cx, iy + 13)),
                            ((cx + 10, iy + 2), (cx, iy + 13)),
                            ((cx - 18, iy + 22), (cx + 18, iy + 22))]:
                    view.create_line(*pts[0], *pts[1], fill=COL["lime"], width=4,
                                     capstyle="round")
                head = "이미지 1장을 여기에 던져 넣으세요" if dnd_ok else "이미지 선택 버튼으로 추가하세요"
                view.create_text(cx, cy + 6, text=head, fill=COL["text"],
                                 font=(FONT, 13, "bold"))
                view.create_text(cx, cy + 32, text="새 이미지를 넣으면 이전 이미지를 대체해요",
                                 fill=COL["sub"], font=(FONT, 9))
                return

            disp_src = st["orig"]
            fit_o = fit_image(disp_src, cw, ch)
            nh, nw = fit_o.shape[:2]
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            st["geo"] = (ox, oy, nw, nh)

            if st["proc"] is None:
                st["photo"] = np_to_photo(fit_o)
                view.create_image(ox, oy, image=st["photo"], anchor="nw")
                view.create_rectangle(ox, oy, ox + nw, oy + nh,
                                      outline=COL["border"])
                return

            fit_p = fit_image(st["proc"], cw, ch)
            sx = int(np.clip(st["split"], 0, 1) * nw)
            comp = fit_o.copy()
            comp[:, :sx] = fit_p[:, :sx]
            st["photo"] = np_to_photo(comp)
            view.create_image(ox, oy, image=st["photo"], anchor="nw")
            view.create_rectangle(ox, oy, ox + nw, oy + nh, outline=COL["border"])
            # 분할선 + 핸들
            lx = ox + sx
            view.create_line(lx, oy, lx, oy + nh, fill=COL["lime"], width=2)
            view.create_oval(lx - 13, oy + nh / 2 - 13, lx + 13, oy + nh / 2 + 13,
                             fill=COL["lime"], outline="")
            view.create_text(lx, oy + nh / 2, text="◀ ▶", fill="#101204",
                             font=(FONT, 8, "bold"))
            view.create_text(ox + 8, oy + 12, text="AFTER", fill=COL["lime"],
                             font=(FONT, 9, "bold"), anchor="w")
            view.create_text(ox + nw - 8, oy + 12, text="BEFORE", fill=COL["sub"],
                             font=(FONT, 9, "bold"), anchor="e")

        view.bind("<Configure>", redraw)

        def on_drag(e):
            if st["proc"] is None:
                return
            ox, oy, nw, nh = st["geo"]
            if nw > 0:
                st["split"] = (e.x - ox) / nw
                redraw()
        view.bind("<B1-Motion>", on_drag)
        view.bind("<Button-1>", on_drag)

        def load_image(p):
            if st["busy"]:
                return
            im = imread_u(p)
            if im is None:
                set_status(f"이미지를 읽을 수 없어요: {os.path.basename(p)}", COL["err"])
                return
            st.update(path=p, orig=im, proc=None, split=0.5)
            save_btn.set_enabled(False)
            set_status(f"첨부됨: {os.path.basename(p)}  ({im.shape[1]}x{im.shape[0]}) — 강도를 고르고 [▶ 변환 시작]")
            redraw()

        def add_files(files):
            files = [f for f in files if f.lower().endswith(EXTS) and os.path.isfile(f)]
            if files:
                load_image(files[0])

        def pick_file(_=None):
            if not pick_btn.is_enabled():
                return
            f = filedialog.askopenfilename(
                title="이미지 선택",
                filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                           ("모든 파일", "*.*")])
            if f:
                load_image(f)
        pick_btn.bind("<Button-1>", pick_file)

        def run_convert(_=None):
            if st["busy"] or not run_btn.is_enabled():
                return
            if st["orig"] is None:
                set_status("이미지를 먼저 첨부하세요", COL["err"])
                return
            st["busy"] = True
            run_btn.set_text("⏳  변환 중…")
            run_btn.set_enabled(False)
            save_btn.set_enabled(False)
            pbar.set_frac(0.0)
            strong = strength.get() == 1
            mode = "강력" if strong else "표준"

            def job():
                try:
                    def prog(v):
                        root.after(0, set_status, f"변환 중({mode})…  {v}%")
                        root.after(0, pbar.set_frac, v / 100.0)
                    result = remove_gpt(st["orig"], strong=strong, progress=prog)
                    def done():
                        st.update(proc=result, split=0.5, busy=False)
                        run_btn.set_text("▶  변환 시작")
                        run_btn.set_enabled(True)
                        save_btn.set_enabled(True)
                        pbar.set_frac(0.0)
                        set_status("완료 — 슬라이더로 비교해 보세요. 마음에 들면 [💾 저장]", COL["lime"])
                        redraw()
                    root.after(0, done)
                except Exception as e:
                    def fail(e=e):
                        st["busy"] = False
                        run_btn.set_text("▶  변환 시작")
                        run_btn.set_enabled(True)
                        pbar.set_frac(0.0)
                        set_status(f"실패: {e}", COL["err"])
                    root.after(0, fail)
            threading.Thread(target=job, daemon=True).start()
        run_btn.bind("<Button-1>", run_convert)

        def save_result(_=None):
            if st["proc"] is None or not save_btn.is_enabled():
                return
            base = os.path.splitext(os.path.basename(st["path"] or "image"))[0]
            f = filedialog.asksaveasfilename(
                title="다른 이름으로 저장",
                defaultextension=".png",
                initialfile=base + "_nogpt.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if not f:
                return
            try:
                imwrite_u(f, st["proc"])
                set_status(f"저장됨: {f}", COL["lime"])
            except Exception as e:
                set_status(f"저장 실패: {e}", COL["err"])
        save_btn.bind("<Button-1>", save_result)

        if dnd_ok:
            def on_drop(event):
                try:
                    paths = root.tk.splitlist(event.data)
                except Exception:
                    paths = [event.data]
                add_files([p.strip("{}") for p in paths])
            for widget in (page, view):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", on_drop)
                except Exception:
                    pass

        page.add_files = add_files
        page.run_convert = run_convert
        return page

    # ---------- 탭 2: 실사 질감 (실시간 편집기) ----------
    def build_photoreal_page():
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(page,
                 text="실제 카메라 질감을 효과별로 조절해요 — 슬라이더를 움직이면 미리보기에 바로 반영되고, 가운데 핸들로 원본과 비교하세요.",
                 bg=COL["bg"], fg=COL["sub"], font=(FONT, 10),
                 anchor="w").pack(fill="x", padx=24, pady=(12, 6))

        ctrl = tk.Frame(page, bg=COL["bg"])
        ctrl.pack(fill="x", padx=24, pady=(0, 4))
        pick_btn = canvas_button(ctrl, "이미지 선택", 140, 44, primary=False)
        pick_btn.pack(side="left")
        save_btn = canvas_button(ctrl, "💾  저장", 130, 44, primary=True)
        save_btn.pack(side="right")
        save_btn.set_enabled(False)

        status = tk.Label(page, text="이미지를 첨부하세요", bg=COL["bg"],
                          fg=COL["sub"], font=(FONT, 10), anchor="w")
        status.pack(fill="x", padx=24)
        pbar = progress_bar(page)
        pbar.pack(fill="x", padx=24, pady=(4, 0))

        content = tk.Frame(page, bg=COL["bg"])
        content.pack(fill="both", expand=True, padx=24, pady=(6, 16))

        view = tk.Canvas(content, bg=COL["bg"], highlightthickness=0)
        view.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(content, bg=COL["card"],
                         highlightbackground=COL["border"], highlightthickness=1,
                         width=UIS(280))
        panel.pack(side="right", fill="y", padx=(14, 0))
        panel.pack_propagate(False)
        tk.Label(panel, text="효과 조절", bg=COL["card"], fg=COL["text"],
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=14, pady=(12, 2))

        st = {"path": None, "orig": None, "proxy": None, "proc": None,
              "split": 0.5, "geo": (0, 0, 0, 0), "photo": None,
              "token": 0, "after_id": None, "saving": False}

        def set_status(msg, color=None):
            status.configure(text=msg, fg=color or COL["sub"])

        EFFECTS = [("lift", "블랙 리프트"), ("rolloff", "하이라이트 롤오프"),
                   ("bloom", "렌즈 블룸"), ("ca", "색수차"),
                   ("vignette", "비네팅"), ("grain", "필름 그레인")]
        controls = {}

        # ---- 프리셋 ----
        from tkinter import ttk
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Np.TCombobox", fieldbackground=COL["card2"],
                        background=COL["card2"], foreground=COL["text"],
                        arrowcolor=COL["lime"], bordercolor=COL["border"],
                        lightcolor=COL["card2"], darkcolor=COL["card2"],
                        selectbackground=COL["card2"], selectforeground=COL["text"])
        style.map("Np.TCombobox",
                  fieldbackground=[("readonly", COL["card2"])],
                  foreground=[("readonly", COL["text"])])
        root.option_add("*TCombobox*Listbox.background", COL["card2"])
        root.option_add("*TCombobox*Listbox.foreground", COL["text"])
        root.option_add("*TCombobox*Listbox.selectBackground", COL["lime"])
        root.option_add("*TCombobox*Listbox.selectForeground", "#101204")
        root.option_add("*TCombobox*Listbox.font", (FONT, 10))

        PRESETS = [
            ("기본값",            {"lift": 100, "rolloff": 100, "bloom": 100, "ca": 100, "vignette": 100, "grain": 100}),
            ("ARRI 시네마틱",     {"lift": 120, "rolloff": 150, "bloom": 60,  "ca": 30,  "vignette": 70,  "grain": 55}),
            ("코닥 35mm 필름",    {"lift": 150, "rolloff": 130, "bloom": 85,  "ca": 60,  "vignette": 95,  "grain": 145}),
            ("8mm 빈티지",        {"lift": 180, "rolloff": 160, "bloom": 130, "ca": 185, "vignette": 175, "grain": 200}),
            ("블록버스터",        {"lift": 80,  "rolloff": 120, "bloom": 155, "ca": 100, "vignette": 120, "grain": 75}),
            ("다큐 디지털 클린",  {"lift": 40,  "rolloff": 80,  "bloom": 30,  "ca": 20,  "vignette": 40,  "grain": 45}),
            ("호러 로우라이트",   {"lift": 60,  "rolloff": 100, "bloom": 70,  "ca": 85,  "vignette": 190, "grain": 170}),
        ]
        PRESET_MAP = dict(PRESETS)

        preset_row = tk.Frame(panel, bg=COL["card"])
        preset_row.pack(fill="x", padx=14, pady=(4, 2))
        tk.Label(preset_row, text="프리셋", bg=COL["card"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left")
        preset_cb = ttk.Combobox(preset_row, values=[n for n, _ in PRESETS],
                                 state="readonly", style="Np.TCombobox",
                                 font=(FONT, 10), width=18)
        preset_cb.current(0)
        preset_cb.pack(side="left", fill="x", expand=True, padx=(8, 0))

        def apply_preset(_=None):
            vals = PRESET_MAP.get(preset_cb.get())
            if not vals:
                return
            for k, (on, val, vl, scv) in controls.items():
                on.set(True)
                val.set(vals[k])
                vl.configure(text=f"{vals[k]}%")
                scv.redraw()
            preset_cb.selection_clear()
            schedule_render()
        preset_cb.bind("<<ComboboxSelected>>", apply_preset)

        # ---- 커스텀 슬라이더 (핸들 상시 표시 + 호버/드래그 반응) ----
        def make_slider(parent, val_var, vl_label):
            cv = tk.Canvas(parent, height=24, bg=COL["card"], highlightthickness=0,
                           cursor="hand2")
            stt = {"hover": False, "press": False}
            PADX = 9

            def draw(_=None):
                cv.delete("all")
                w = max(40, cv.winfo_width())
                ty, tr = 12, 3
                rrect(cv, PADX, ty - tr, w - PADX, ty + tr, tr,
                      fill=COL["card2"], outline="")
                frac = val_var.get() / 200.0
                hx = PADX + frac * (w - PADX * 2)
                if hx - PADX > 4:
                    rrect(cv, PADX, ty - tr, hx, ty + tr, tr,
                          fill="#5f7030", outline="")
                r = 8 if (stt["hover"] or stt["press"]) else 6
                color = COL["lime2"] if stt["press"] else COL["lime"]
                cv.create_oval(hx - r, ty - r, hx + r, ty + r, fill=color, outline="")
            cv.bind("<Configure>", draw)

            def setv(x):
                w = max(40, cv.winfo_width())
                frac = (x - PADX) / max(1, w - PADX * 2)
                v = int(round(min(1.0, max(0.0, frac)) * 200))
                val_var.set(v)
                vl_label.configure(text=f"{v}%")
                draw()
                schedule_render()
            cv.bind("<Button-1>", lambda e: (stt.update(press=True), setv(e.x)))
            cv.bind("<B1-Motion>", lambda e: setv(e.x))
            cv.bind("<ButtonRelease-1>", lambda e: (stt.update(press=False), draw()))
            cv.bind("<Enter>", lambda e: (stt.update(hover=True), draw()))
            cv.bind("<Leave>", lambda e: (stt.update(hover=False), draw()))
            cv.pack(fill="x")
            cv.redraw = draw
            return cv

        def make_effect_row(key, label):
            row = tk.Frame(panel, bg=COL["card"])
            row.pack(fill="x", padx=14, pady=(6, 0))
            on = tk.BooleanVar(value=True)
            val = tk.IntVar(value=100)
            head = tk.Frame(row, bg=COL["card"])
            head.pack(fill="x")
            tk.Checkbutton(head, text=label, variable=on, bg=COL["card"],
                           fg=COL["text"], activebackground=COL["card"],
                           activeforeground=COL["text"], selectcolor=COL["card2"],
                           font=(FONT, 10), bd=0, highlightthickness=0,
                           command=lambda: schedule_render()).pack(side="left")
            vl = tk.Label(head, text="100%", bg=COL["card"], fg=COL["lime"],
                          font=(MONO, 9))
            vl.pack(side="right")
            scv = make_slider(row, val, vl)
            controls[key] = (on, val, vl, scv)
        for k, lbl in EFFECTS:
            make_effect_row(k, lbl)

        def reset_all():
            preset_cb.current(0)
            for k, (on, val, vl, scv) in controls.items():
                on.set(True)
                val.set(100)
                vl.configure(text="100%")
                scv.redraw()
            schedule_render()
        tk.Button(panel, text="↺  전체 기본값", command=reset_all,
                  bg=COL["card2"], fg=COL["text"], activebackground=COL["border"],
                  activeforeground=COL["text"], bd=0, font=(FONT, 9, "bold"),
                  cursor="hand2", pady=8).pack(fill="x", padx=14, pady=(12, 12))

        def params():
            return {k: (val.get() / 100.0 if on.get() else 0.0)
                    for k, (on, val, vl, scv) in controls.items()}

        def fit_image(im, cw, ch):
            ih, iw = im.shape[:2]
            s = min(cw / iw, ch / ih)
            nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
            return cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

        def np_to_photo(arr):
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            hh, ww = rgb.shape[:2]
            return tk.PhotoImage(data=f"P6 {ww} {hh} 255 ".encode() + rgb.tobytes())

        def redraw(_=None):
            view.delete("all")
            cw = max(2, view.winfo_width())
            ch = max(2, view.winfo_height())
            if st["proxy"] is None:
                rrect(view, 2, 2, cw - 2, ch - 2, 18, fill=COL["bg"],
                      outline=COL["dash"], width=2)
                cx, cy = cw / 2, ch / 2
                iy = cy - 46
                for pts in [((cx, iy - 14), (cx, iy + 12)),
                            ((cx - 10, iy + 2), (cx, iy + 13)),
                            ((cx + 10, iy + 2), (cx, iy + 13)),
                            ((cx - 18, iy + 22), (cx + 18, iy + 22))]:
                    view.create_line(*pts[0], *pts[1], fill=COL["lime"], width=4,
                                     capstyle="round")
                head = "이미지 1장을 여기에 던져 넣으세요" if dnd_ok else "이미지 선택 버튼으로 추가하세요"
                view.create_text(cx, cy + 6, text=head, fill=COL["text"],
                                 font=(FONT, 13, "bold"))
                return
            fit_o = fit_image(st["proxy"], cw, ch)
            nh, nw = fit_o.shape[:2]
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            st["geo"] = (ox, oy, nw, nh)
            if st["proc"] is None:
                comp = fit_o
            else:
                fit_p = fit_image(st["proc"], cw, ch)
                sx = int(np.clip(st["split"], 0, 1) * nw)
                comp = fit_o.copy()
                comp[:, :sx] = fit_p[:, :sx]
            st["photo"] = np_to_photo(comp)
            view.create_image(ox, oy, image=st["photo"], anchor="nw")
            view.create_rectangle(ox, oy, ox + nw, oy + nh, outline=COL["border"])
            if st["proc"] is not None:
                sx = int(np.clip(st["split"], 0, 1) * nw)
                lx = ox + sx
                view.create_line(lx, oy, lx, oy + nh, fill=COL["lime"], width=2)
                view.create_oval(lx - 13, oy + nh / 2 - 13, lx + 13, oy + nh / 2 + 13,
                                 fill=COL["lime"], outline="")
                view.create_text(lx, oy + nh / 2, text="◀ ▶", fill="#101204",
                                 font=(FONT, 8, "bold"))
                view.create_text(ox + 8, oy + 12, text="AFTER", fill=COL["lime"],
                                 font=(FONT, 9, "bold"), anchor="w")
                view.create_text(ox + nw - 8, oy + 12, text="BEFORE", fill=COL["sub"],
                                 font=(FONT, 9, "bold"), anchor="e")

        view.bind("<Configure>", redraw)

        def on_drag(e):
            if st["proc"] is None:
                return
            ox, oy, nw, nh = st["geo"]
            if nw > 0:
                st["split"] = (e.x - ox) / nw
                redraw()
        view.bind("<B1-Motion>", on_drag)
        view.bind("<Button-1>", on_drag)

        def do_render():
            st["after_id"] = None
            if st["proxy"] is None:
                return
            st["token"] += 1
            token = st["token"]
            p = params()

            def job():
                res = photoreal_params(st["proxy"], p)
                def apply():
                    if token == st["token"]:
                        st["proc"] = res
                        redraw()
                root.after(0, apply)
            threading.Thread(target=job, daemon=True).start()

        def schedule_render():
            if st["proxy"] is None:
                return
            if st["after_id"]:
                root.after_cancel(st["after_id"])
            st["after_id"] = root.after(80, do_render)

        def load_image(p):
            if st["saving"]:
                return
            im = imread_u(p)
            if im is None:
                set_status(f"이미지를 읽을 수 없어요: {os.path.basename(p)}", COL["err"])
                return
            ih, iw = im.shape[:2]
            s = min(1.0, 1100 / max(iw, ih))
            proxy = cv2.resize(im, (max(1, int(iw * s)), max(1, int(ih * s))),
                               interpolation=cv2.INTER_AREA) if s < 1.0 else im.copy()
            st.update(path=p, orig=im, proxy=proxy, proc=None, split=0.5)
            save_btn.set_enabled(True)
            set_status(f"첨부됨: {os.path.basename(p)}  ({iw}x{ih}) — 슬라이더를 조절해 보세요")
            redraw()
            schedule_render()

        def add_files(files):
            files = [f for f in files if f.lower().endswith(EXTS) and os.path.isfile(f)]
            if files:
                load_image(files[0])

        def pick_file(_=None):
            if not pick_btn.is_enabled():
                return
            f = filedialog.askopenfilename(
                title="이미지 선택",
                filetypes=[("이미지", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                           ("모든 파일", "*.*")])
            if f:
                load_image(f)
        pick_btn.bind("<Button-1>", pick_file)

        def save_result(_=None):
            if st["orig"] is None or st["saving"] or not save_btn.is_enabled():
                return
            base = os.path.splitext(os.path.basename(st["path"] or "image"))[0]
            f = filedialog.asksaveasfilename(
                title="다른 이름으로 저장",
                defaultextension=".png",
                initialfile=base + "_photoreal.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if not f:
                return
            st["saving"] = True
            save_btn.set_text("⏳  저장 중…")
            save_btn.set_enabled(False)
            p = params()

            def job():
                try:
                    def prog(v):
                        root.after(0, pbar.set_frac, v / 100.0)
                        root.after(0, set_status, f"원본 해상도로 렌더링 중…  {v}%")
                    result = photoreal_params(st["orig"], p, progress=prog)
                    imwrite_u(f, result)
                    def done():
                        st["saving"] = False
                        save_btn.set_text("💾  저장")
                        save_btn.set_enabled(True)
                        pbar.set_frac(0.0)
                        set_status(f"저장됨: {f}", COL["lime"])
                    root.after(0, done)
                except Exception as e:
                    def fail(e=e):
                        st["saving"] = False
                        save_btn.set_text("💾  저장")
                        save_btn.set_enabled(True)
                        pbar.set_frac(0.0)
                        set_status(f"저장 실패: {e}", COL["err"])
                    root.after(0, fail)
            threading.Thread(target=job, daemon=True).start()
        save_btn.bind("<Button-1>", save_result)

        if dnd_ok:
            def on_drop(event):
                try:
                    paths = root.tk.splitlist(event.data)
                except Exception:
                    paths = [event.data]
                add_files([p.strip("{}") for p in paths])
            for widget in (page, view):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", on_drop)
                except Exception:
                    pass

        page.add_files = add_files
        return page

    # ---------- 탭 4: AI 우글거림 수정 (영상) ----------
    def build_sizzle_page():
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(page,
                 text="AI 영상의 파티클 우글거림(끓는 텍스처)을 자동 탐지해 모션블러·시간 안정화·진짜 파티클 합성으로 고쳐요. "
                      "원본 폴더에 _particlefix.mp4 로 저장돼요.",
                 bg=COL["bg"], fg=COL["sub"], font=(FONT, 10),
                 anchor="w").pack(fill="x", padx=24, pady=(12, 4))

        # 파라미터 카드
        card = tk.Frame(page, bg=COL["card"], highlightbackground=COL["border"],
                        highlightthickness=1)
        card.pack(fill="x", padx=24, pady=(2, 4))
        prow = tk.Frame(card, bg=COL["card"])
        prow.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(prow, text="파티클 타입", bg=COL["card"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left")
        from tkinter import ttk
        TYPE_LIST = [("기포 (상승)", "bubble"), ("먼지 (표류)", "dust"),
                     ("눈 (낙하)", "snow"), ("불씨 (상승)", "ember"),
                     ("자동 감지", "auto")]
        type_cb = ttk.Combobox(prow, values=[k for k, _ in TYPE_LIST],
                               state="readonly", style="Np.TCombobox",
                               font=(FONT, 10), width=14)
        type_cb.current(0)
        type_cb.pack(side="left", padx=(8, 0))

        # 강도 프리셋 1·2·3 — 누르면 아래 슬라이더가 검증된 조합으로 세팅됨
        tk.Label(prow, text="강도", bg=COL["card"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left", padx=(16, 6))
        STR_PRESETS = [
            ("1 · 약하게", {"blend": 30, "blur": 25, "density": 60, "grain": 50}),
            ("2 · 표준",   {"blend": 50, "blur": 40, "density": 100, "grain": 80}),
            ("3 · 강하게", {"blend": 75, "blur": 60, "density": 150, "grain": 120}),
        ]
        str_sel = {"idx": 1}
        swidgets = {}
        SWp, SHp, SPp = 88, 28, 4
        str_cv = tk.Canvas(prow, width=SWp * 3 + SPp * 2, height=SHp + SPp * 2 + 2,
                           bg=COL["card"], highlightthickness=0, cursor="hand2")
        str_cv.pack(side="left")

        def draw_strength():
            str_cv.delete("all")
            rrect(str_cv, 0, 1, SWp * 3 + SPp * 2, 1 + SHp + SPp * 2, 11,
                  fill=COL["card2"], outline=COL["border"])
            for ii, (nm, _) in enumerate(STR_PRESETS):
                x0 = SPp + ii * SWp
                if ii == str_sel["idx"]:
                    rrect(str_cv, x0, 1 + SPp, x0 + SWp, 1 + SPp + SHp, 9,
                          fill=COL["lime"], outline="")
                    fg = "#101204"
                else:
                    fg = COL["sub"]
                str_cv.create_text(x0 + SWp / 2, 1 + SPp + SHp / 2, text=nm,
                                   fill=fg, font=(FONT, 9, "bold"))
        draw_strength()

        def on_str_click(e):
            ii = min(2, max(0, int((e.x - SPp) // SWp)))
            str_sel["idx"] = ii
            draw_strength()
            for k, v in STR_PRESETS[ii][1].items():
                svars[k].set(v)
                vl_, dr_ = swidgets[k]
                vl_.configure(text=f"{v}%")
                dr_()
        str_cv.bind("<Button-1>", on_str_click)

        sgrid = tk.Frame(card, bg=COL["card"])
        sgrid.pack(fill="x", padx=14, pady=(4, 10))
        sgrid.columnconfigure(0, weight=1)
        sgrid.columnconfigure(1, weight=1)
        svars = {}

        def mini_slider(parent, label, key, default):
            holder = tk.Frame(parent, bg=COL["card"])
            head = tk.Frame(holder, bg=COL["card"])
            head.pack(fill="x")
            tk.Label(head, text=label, bg=COL["card"], fg=COL["text"],
                     font=(FONT, 9)).pack(side="left")
            vl = tk.Label(head, text=f"{default}%", bg=COL["card"],
                          fg=COL["lime"], font=(MONO, 9))
            vl.pack(side="right")
            var = tk.IntVar(value=default)
            cv = tk.Canvas(holder, height=22, bg=COL["card"],
                           highlightthickness=0, cursor="hand2")
            stt = {"hover": False, "press": False}
            PADX = 9

            def draw(_=None):
                cv.delete("all")
                w = max(40, cv.winfo_width())
                ty, tr = 11, 3
                rrect(cv, PADX, ty - tr, w - PADX, ty + tr, tr,
                      fill=COL["card2"], outline="")
                frac = var.get() / 200.0
                hx = PADX + frac * (w - PADX * 2)
                if hx - PADX > 4:
                    rrect(cv, PADX, ty - tr, hx, ty + tr, tr,
                          fill="#5f7030", outline="")
                r = 8 if (stt["hover"] or stt["press"]) else 6
                color = COL["lime2"] if stt["press"] else COL["lime"]
                cv.create_oval(hx - r, ty - r, hx + r, ty + r, fill=color, outline="")
            cv.bind("<Configure>", draw)

            def setv(x):
                w = max(40, cv.winfo_width())
                v = int(round(min(1.0, max(0.0, (x - PADX) / max(1, w - PADX * 2))) * 200))
                var.set(v)
                vl.configure(text=f"{v}%")
                draw()
                if str_sel["idx"] != -1:   # 수동 조절 → 강도 프리셋 선택 해제
                    str_sel["idx"] = -1
                    draw_strength()
            cv.bind("<Button-1>", lambda e: (stt.update(press=True), setv(e.x)))
            cv.bind("<B1-Motion>", lambda e: setv(e.x))
            cv.bind("<ButtonRelease-1>", lambda e: (stt.update(press=False), draw()))
            cv.bind("<Enter>", lambda e: (stt.update(hover=True), draw()))
            cv.bind("<Leave>", lambda e: (stt.update(hover=False), draw()))
            cv.pack(fill="x")
            svars[key] = var
            swidgets[key] = (vl, draw)
            return holder

        mini_slider(sgrid, "잔상 억제(시간 안정화)", "blend", 50).grid(
            row=0, column=0, sticky="ew", padx=(0, 10))
        mini_slider(sgrid, "모션블러", "blur", 40).grid(
            row=0, column=1, sticky="ew")
        mini_slider(sgrid, "파티클 밀도", "density", 100).grid(
            row=1, column=0, sticky="ew", padx=(0, 10), pady=(4, 0))
        mini_slider(sgrid, "추가 그레인", "grain", 80).grid(
            row=1, column=1, sticky="ew", pady=(4, 0))

        # 드롭존
        drop = tk.Canvas(page, bg=COL["bg"], highlightthickness=0, height=UIS(150))
        drop.pack(fill="both", expand=True, padx=24, pady=(4, 0))
        pick_btn_box = {"bbox": (0, 0, 0, 0)}

        def draw_drop(_=None):
            drop.delete("all")
            w = drop.winfo_width() or 900
            h = drop.winfo_height() or 150
            rrect(drop, 2, 2, w - 2, h - 2, 18, fill=COL["bg"],
                  outline=COL["dash"], width=2)
            cx, cy = w / 2, h / 2
            head = "영상(mp4 등)을 여기에 던져 넣으세요" if dnd_ok else "파일 선택 버튼으로 영상을 추가하세요"
            drop.create_text(cx, cy - 22, text=head, fill=COL["text"],
                             font=(FONT, 13, "bold"))
            bw, bh = 200, 38
            bx0, by0 = cx - bw / 2, cy + 2
            rrect(drop, bx0, by0, bx0 + bw, by0 + bh, 10, fill=COL["bg"],
                  outline=COL["dash"], width=1)
            drop.create_text(cx, by0 + bh / 2, text="영상 선택 · 여러 개 가능",
                             fill=COL["text"], font=(FONT, 10, "bold"))
            pick_btn_box["bbox"] = (bx0, by0, bx0 + bw, by0 + bh)
        drop.bind("<Configure>", draw_drop)

        # 첨부/로그/하단 — build_page와 동일 패턴
        att_row = tk.Frame(page, bg=COL["bg"])
        att_row.pack(fill="x", padx=24, pady=(8, 0))
        att_label = tk.Label(att_row, text="첨부 0개", bg=COL["bg"], fg=COL["sub"],
                             font=(FONT, 10), width=8, anchor="w")
        att_label.pack(side="left", anchor="n", pady=14)
        thumbs_cv = tk.Canvas(att_row, bg=COL["bg"], highlightthickness=0, height=72)
        thumbs_cv.pack(side="left", fill="x", expand=True)
        thumbs_in = tk.Frame(thumbs_cv, bg=COL["bg"])
        thumbs_cv.create_window((0, 0), window=thumbs_in, anchor="nw")
        thumbs_in.bind("<Configure>", lambda e: thumbs_cv.configure(
            scrollregion=thumbs_cv.bbox("all") or (0, 0, 0, 0)))
        thumbs_cv.bind("<MouseWheel>",
                       lambda e: thumbs_cv.xview_scroll(-1 if e.delta > 0 else 1, "units"))

        log_card = tk.Frame(page, bg=COL["card"], highlightbackground=COL["border"],
                            highlightthickness=1)
        log_card.pack(fill="x", padx=24, pady=(8, 0))
        log_head = tk.Frame(log_card, bg=COL["card"])
        log_head.pack(fill="x", padx=14, pady=(8, 2))
        tk.Label(log_head, text="처리 로그", bg=COL["card"], fg=COL["text"],
                 font=(FONT, 10, "bold")).pack(side="left")
        ok_label = tk.Label(log_head, text="", bg=COL["card"], fg=COL["lime"],
                            font=(FONT, 10, "bold"))
        ok_label.pack(side="right")
        log = tk.Text(log_card, bg=COL["card"], fg=COL["text"], bd=0,
                      highlightthickness=0, height=4, state="disabled",
                      font=(MONO, 10))
        log.pack(fill="x", padx=14, pady=(0, 8))
        for tag, c in [("ok", COL["ok"]), ("name", COL["text"]),
                       ("out", COL["dim"]), ("err", COL["err"])]:
            log.tag_configure(tag, foreground=c)

        bottom = tk.Frame(page, bg=COL["bg"])
        bottom.pack(fill="x", padx=24, pady=(8, 16))
        start_btn = canvas_button(bottom, "▶  변환 시작", 170, 48, primary=True)
        start_btn.pack(side="right")
        status = tk.Label(bottom, text="대기 중", bg=COL["bg"], fg=COL["sub"],
                          font=(FONT, 10), anchor="e")
        status.pack(side="right", padx=(0, 14), pady=10)
        pbar = progress_bar(bottom)
        pbar.pack(side="left", fill="x", expand=True, padx=(0, 14), pady=20)

        def set_status(msg):
            status.configure(text=msg)

        def append_log(parts):
            log.configure(state="normal")
            for text, tag in parts:
                log.insert("end", text, tag)
            log.see("end")
            log.configure(state="disabled")

        jobs = queue.Queue()
        counter = {"done": 0, "fail": 0}
        batch = {"total": 0, "done": 0}
        seen = set()
        pending = []
        thumbs = {}

        def thumb_photo(p):
            try:
                cap_t = cv2.VideoCapture(p)
                ok_t, im = cap_t.read()
                cap_t.release()
                if not ok_t:
                    return None
                tw, th = 84, 60
                ih, iw = im.shape[:2]
                s = max(tw / iw, th / ih)
                im = cv2.resize(im, (max(1, int(iw * s)), max(1, int(ih * s))),
                                interpolation=cv2.INTER_AREA)
                ih, iw = im.shape[:2]
                x0, y0 = (iw - tw) // 2, (ih - th) // 2
                im = im[y0:y0 + th, x0:x0 + tw]
                ok2, buf = cv2.imencode(".png", im)
                return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
            except Exception:
                return None

        def add_thumb(p):
            holder = tk.Frame(thumbs_in, bg=COL["card2"],
                              highlightbackground=COL["border"], highlightthickness=1)
            holder.pack(side="left", padx=(0, 10))
            ph = thumb_photo(p)
            if ph is not None:
                tk.Label(holder, image=ph, bg=COL["card2"], bd=0).pack()
            else:
                tk.Label(holder, text="🎬", width=10, height=4,
                         bg=COL["card2"], fg=COL["dim"]).pack()
            name = os.path.basename(p)
            if len(name) > 14:
                name = name[:11] + "…"
            tk.Label(holder, text=name, bg=COL["card2"], fg=COL["dim"],
                     font=(MONO, 7)).pack(fill="x")
            thumbs[p] = (holder, ph)
            att_label.configure(text=f"첨부 {len(thumbs)}개")

        def remove_thumb(p):
            item = thumbs.pop(p, None)
            if item:
                item[0].destroy()
            seen.discard(os.path.normcase(os.path.abspath(p)))
            att_label.configure(text=f"첨부 {len(thumbs)}개")

        def worker():
            while True:
                p, prm = jobs.get()
                name = os.path.basename(p)
                try:
                    def prog(v, name=name):
                        root.after(0, set_status,
                                   f"처리 중: {name}  {v}% · 대기 {jobs.qsize()}개")
                        frac = (batch["done"] + v / 100.0) / max(1, batch["total"])
                        root.after(0, pbar.set_frac, frac)
                    root.after(0, set_status, f"처리 중: {name} · 대기 {jobs.qsize()}개")
                    dst = fix_sizzle_video(p, ptype=prm["type"],
                                           k_blend=prm["blend"], k_blur=prm["blur"],
                                           k_dens=prm["density"], k_grain=prm["grain"],
                                           progress=prog)
                    counter["done"] += 1
                    root.after(0, append_log, [("✓ ", "ok"), (name + "\n", "name"),
                                               ("   →  " + os.path.basename(dst) + "\n", "out")])
                except Exception as e:
                    counter["fail"] += 1
                    root.after(0, append_log, [("✗ ", "err"),
                                               (f"{name} 실패: {e}\n", "err")])
                finally:
                    batch["done"] += 1
                    root.after(0, pbar.set_frac, batch["done"] / max(1, batch["total"]))
                    root.after(0, remove_thumb, p)
                    jobs.task_done()
                    msg = f"● 성공 {counter['done']}개"
                    if counter["fail"]:
                        msg += f" · 실패 {counter['fail']}개"
                    root.after(0, ok_label.configure, {"text": msg})
                    if jobs.empty():
                        def finish():
                            set_status("완료")
                            pbar.set_frac(0.0)
                            batch["total"] = batch["done"] = 0
                            start_btn.set_text("▶  변환 시작")
                            start_btn.set_enabled(True)
                        root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

        def add_files(files):
            files = [f for f in files if f.lower().endswith(VEXTS) and os.path.isfile(f)]
            n_new = 0
            for p in files:
                key = os.path.normcase(os.path.abspath(p))
                if key in seen:
                    continue
                seen.add(key)
                n_new += 1
                pending.append(p)
                add_thumb(p)
            if n_new:
                set_status(f"{len(pending)}개 대기 중 — [▶ 변환 시작]을 누르면 처리합니다 "
                           f"(영상당 수 분 소요)")

        def start_convert(_=None):
            if not start_btn.is_enabled():
                return
            if not pending:
                set_status("첨부된 대기 영상이 없어요 — 먼저 던져 넣으세요")
                return
            prm = {"type": dict(TYPE_LIST)[type_cb.get()],
                   "blend": svars["blend"].get() / 100.0,
                   "blur": svars["blur"].get() / 100.0,
                   "density": svars["density"].get() / 100.0,
                   "grain": svars["grain"].get() / 100.0}
            batch["total"] += len(pending)
            for p in pending:
                jobs.put((p, prm))
            pending.clear()
            start_btn.set_text("⏳  변환 중…")
            start_btn.set_enabled(False)
            pbar.set_frac(0.0)

        def pick_files():
            files = filedialog.askopenfilenames(
                title="영상 선택",
                filetypes=[("영상", "*.mp4 *.mov *.avi *.mkv *.webm"),
                           ("모든 파일", "*.*")])
            add_files(list(files))

        def on_drop_click(e):
            x0, y0, x1, y1 = pick_btn_box["bbox"]
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                pick_files()
        drop.bind("<Button-1>", on_drop_click)
        start_btn.bind("<Button-1>", start_convert)

        if dnd_ok:
            def on_drop(event):
                try:
                    paths = root.tk.splitlist(event.data)
                except Exception:
                    paths = [event.data]
                add_files([p.strip("{}") for p in paths])
            for widget in (page, drop, log, thumbs_cv):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", on_drop)
                except Exception:
                    pass

        page.add_files = add_files
        return page

    def build_detailboost_page():
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(page,
                 text="AI 풀샷의 뭉개진 디테일을 살리고(하늘·구름은 자동 보호) 바닥 텍스처 우글거림과 "
                      "카메라 끊김(저더)을 함께 보정해요. 원본 폴더에 _detail.mp4 로 저장돼요.",
                 bg=COL["bg"], fg=COL["sub"], font=(FONT, 10),
                 anchor="w").pack(fill="x", padx=24, pady=(12, 4))

        card = tk.Frame(page, bg=COL["card"], highlightbackground=COL["border"],
                        highlightthickness=1)
        card.pack(fill="x", padx=24, pady=(2, 4))
        prow = tk.Frame(card, bg=COL["card"])
        prow.pack(fill="x", padx=14, pady=(10, 2))

        tk.Label(prow, text="강도", bg=COL["card"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left", padx=(0, 6))
        STR_PRESETS = [
            ("1 · 약하게", {"detail": 60, "lock": 80, "blur": 70,
                            "smooth": 70, "grain": 60}),
            ("2 · 표준",   {"detail": 100, "lock": 100, "blur": 100,
                            "smooth": 100, "grain": 100}),
            ("3 · 강하게", {"detail": 145, "lock": 120, "blur": 135,
                            "smooth": 100, "grain": 140}),
        ]
        str_sel = {"idx": 1}
        swidgets = {}
        SWp, SHp, SPp = 88, 28, 4
        str_cv = tk.Canvas(prow, width=SWp * 3 + SPp * 2, height=SHp + SPp * 2 + 2,
                           bg=COL["card"], highlightthickness=0, cursor="hand2")
        str_cv.pack(side="left")

        def draw_strength():
            str_cv.delete("all")
            rrect(str_cv, 0, 1, SWp * 3 + SPp * 2, 1 + SHp + SPp * 2, 11,
                  fill=COL["card2"], outline=COL["border"])
            for ii, (nm, _) in enumerate(STR_PRESETS):
                x0 = SPp + ii * SWp
                if ii == str_sel["idx"]:
                    rrect(str_cv, x0, 1 + SPp, x0 + SWp, 1 + SPp + SHp, 9,
                          fill=COL["lime"], outline="")
                    fg = "#101204"
                else:
                    fg = COL["sub"]
                str_cv.create_text(x0 + SWp / 2, 1 + SPp + SHp / 2, text=nm,
                                   fill=fg, font=(FONT, 9, "bold"))
        draw_strength()

        def on_str_click(e):
            ii = min(2, max(0, int((e.x - SPp) // SWp)))
            str_sel["idx"] = ii
            draw_strength()
            for k, v in STR_PRESETS[ii][1].items():
                svars[k].set(v)
                vl_, dr_ = swidgets[k]
                vl_.configure(text=f"{v}%")
                dr_()
        str_cv.bind("<Button-1>", on_str_click)

        sgrid = tk.Frame(card, bg=COL["card"])
        sgrid.pack(fill="x", padx=14, pady=(4, 10))
        sgrid.columnconfigure(0, weight=1)
        sgrid.columnconfigure(1, weight=1)
        svars = {}

        def mini_slider(parent, label, key, default):
            holder = tk.Frame(parent, bg=COL["card"])
            head = tk.Frame(holder, bg=COL["card"])
            head.pack(fill="x")
            tk.Label(head, text=label, bg=COL["card"], fg=COL["text"],
                     font=(FONT, 9)).pack(side="left")
            vl = tk.Label(head, text=f"{default}%", bg=COL["card"],
                          fg=COL["lime"], font=(MONO, 9))
            vl.pack(side="right")
            var = tk.IntVar(value=default)
            cv = tk.Canvas(holder, height=22, bg=COL["card"],
                           highlightthickness=0, cursor="hand2")
            stt = {"hover": False, "press": False}
            PADX = 9

            def draw(_=None):
                cv.delete("all")
                w = max(40, cv.winfo_width())
                ty, tr = 11, 3
                rrect(cv, PADX, ty - tr, w - PADX, ty + tr, tr,
                      fill=COL["card2"], outline="")
                frac = var.get() / 200.0
                hx = PADX + frac * (w - PADX * 2)
                if hx - PADX > 4:
                    rrect(cv, PADX, ty - tr, hx, ty + tr, tr,
                          fill="#5f7030", outline="")
                r = 8 if (stt["hover"] or stt["press"]) else 6
                color = COL["lime2"] if stt["press"] else COL["lime"]
                cv.create_oval(hx - r, ty - r, hx + r, ty + r, fill=color, outline="")
            cv.bind("<Configure>", draw)

            def setv(x):
                w = max(40, cv.winfo_width())
                v = int(round(min(1.0, max(0.0, (x - PADX) / max(1, w - PADX * 2))) * 200))
                var.set(v)
                vl.configure(text=f"{v}%")
                draw()
                if str_sel["idx"] != -1:
                    str_sel["idx"] = -1
                    draw_strength()
            cv.bind("<Button-1>", lambda e: (stt.update(press=True), setv(e.x)))
            cv.bind("<B1-Motion>", lambda e: setv(e.x))
            cv.bind("<ButtonRelease-1>", lambda e: (stt.update(press=False), draw()))
            cv.bind("<Enter>", lambda e: (stt.update(hover=True), draw()))
            cv.bind("<Leave>", lambda e: (stt.update(hover=False), draw()))
            cv.pack(fill="x")
            svars[key] = var
            swidgets[key] = (vl, draw)
            return holder

        mini_slider(sgrid, "디테일 강도", "detail", 100).grid(
            row=0, column=0, sticky="ew", padx=(0, 10))
        mini_slider(sgrid, "텍스처 고정(우글거림 억제)", "lock", 100).grid(
            row=0, column=1, sticky="ew")
        mini_slider(sgrid, "모션블러(끊김 완화)", "blur", 100).grid(
            row=1, column=0, sticky="ew", padx=(0, 10), pady=(4, 0))
        mini_slider(sgrid, "카메라 경로 스무딩", "smooth", 100).grid(
            row=1, column=1, sticky="ew", pady=(4, 0))
        mini_slider(sgrid, "그레인", "grain", 100).grid(
            row=2, column=0, sticky="ew", padx=(0, 10), pady=(4, 0))

        drop = tk.Canvas(page, bg=COL["bg"], highlightthickness=0, height=UIS(150))
        drop.pack(fill="both", expand=True, padx=24, pady=(4, 0))
        pick_btn_box = {"bbox": (0, 0, 0, 0)}

        def draw_drop(_=None):
            drop.delete("all")
            w = drop.winfo_width() or 900
            h = drop.winfo_height() or 150
            rrect(drop, 2, 2, w - 2, h - 2, 18, fill=COL["bg"],
                  outline=COL["dash"], width=2)
            cx, cy = w / 2, h / 2
            head = "영상(mp4 등)을 여기에 던져 넣으세요" if dnd_ok else "파일 선택 버튼으로 영상을 추가하세요"
            drop.create_text(cx, cy - 22, text=head, fill=COL["text"],
                             font=(FONT, 13, "bold"))
            bw, bh = 200, 38
            bx0, by0 = cx - bw / 2, cy + 2
            rrect(drop, bx0, by0, bx0 + bw, by0 + bh, 10, fill=COL["bg"],
                  outline=COL["dash"], width=1)
            drop.create_text(cx, by0 + bh / 2, text="영상 선택 · 여러 개 가능",
                             fill=COL["text"], font=(FONT, 10, "bold"))
            pick_btn_box["bbox"] = (bx0, by0, bx0 + bw, by0 + bh)
        drop.bind("<Configure>", draw_drop)

        att_row = tk.Frame(page, bg=COL["bg"])
        att_row.pack(fill="x", padx=24, pady=(8, 0))
        att_label = tk.Label(att_row, text="첨부 0개", bg=COL["bg"], fg=COL["sub"],
                             font=(FONT, 10), width=8, anchor="w")
        att_label.pack(side="left", anchor="n", pady=14)
        thumbs_cv = tk.Canvas(att_row, bg=COL["bg"], highlightthickness=0, height=72)
        thumbs_cv.pack(side="left", fill="x", expand=True)
        thumbs_in = tk.Frame(thumbs_cv, bg=COL["bg"])
        thumbs_cv.create_window((0, 0), window=thumbs_in, anchor="nw")
        thumbs_in.bind("<Configure>", lambda e: thumbs_cv.configure(
            scrollregion=thumbs_cv.bbox("all") or (0, 0, 0, 0)))
        thumbs_cv.bind("<MouseWheel>",
                       lambda e: thumbs_cv.xview_scroll(-1 if e.delta > 0 else 1, "units"))

        log_card = tk.Frame(page, bg=COL["card"], highlightbackground=COL["border"],
                            highlightthickness=1)
        log_card.pack(fill="x", padx=24, pady=(8, 0))
        log_head = tk.Frame(log_card, bg=COL["card"])
        log_head.pack(fill="x", padx=14, pady=(8, 2))
        tk.Label(log_head, text="처리 로그", bg=COL["card"], fg=COL["text"],
                 font=(FONT, 10, "bold")).pack(side="left")
        ok_label = tk.Label(log_head, text="", bg=COL["card"], fg=COL["lime"],
                            font=(FONT, 10, "bold"))
        ok_label.pack(side="right")
        log = tk.Text(log_card, bg=COL["card"], fg=COL["text"], bd=0,
                      highlightthickness=0, height=4, state="disabled",
                      font=(MONO, 10))
        log.pack(fill="x", padx=14, pady=(0, 8))
        for tag, c in [("ok", COL["ok"]), ("name", COL["text"]),
                       ("out", COL["dim"]), ("err", COL["err"])]:
            log.tag_configure(tag, foreground=c)

        bottom = tk.Frame(page, bg=COL["bg"])
        bottom.pack(fill="x", padx=24, pady=(8, 16))
        start_btn = canvas_button(bottom, "▶  변환 시작", 170, 48, primary=True)
        start_btn.pack(side="right")
        status = tk.Label(bottom, text="대기 중", bg=COL["bg"], fg=COL["sub"],
                          font=(FONT, 10), anchor="e")
        status.pack(side="right", padx=(0, 14), pady=10)
        pbar = progress_bar(bottom)
        pbar.pack(side="left", fill="x", expand=True, padx=(0, 14), pady=20)

        def set_status(msg):
            status.configure(text=msg)

        def append_log(parts):
            log.configure(state="normal")
            for text, tag in parts:
                log.insert("end", text, tag)
            log.see("end")
            log.configure(state="disabled")

        jobs = queue.Queue()
        counter = {"done": 0, "fail": 0}
        batch = {"total": 0, "done": 0}
        seen = set()
        pending = []
        thumbs = {}

        def thumb_photo(p):
            try:
                cap_t = cv2.VideoCapture(p)
                ok_t, im = cap_t.read()
                cap_t.release()
                if not ok_t:
                    return None
                tw, th = 84, 60
                ih, iw = im.shape[:2]
                s = max(tw / iw, th / ih)
                im = cv2.resize(im, (max(1, int(iw * s)), max(1, int(ih * s))),
                                interpolation=cv2.INTER_AREA)
                ih, iw = im.shape[:2]
                x0, y0 = (iw - tw) // 2, (ih - th) // 2
                im = im[y0:y0 + th, x0:x0 + tw]
                ok2, buf = cv2.imencode(".png", im)
                return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
            except Exception:
                return None

        def add_thumb(p):
            holder = tk.Frame(thumbs_in, bg=COL["card2"],
                              highlightbackground=COL["border"], highlightthickness=1)
            holder.pack(side="left", padx=(0, 10))
            ph = thumb_photo(p)
            if ph is not None:
                tk.Label(holder, image=ph, bg=COL["card2"], bd=0).pack()
            else:
                tk.Label(holder, text="🎬", width=10, height=4,
                         bg=COL["card2"], fg=COL["dim"]).pack()
            name = os.path.basename(p)
            if len(name) > 14:
                name = name[:11] + "…"
            tk.Label(holder, text=name, bg=COL["card2"], fg=COL["dim"],
                     font=(MONO, 7)).pack(fill="x")
            thumbs[p] = (holder, ph)
            att_label.configure(text=f"첨부 {len(thumbs)}개")

        def remove_thumb(p):
            item = thumbs.pop(p, None)
            if item:
                item[0].destroy()
            seen.discard(os.path.normcase(os.path.abspath(p)))
            att_label.configure(text=f"첨부 {len(thumbs)}개")

        def worker():
            while True:
                p, prm = jobs.get()
                name = os.path.basename(p)
                try:
                    def prog(v, name=name):
                        root.after(0, set_status,
                                   f"처리 중: {name}  {v}% · 대기 {jobs.qsize()}개")
                        frac = (batch["done"] + v / 100.0) / max(1, batch["total"])
                        root.after(0, pbar.set_frac, frac)
                    root.after(0, set_status, f"처리 중: {name} · 대기 {jobs.qsize()}개")
                    dst = detail_boost_video(p, k_detail=prm["detail"],
                                             k_lock=prm["lock"], k_blur=prm["blur"],
                                             k_smooth=prm["smooth"],
                                             k_grain=prm["grain"], progress=prog)
                    counter["done"] += 1
                    root.after(0, append_log, [("✓ ", "ok"), (name + "\n", "name"),
                                               ("   →  " + os.path.basename(dst) + "\n", "out")])
                except Exception as e:
                    counter["fail"] += 1
                    root.after(0, append_log, [("✗ ", "err"),
                                               (f"{name} 실패: {e}\n", "err")])
                finally:
                    batch["done"] += 1
                    root.after(0, pbar.set_frac, batch["done"] / max(1, batch["total"]))
                    root.after(0, remove_thumb, p)
                    jobs.task_done()
                    msg = f"● 성공 {counter['done']}개"
                    if counter["fail"]:
                        msg += f" · 실패 {counter['fail']}개"
                    root.after(0, ok_label.configure, {"text": msg})
                    if jobs.empty():
                        def finish():
                            set_status("완료")
                            pbar.set_frac(0.0)
                            batch["total"] = batch["done"] = 0
                            start_btn.set_text("▶  변환 시작")
                            start_btn.set_enabled(True)
                        root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

        def add_files(files):
            files = [f for f in files if f.lower().endswith(VEXTS) and os.path.isfile(f)]
            n_new = 0
            for p in files:
                key = os.path.normcase(os.path.abspath(p))
                if key in seen:
                    continue
                seen.add(key)
                n_new += 1
                pending.append(p)
                add_thumb(p)
            if n_new:
                set_status(f"{len(pending)}개 대기 중 — [▶ 변환 시작]을 누르면 처리합니다 "
                           f"(영상당 수 분 소요)")

        def start_convert(_=None):
            if not start_btn.is_enabled():
                return
            if not pending:
                set_status("첨부된 대기 영상이 없어요 — 먼저 던져 넣으세요")
                return
            prm = {k: svars[k].get() / 100.0
                   for k in ("detail", "lock", "blur", "smooth", "grain")}
            batch["total"] += len(pending)
            for p in pending:
                jobs.put((p, prm))
            pending.clear()
            start_btn.set_text("⏳  변환 중…")
            start_btn.set_enabled(False)
            pbar.set_frac(0.0)

        def pick_files():
            files = filedialog.askopenfilenames(
                title="영상 선택",
                filetypes=[("영상", "*.mp4 *.mov *.avi *.mkv *.webm"),
                           ("모든 파일", "*.*")])
            add_files(list(files))

        def on_drop_click(e):
            x0, y0, x1, y1 = pick_btn_box["bbox"]
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                pick_files()
        drop.bind("<Button-1>", on_drop_click)
        start_btn.bind("<Button-1>", start_convert)

        if dnd_ok:
            def on_drop(event):
                try:
                    paths = root.tk.splitlist(event.data)
                except Exception:
                    paths = [event.data]
                add_files([p.strip("{}") for p in paths])
            for widget in (page, drop, log, thumbs_cv):
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", on_drop)
                except Exception:
                    pass

        page.add_files = add_files
        return page

    def build_focus_page():
        import math
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)
        S = {"frames": None, "pv": None, "depth16": None, "pvd": None,
             "N": 0, "W": 0, "H": 0, "fps": 24.0, "pw": 0, "ph": 0,
             "focus": 0.5, "ap": 22.0, "cur": 0, "playing": False, "recording": False,
             "fc": None, "recmask": None, "src": None, "photo": None, "busy": False,
             "last_ang": None, "stk": None, "stk_key": None, "rq": False,
             "mag": None, "magcv": None, "magphoto": None, "view": None,
             "mx": 320, "my": 180, "hl": 0.0, "ftg": 0.5, "tw": False, "marks": {},
             "glide": None, "cnt": None, "keys": [], "akeys": [], "gch": "f",
             "lshape": "disc", "lca": 0.30, "lbr": 0.40}

        wrap = tk.Frame(page, bg=COL["bg"])
        wrap.pack(fill="both", expand=True, padx=22, pady=14)

        hint = tk.Label(wrap, justify="left", bg=COL["bg"], fg=COL["sub"], font=(FONT, 10),
                        text="영상을 넣으면 깊이를 인식해요. 휠(마우스 휠)로 초점을 당기고, 조리개로 "
                             "심도를 조절하세요. [녹화] 켜고 재생하며 포커스 풀 → [렌더]로 구워냅니다. "
                             "Tab: ×10 돋보기 · 1~4: 감도 기어 · Space: 재생/정지 · "
                             "Q/W/E/R: 포커스 마커(첫입력=찍기, 재입력=이동, Shift=다시찍기, Ctrl=삭제) · A/D: 조리개")
        hint.pack(anchor="w", pady=(0, 8))

        top = tk.Frame(wrap, bg=COL["bg"]); top.pack(fill="x")
        status = tk.Label(top, text="영상을 드래그하거나 [영상 열기]", bg=COL["bg"],
                          fg=COL["dim"], font=(MONO, 10)); status.pack(side="left")

        main = tk.Frame(wrap, bg=COL["bg"]); main.pack(fill="both", expand=True, pady=10)
        PVW = 640                       # DoF 연산 해상도 (고정 — 인터랙션 속도 우선)
        prev_cv = tk.Canvas(main, width=PVW, height=int(PVW * 9 / 16), bg="#0a0b0c",
                            highlightthickness=1, highlightbackground=COL["border"])
        prev_cv.pack(side="left", fill="both", expand=True)   # 표시 크기는 창 추종
        ctrl = tk.Frame(main, bg=COL["bg"]); ctrl.pack(side="left", fill="y", padx=(18, 0))

        def _blit(bgr, fast=False):
            """연산 해상도 프레임을 캔버스 크기에 맞춰 표시.
            연산(고정 저해상)과 표시(창 추종)를 분리 — 리사이즈 시 재연산 없음.
            fast=True(인터랙션 중)이며 표시가 크면 절반 해상도 + ×2 줌으로 그리고,
            입력이 멎으면 220ms 뒤 고품질로 한 번 다시 그린다."""
            cw = max(prev_cv.winfo_width(), 60)
            ch2 = max(prev_cv.winfo_height(), 40)
            sc = min((cw - 2) / bgr.shape[1], (ch2 - 2) / bgr.shape[0])
            dw = max(2, int(bgr.shape[1] * sc)); dh = max(2, int(bgr.shape[0] * sc))
            z = 1
            if fast and dw > UIS(900):
                z = 2
                dw = max(2, dw // 2); dh = max(2, dh // 2)
            disp = (bgr if (dw, dh) == (bgr.shape[1], bgr.shape[0]) else
                    cv2.resize(bgr, (dw, dh),
                               interpolation=(cv2.INTER_AREA if dw < bgr.shape[1]
                                              else cv2.INTER_LINEAR)))
            ok2, buf2 = cv2.imencode(".ppm", disp)
            if not ok2:
                return
            ph2 = tk.PhotoImage(data=buf2.tobytes())
            if z == 2:
                ph2 = ph2.zoom(2)
            S["photo"] = ph2
            S["dw"], S["dh"] = dw * z, dh * z
            S["ox"], S["oy"] = cw / 2 - S["dw"] / 2, ch2 / 2 - S["dh"] / 2
            prev_cv.delete("all")
            prev_cv.create_image(cw / 2, ch2 / 2, image=S["photo"])
            if S["cnt"]:
                prev_cv.create_text(cw / 2, ch2 / 2, text=str(S["cnt"]),
                                    fill="#ff5252", font=(FONT, 64, "bold"), tags="cntdown")
            if S.get("_hq"):
                prev_cv.after_cancel(S["_hq"]); S["_hq"] = None
            if fast and z == 2:                 # 인터랙션 멎으면 고품질 리프레시
                S["_hq"] = prev_cv.after(
                    220, lambda: (S.update(_hq=None),
                                  _blit(S["view"]) if S["view"] is not None else None))

        # ---- GDI 고속 표시 경로 — Tk 이미지 파이프라인(인코딩·파싱·복사)을 우회해
        # numpy 프레임을 캔버스 HWND에 StretchDIBits로 직접 그린다. 전체화면에서도
        # 회당 몇 ms라 작은 창과 같은 부드러움. 실패 시 기존 Tk 경로 폴백.
        import ctypes as _ct

        class _BIH(_ct.Structure):
            _fields_ = [("biSize", _ct.c_uint32), ("biWidth", _ct.c_int32),
                        ("biHeight", _ct.c_int32), ("biPlanes", _ct.c_uint16),
                        ("biBitCount", _ct.c_uint16), ("biCompression", _ct.c_uint32),
                        ("biSizeImage", _ct.c_uint32), ("biXPelsPerMeter", _ct.c_int32),
                        ("biYPelsPerMeter", _ct.c_int32), ("biClrUsed", _ct.c_uint32),
                        ("biClrImportant", _ct.c_uint32)]

        _gdi = {"ok": os.name == "nt"}
        if _gdi["ok"]:
            try:
                _u32 = _ct.windll.user32; _g32 = _ct.windll.gdi32
                _u32.GetDC.restype = _ct.c_void_p
                _u32.GetDC.argtypes = [_ct.c_void_p]
                _u32.ReleaseDC.argtypes = [_ct.c_void_p, _ct.c_void_p]
                _g32.SetStretchBltMode.argtypes = [_ct.c_void_p, _ct.c_int]
                _g32.SetBrushOrgEx.argtypes = [_ct.c_void_p, _ct.c_int, _ct.c_int,
                                               _ct.c_void_p]
                _g32.StretchDIBits.argtypes = ([_ct.c_void_p] + [_ct.c_int] * 8 +
                                               [_ct.c_void_p, _ct.c_void_p,
                                                _ct.c_uint, _ct.c_uint32])
            except Exception:
                _gdi["ok"] = False

        def _blit_gdi(bgr):
            try:
                hwnd = prev_cv.winfo_id()
                cw = max(prev_cv.winfo_width(), 8)
                ch2 = max(prev_cv.winfo_height(), 8)
                h, w = bgr.shape[:2]
                sc = min(cw / w, ch2 / h)
                dw = max(2, int(w * sc)); dh = max(2, int(h * sc))
                ox = (cw - dw) // 2; oy = (ch2 - dh) // 2
                if (w * 3) % 4:                    # DIB 행은 4바이트 정렬 필요
                    pad = 4 - (w * 3) % 4
                    data = np.zeros((h, w * 3 + pad), np.uint8)
                    data[:, :w * 3] = bgr.reshape(h, -1)
                else:
                    data = np.ascontiguousarray(bgr)
                hdc = _u32.GetDC(hwnd)
                if not hdc:
                    return False
                try:
                    _g32.SetStretchBltMode(hdc, 4)             # HALFTONE
                    _g32.SetBrushOrgEx(hdc, 0, 0, None)
                    bi = _BIH(biSize=40, biWidth=w, biHeight=-h, biPlanes=1,
                              biBitCount=24, biCompression=0)
                    _g32.StretchDIBits(hdc, ox, oy, dw, dh, 0, 0, w, h,
                                       data.ctypes.data_as(_ct.c_void_p),
                                       _ct.byref(bi), 0, 0x00CC0020)   # SRCCOPY
                finally:
                    _u32.ReleaseDC(hwnd, hdc)
                S["dw"], S["dh"] = dw, dh
                S["ox"], S["oy"] = ox, oy
                return True
            except Exception:
                return False

        def _present(bgr):
            """프레임 표시 진입점 — 카운트다운 합성 + GDI/Tk 경로 선택"""
            if S["cnt"]:
                bgr = bgr.copy()
                txt = str(S["cnt"])
                fs = max(1.5, bgr.shape[0] / 130.0)
                th = max(4, int(fs * 3))
                (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                org = ((bgr.shape[1] - tw2) // 2, (bgr.shape[0] + th2) // 2)
                cv2.putText(bgr, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (20, 20, 20), th + 6, cv2.LINE_AA)
                cv2.putText(bgr, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (82, 82, 255), th, cv2.LINE_AA)
            if _gdi["ok"] and _blit_gdi(bgr):
                return
            _gdi["ok"] = False
            _blit(bgr, fast=True)

        def _on_pv_cfg(e):
            if S["view"] is None:
                return
            if abs(e.width - S.get("_lw", 0)) + abs(e.height - S.get("_lh", 0)) > 8:
                S["_lw"], S["_lh"] = e.width, e.height
                _present(S["view"])
        prev_cv.bind("<Configure>", _on_pv_cfg)

        def _on_expose(_=None):                    # Tk가 배경을 다시 칠하면 재표시
            if S["view"] is not None and not S.get("_exq"):
                S["_exq"] = True
                prev_cv.after_idle(lambda: (S.update(_exq=False),
                                            _present(S["view"])
                                            if S["view"] is not None else None))
        prev_cv.bind("<Expose>", _on_expose)

        # ---- 포커스 휠 ----
        tk.Label(ctrl, text="FOCUS", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 10, "bold")).pack(anchor="w")
        WHEEL = UIS(168)
        wheel = tk.Canvas(ctrl, width=WHEEL, height=WHEEL, bg=COL["bg"],
                          highlightthickness=0, cursor="hand2"); wheel.pack(pady=(4, 2))
        fval = tk.Label(ctrl, text="초점 —", bg=COL["bg"], fg=COL["lime"],
                        font=(MONO, 11)); fval.pack()

        def draw_wheel():
            wheel.delete("all")
            c = WHEEL / 2; R = c - 10
            wheel.create_oval(c - R, c - R, c + R, c + R, outline=COL["border"], width=2)
            wheel.create_oval(c - R + 14, c - R + 14, c + R - 14, c + R - 14,
                              outline=COL["dash"], width=1)
            for k in range(24):
                a = math.radians(k * 15)
                r0 = R - 3 if k % 6 else R - 9
                wheel.create_line(c + r0 * math.cos(a), c + r0 * math.sin(a),
                                  c + R * math.cos(a), c + R * math.sin(a),
                                  fill=COL["dim"], width=1)
            for ch, fv in S["marks"].items():              # 포커스 마커 (테이프)
                mcol = MARKC.get(ch, COL["lime"])
                ma = math.radians(-215 + fv * 250)
                wheel.create_line(c + (R - 8) * math.cos(ma), c + (R - 8) * math.sin(ma),
                                  c + (R + 6) * math.cos(ma), c + (R + 6) * math.sin(ma),
                                  fill=mcol, width=3)
                wheel.create_text(c + (R - 19) * math.cos(ma), c + (R - 19) * math.sin(ma),
                                  text=ch.upper(), fill=mcol, font=(MONO, 8, "bold"))
            ang = math.radians(-215 + S["ftg"] * 250)     # 250도 스윕 (목표값 즉시 반영)
            wheel.create_line(c, c, c + (R - 16) * math.cos(ang), c + (R - 16) * math.sin(ang),
                              fill=COL["lime"], width=3)
            wheel.create_oval(c + (R - 16) * math.cos(ang) - 6, c + (R - 16) * math.sin(ang) - 6,
                              c + (R - 16) * math.cos(ang) + 6, c + (R - 16) * math.sin(ang) + 6,
                              fill=COL["lime"], outline="")
            wheel.create_oval(c - 5, c - 5, c + 5, c + 5, fill=COL["sub"], outline="")

        def tween_tick():
            """포커스가 목표값(ftg)으로 관성 있게 따라감 — 시간 기반 지수 감쇠라
            렌더가 느려도 벽시계 기준 등속(끊김 없음)이고, 통과 속도가 느려져
            초점면을 지날 때의 플래시('갔다 돌아오는 느낌')가 여러 프레임에 퍼진다."""
            now = time.perf_counter()
            dt = min(0.1, now - S.get("tw_t", now - 0.03))
            S["tw_t"] = now
            g = S["glide"]
            if g is not None:                  # 마커 랙: 스무스스텝 글라이드 (가속-순항-감속)
                f0, f1, t0, dur = g
                p = min(1.0, (now - t0) / dur)
                S["focus"] = f0 + (f1 - f0) * p * p * (3.0 - 2.0 * p)
                if p >= 1.0:
                    S["glide"] = None
                    S["tw"] = abs(S["ftg"] - S["focus"]) >= 0.002
                    if S["recording"]:
                        add_key(S["cur"], S["focus"])   # 랙 도착 키프레임
            else:
                d = S["ftg"] - S["focus"]
                if abs(d) < 0.002:
                    S["focus"] = S["ftg"]
                    S["tw"] = False
                else:
                    S["focus"] += d * (1.0 - math.exp(-dt / 0.18))
            if S["recording"] and S["fc"] is not None:
                S["fc"][S["cur"]] = S["focus"]; S["recmask"][S["cur"]] = True
            request_render()
            if S["tw"]:
                root.after(30, tween_tick)

        def set_focus(f):
            S["glide"] = None                  # 휠 개입 시 마커 글라이드 즉시 해제
            S["ftg"] = max(0.0, min(1.0, f))
            near = S["ftg"]
            fval.configure(text=f"초점 {near*100:4.0f}%  ({'가까움' if near>0.6 else '멀리' if near<0.4 else '중간'})")
            draw_wheel()
            if not S["tw"]:
                S["tw"] = True
                tween_tick()

        def wheel_drag(e):
            c = WHEEL / 2
            ang = math.degrees(math.atan2(e.y - c, e.x - c))
            if S["last_ang"] is not None:
                d = ang - S["last_ang"]
                if d > 180: d -= 360
                if d < -180: d += 360
                set_focus(S["ftg"] + d / 250.0)
            S["last_ang"] = ang

        def wheel_release(_):
            S["last_ang"] = None
        wheel.bind("<Button-1>", lambda e: (wheel.focus_set(), wheel_drag(e)))
        wheel.bind("<B1-Motion>", wheel_drag)
        wheel.bind("<ButtonRelease-1>", wheel_release)

        # ---- 휠 감도 ----
        sens_var = tk.DoubleVar(value=3.0)   # 휠 한 칸당 초점 이동 % (20이면 한 번 휙에 끝↔끝)
        tk.Label(ctrl, text="휠 감도", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(8, 0))
        tk.Scale(ctrl, from_=1, to=20, resolution=0.5, orient="horizontal",
                 variable=sens_var, showvalue=True, length=UIS(168), bg=COL["card"],
                 fg=COL["text"], troughcolor=COL["card2"], highlightthickness=0,
                 bd=0, font=(MONO, 8)).pack()
        tk.Label(ctrl, text="← 미세 조절          한 번에 휙 →", bg=COL["bg"],
                 fg=COL["dim"], font=(MONO, 8)).pack()

        def on_scroll(e):
            notches = e.delta / 120.0        # 고해상도 휠·트랙패드도 비례 반영
            set_focus(S["ftg"] + notches * sens_var.get() / 100.0)
            return "break"
        for w in (prev_cv, wheel):
            w.bind("<MouseWheel>", on_scroll)

        # ---- 조리개 ----
        tk.Label(ctrl, text="APERTURE (심도)", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(12, 0))
        ap_var = tk.DoubleVar(value=S["ap"])

        def on_ap(v):
            S["ap"] = float(v); request_render()
        ap_scale = tk.Scale(ctrl, from_=0, to=100, orient="horizontal", variable=ap_var,
                            command=on_ap, showvalue=True, length=UIS(168), bg=COL["card"],
                            fg=COL["text"], troughcolor=COL["card2"], highlightthickness=0,
                            bd=0, font=(MONO, 8)); ap_scale.pack()
        tk.Label(ctrl, text="← 깊은 심도        얕은 심도(보케) →", bg=COL["bg"],
                 fg=COL["dim"], font=(MONO, 8)).pack()

        # ---- 보케 하이라이트 (광원용, 0=색 그대로) ----
        tk.Label(ctrl, text="보케 하이라이트", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(8, 0))
        hl_var = tk.DoubleVar(value=0.0)

        def on_hl(v):
            S["hl"] = float(v) / 100.0
            S["stk_key"] = None                # 작업공간 바뀜 → 슬라이스 캐시 무효화
            request_render()
        tk.Scale(ctrl, from_=0, to=100, orient="horizontal", variable=hl_var,
                 command=on_hl, showvalue=True, length=UIS(168), bg=COL["card"],
                 fg=COL["text"], troughcolor=COL["card2"], highlightthickness=0,
                 bd=0, font=(MONO, 8)).pack()
        tk.Label(ctrl, text="0 = 색 보존 · 광원 장면만 올리기", bg=COL["bg"],
                 fg=COL["dim"], font=(MONO, 8)).pack()

        # ---- 녹화 / 렌더 버튼 ----
        btns = tk.Frame(ctrl, bg=COL["bg"]); btns.pack(pady=(16, 0))
        lamp = tk.Canvas(btns, width=14, height=34, bg=COL["bg"], highlightthickness=0)
        lamp.pack(side="left")
        lamp_id = lamp.create_oval(2, 13, 12, 23, fill=COL["card2"], outline=COL["border"])

        def set_lamp(color):
            lamp.itemconfigure(lamp_id, fill=color)
        rec_btn = canvas_button(btns, "● 녹화", 80, 34, primary=False)
        rec_btn.pack(side="left", padx=(0, 6))
        render_btn = canvas_button(btns, "렌더", 80, 34, primary=True)
        render_btn.pack(side="left")

        def _cnt_tick():
            """3-2-1 카운트 — 손 준비 시간. 카운트 동안은 recording=False라
            초점 조작이 커브에 스탬프되지 않는다(구울 영상에 미포함)."""
            if not S["cnt"]:
                return
            prev_cv.delete("cntdown")
            if S["view"] is not None:              # 카운트다운 숫자는 프레임에 합성
                _present(S["view"])
            rec_btn.set_text(f"●  {S['cnt']} ")

            def nxt():
                if not S["cnt"]:
                    return
                S["cnt"] -= 1
                if S["cnt"] <= 0:
                    S["cnt"] = None
                    prev_cv.delete("cntdown")
                    S["cur"] = 0; pos_var.set(0)   # 한 테이크 = 0프레임부터 전체
                    S["recording"] = True
                    S["keys"] = [{"f": 0, "v": float(S["ftg"])}]
                    rec_btn.set_text("■ REC")
                    set_lamp("#ff3b30")            # 빨간불
                    set_focus(S["ftg"])            # 시작 프레임 스탬프
                    if not S["playing"]:
                        toggle_play()              # 자동 재생
                else:
                    _cnt_tick()
            root.after(1000, nxt)

        def toggle_rec():
            if S["frames"] is None:
                return
            if S["cnt"]:                           # 카운트 중 재클릭 = 취소
                S["cnt"] = None
                prev_cv.delete("cntdown")
                rec_btn.set_text("● 녹화")
                set_lamp(COL["card2"])
                return
            if S["recording"]:                     # 녹화 종료
                S["recording"] = False
                rec_btn.set_text("● 녹화")
                set_lamp(COL["card2"])
                return
            S["cnt"] = 3                           # 3-2-1 후 녹화+재생 시작
            set_lamp("#ffd93d")
            _cnt_tick()
        rec_btn.bind("<Button-1>", lambda e: toggle_rec())

        # ---- 타임라인 / 재생 ----
        tl = tk.Frame(wrap, bg=COL["bg"]); tl.pack(fill="x", pady=(8, 0))
        play_btn = canvas_button(tl, "▶", 40, 30, primary=False); play_btn.pack(side="left")
        pos_var = tk.IntVar(value=0)

        def on_scrub(v):
            if S["frames"] is None:
                return
            S["cur"] = int(float(v))
            if S["recording"]:
                S["fc"][S["cur"]] = S["focus"]; S["recmask"][S["cur"]] = True
            request_render()
        tl_scale = tk.Scale(tl, from_=0, to=0, orient="horizontal", variable=pos_var,
                            command=on_scrub, showvalue=False, length=UIS(560), bg=COL["card"],
                            fg=COL["text"], troughcolor=COL["card2"], highlightthickness=0,
                            bd=0); tl_scale.pack(side="left", fill="x", expand=True, padx=8)
        frm_lbl = tk.Label(tl, text="0/0", bg=COL["bg"], fg=COL["dim"],
                           font=(MONO, 9)); frm_lbl.pack(side="left")

        # ---- 포커스/조리개 그래프 (프리미어식 키프레임 에디터) ----
        GPAD = 14
        grow = tk.Frame(wrap, bg=COL["bg"]); grow.pack(fill="x", pady=(6, 0))
        gch_btn = canvas_button(grow, "초점", 64, 26, primary=False)
        gch_btn.pack(side="left", padx=(0, 6), anchor="n")

        def toggle_gch():
            S["gch"] = "a" if S["gch"] == "f" else "f"
            gch_btn.set_text("초점" if S["gch"] == "f" else "조리개")
            draw_graph()
        gch_btn.bind("<Button-1>", lambda e: toggle_gch())
        graph = tk.Canvas(grow, height=90, bg="#0a0b0c", highlightthickness=1,
                          highlightbackground=COL["border"])
        graph.pack(side="left", fill="x", expand=True)
        gdrag = {"i": None}

        def ap_curve():
            """조리개 커브 (0..100). akeys 없으면 슬라이더 상수."""
            if not S["akeys"]:
                return np.full(S["N"], S["ap"], np.float32)
            ks = S["akeys"]
            arr = np.full(S["N"], ks[0]["v"], np.float32)
            for a, b in zip(ks[:-1], ks[1:]):
                if b["f"] > a["f"]:
                    t = np.linspace(0.0, 1.0, b["f"] - a["f"] + 1)
                    arr[a["f"]:b["f"] + 1] = a["v"] + (b["v"] - a["v"]) * (t * t * (3 - 2 * t))
            arr[ks[-1]["f"]:] = ks[-1]["v"]
            return arr

        def cur_curve():
            if S["recmask"] is not None and S["recmask"].any():
                idx = np.where(S["recmask"])[0]
                return np.interp(np.arange(S["N"]), idx, S["fc"][idx]).astype(np.float32)
            return np.full(S["N"], S["ftg"], np.float32)

        def g_xy(i, v, gw, gh):
            x = GPAD + (gw - 2 * GPAD) * (i / max(S["N"] - 1, 1))
            y = 8 + (gh - 16) * (1.0 - v)
            return x, y

        def draw_graph():
            graph.delete("all")
            if S["frames"] is None:
                return
            gw = max(graph.winfo_width(), 60); gh = max(graph.winfo_height(), 40)
            fa = S["gch"] == "f"
            # 조리개 커브 (주황) — 비활성 채널은 흐리게
            ac = ap_curve() / 100.0
            pts = []
            for i in range(S["N"]):
                x, y = g_xy(i, float(ac[i]), gw, gh)
                pts += [x, y]
            if len(pts) >= 4 and (S["akeys"] or not fa):
                graph.create_line(*pts, fill="#ff9f43" if not fa else "#5a3a20", width=2)
            # 초점 커브 (라임)
            c = cur_curve()
            pts = []
            for i in range(S["N"]):
                x, y = g_xy(i, float(c[i]), gw, gh)
                pts += [x, y]
            if len(pts) >= 4:
                graph.create_line(*pts, fill=COL["lime"] if fa else "#3f5220", width=2)
            for j, kf in enumerate(S["keys"]):
                x, y = g_xy(kf["f"], kf["v"], gw, gh)
                col = "#ffd93d" if fa else "#6a5a20"
                graph.create_polygon(x, y - 5, x + 5, y, x, y + 5, x - 5, y,
                                     fill=col, outline="#0a0b0c")
            for j, kf in enumerate(S["akeys"]):
                x, y = g_xy(kf["f"], kf["v"] / 100.0, gw, gh)
                col = "#ff9f43" if not fa else "#6a4423"
                graph.create_rectangle(x - 4, y - 4, x + 4, y + 4,
                                       fill=col, outline="#0a0b0c")
            x, _ = g_xy(S["cur"], 0, gw, gh)
            graph.create_line(x, 2, x, gh - 2, fill=COL["dim"])
            lab = ("가까움", "멀리") if fa else ("얕은 심도", "깊은 심도")
            graph.create_text(GPAD + 2, 9, text=lab[0], anchor="w", fill=COL["dim"], font=(MONO, 7))
            graph.create_text(GPAD + 2, gh - 9, text=lab[1], anchor="w", fill=COL["dim"], font=(MONO, 7))

        def rebuild_segments(j):
            """키프레임 j 이웃 구간을 스무스스텝으로 재생성 (프리미어식 재보간)"""
            ks = S["keys"]

            def seg(af, av, bf, bv):
                if bf <= af:
                    return
                t = np.linspace(0.0, 1.0, bf - af + 1)
                S["fc"][af:bf + 1] = av + (bv - av) * (t * t * (3 - 2 * t))
                S["recmask"][af:bf + 1] = True
            kf = ks[j]
            if j > 0:
                seg(ks[j - 1]["f"], ks[j - 1]["v"], kf["f"], kf["v"])
            else:
                S["fc"][:kf["f"] + 1] = kf["v"]; S["recmask"][:kf["f"] + 1] = True
            if j < len(ks) - 1:
                seg(kf["f"], kf["v"], ks[j + 1]["f"], ks[j + 1]["v"])
            else:
                S["fc"][kf["f"]:] = kf["v"]; S["recmask"][kf["f"]:] = True

        def add_key(f, v):
            for kf in S["keys"]:
                if abs(kf["f"] - f) <= 2:
                    kf["v"] = float(v)
                    draw_graph()
                    return
            S["keys"].append({"f": int(f), "v": float(v)})
            S["keys"].sort(key=lambda k: k["f"])
            draw_graph()

        def g_frame_of(e):
            gw = max(graph.winfo_width(), 60)
            t = (e.x - GPAD) / max(gw - 2 * GPAD, 1)
            return int(round(min(max(t, 0.0), 1.0) * (S["N"] - 1)))

        def _klist():
            return S["keys"] if S["gch"] == "f" else S["akeys"]

        def _kv_of_y(e):
            gh = max(graph.winfo_height(), 40)
            v = float(min(max(1.0 - (e.y - 8) / max(gh - 16, 1), 0.0), 1.0))
            return v if S["gch"] == "f" else v * 100.0

        def _hit(e):
            gw = max(graph.winfo_width(), 60); gh = max(graph.winfo_height(), 40)
            for j, kf in enumerate(_klist()):
                v = kf["v"] if S["gch"] == "f" else kf["v"] / 100.0
                x, y = g_xy(kf["f"], v, gw, gh)
                if abs(e.x - x) <= 6 and abs(e.y - y) <= 6:
                    return j
            return None

        def g_press(e):
            if S["frames"] is None:
                return
            gdrag["i"] = _hit(e)
            if gdrag["i"] is None:
                S["cur"] = g_frame_of(e); pos_var.set(S["cur"]); request_render()

        def g_drag(e):
            if S["frames"] is None:
                return
            j = gdrag["i"]
            if j is None:
                S["cur"] = g_frame_of(e); pos_var.set(S["cur"]); request_render()
                return
            ks = _klist(); kf = ks[j]
            lo = ks[j - 1]["f"] + 1 if j > 0 else 0
            hi = ks[j + 1]["f"] - 1 if j < len(ks) - 1 else S["N"] - 1
            kf["f"] = int(min(max(g_frame_of(e), lo), hi))
            kf["v"] = _kv_of_y(e)
            if S["gch"] == "f":
                rebuild_segments(j)
            draw_graph()

        def g_release(e):
            if gdrag["i"] is not None:
                gdrag["i"] = None
                request_render(); save_proj()

        def g_add(e):
            """더블클릭 = 활성 채널에 키프레임 추가 (그 프레임의 현재 커브값으로)"""
            if S["frames"] is None:
                return
            f = g_frame_of(e)
            ks = _klist()
            for kf in ks:
                if abs(kf["f"] - f) <= 1:
                    return
            v = float(cur_curve()[f]) if S["gch"] == "f" else float(ap_curve()[f])
            ks.append({"f": int(f), "v": v})
            ks.sort(key=lambda k: k["f"])
            draw_graph()

        def g_del(e):
            """우클릭 = 키프레임 삭제 (이웃 사이 스무스 재보간)"""
            if S["frames"] is None:
                return
            j = _hit(e)
            if j is None:
                return
            ks = _klist()
            kf = ks.pop(j)
            if S["gch"] == "f" and ks:
                jn = min(j, len(ks) - 1)
                rebuild_segments(jn)
            draw_graph(); request_render()
        # ---- 렌즈 팩 (보케 모양 · 색수차 · 브리딩) ----
        lrow = tk.Frame(wrap, bg=COL["bg"])
        SHAPES = [("disc", "보케: 원형"), ("hex", "보케: 6날"),
                  ("oct", "보케: 8날"), ("ana", "보케: 아나모픽")]
        shp_btn = canvas_button(lrow, "보케: 원형", 110, 26, primary=False)
        shp_btn.pack(side="left", padx=(0, 12))

        def cycle_shape():
            idx = [s0 for s0, _ in SHAPES].index(S["lshape"])
            S["lshape"], label = SHAPES[(idx + 1) % len(SHAPES)]
            shp_btn.set_text(label)
            request_render()
        shp_btn.bind("<Button-1>", lambda e: cycle_shape())
        tk.Label(lrow, text="색수차", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left")
        ca_var = tk.DoubleVar(value=30.0)

        def on_ca(v):
            S["lca"] = float(v) / 100.0
            request_render()
        tk.Scale(lrow, from_=0, to=100, orient="horizontal", variable=ca_var,
                 command=on_ca, showvalue=True, length=UIS(130), bg=COL["card"],
                 fg=COL["text"], troughcolor=COL["card2"], highlightthickness=0,
                 bd=0, font=(MONO, 8)).pack(side="left", padx=(4, 14))
        tk.Label(lrow, text="브리딩", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left")
        br_var = tk.DoubleVar(value=40.0)

        def on_br(v):
            S["lbr"] = float(v) / 100.0
            request_render()
        tk.Scale(lrow, from_=0, to=100, orient="horizontal", variable=br_var,
                 command=on_br, showvalue=True, length=UIS(130), bg=COL["card"],
                 fg=COL["text"], troughcolor=COL["card2"], highlightthickness=0,
                 bd=0, font=(MONO, 8)).pack(side="left", padx=(4, 0))
        lrow.pack(fill="x", pady=(4, 0))

        graph.bind("<Button-1>", g_press)
        graph.bind("<B1-Motion>", g_drag)
        graph.bind("<ButtonRelease-1>", g_release)
        graph.bind("<Double-Button-1>", g_add)
        graph.bind("<Button-3>", g_del)
        graph.bind("<Configure>", lambda e: draw_graph())

        # ---- 포커스 프로젝트 저장/불러오기 (영상 옆 *_focus.json) ----
        def proj_path():
            return os.path.splitext(S["src"])[0] + "_focus.json" if S["src"] else None

        def save_proj():
            p = proj_path()
            if not p or S["fc"] is None:
                return
            try:
                import json
                data = {"ver": 1, "ftg": float(S["ftg"]), "ap": float(S["ap"]),
                        "hl": float(S["hl"]), "lshape": S["lshape"],
                        "lca": float(S["lca"]), "lbr": float(S["lbr"]),
                        "marks": {k: float(v) for k, v in S["marks"].items()},
                        "keys": S["keys"], "akeys": S["akeys"],
                        "fc": [round(float(v), 4) for v in S["fc"]],
                        "rec": [int(i) for i in np.where(S["recmask"])[0]]}
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception:
                pass

        def load_proj():
            p = proj_path()
            if not p or not os.path.exists(p):
                return False
            try:
                import json
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                if len(d.get("fc", [])) != S["N"]:
                    return False
                S["ftg"] = S["focus"] = float(d["ftg"])
                S["ap"] = float(d["ap"]); ap_var.set(S["ap"])
                S["hl"] = float(d["hl"]); hl_var.set(S["hl"] * 100.0)
                S["lca"] = float(d.get("lca", 0.3)); ca_var.set(S["lca"] * 100.0)
                S["lbr"] = float(d.get("lbr", 0.4)); br_var.set(S["lbr"] * 100.0)
                S["lshape"] = d.get("lshape", "disc")
                for s0, lab in SHAPES:
                    if s0 == S["lshape"]:
                        shp_btn.set_text(lab)
                S["marks"] = {k: float(v) for k, v in d.get("marks", {}).items()}
                S["keys"] = [{"f": int(k["f"]), "v": float(k["v"])} for k in d.get("keys", [])]
                S["akeys"] = [{"f": int(k["f"]), "v": float(k["v"])} for k in d.get("akeys", [])]
                S["fc"] = np.array(d["fc"], np.float32)
                S["recmask"] = np.zeros(S["N"], bool)
                for i in d.get("rec", []):
                    if 0 <= i < S["N"]:
                        S["recmask"][i] = True
                return True
            except Exception:
                return False

        def play_loop():
            if not S["playing"] or S["frames"] is None:
                return
            if S["recording"] and S["cur"] >= S["N"] - 1:
                S["recording"] = False             # 테이크 완료: 끝 프레임에서 자동 종료
                S["playing"] = False
                rec_btn.set_text("● 녹화"); set_lamp(COL["card2"]); play_btn.set_text("▶")
                status.configure(text="녹화 완료 — 아래 그래프에서 키프레임 수정 가능", fg=COL["ok"])
                draw_graph(); request_render(); save_proj()
                return
            S["cur"] = (S["cur"] + 1) % S["N"]
            if S["recording"]:
                S["fc"][S["cur"]] = S["focus"]; S["recmask"][S["cur"]] = True
            pos_var.set(S["cur"]); request_render()
            root.after(int(1000 / min(S["fps"], 20)), play_loop)

        def toggle_play():
            if S["frames"] is None:
                return
            S["playing"] = not S["playing"]
            play_btn.set_text("❚❚" if S["playing"] else "▶")
            if S["playing"]:
                play_loop()
            else:
                request_render()               # 정지 시 현재 프레임 풀 해상도로 재렌더
        play_btn.bind("<Button-1>", lambda e: toggle_play())

        # ---- 키보드: 1~4 = 휠 감도 기어(왼손 변속), Space = 재생/정지 ----
        GEARS = {"1": 1.5, "2": 4.0, "3": 8.0, "4": 14.0}

        def on_gear(e):
            if current["idx"] != 5:
                return                          # 다른 탭에선 기본 동작 유지
            g = GEARS.get(e.char)
            if g is None:
                return
            sens_var.set(g)
            status.configure(text=f"휠 감도 {e.char}단 ({g:g})", fg=COL["sub"])
            return "break"

        def on_space(e):
            if current["idx"] != 5:
                return
            toggle_play()
            return "break"
        for _k in ("1", "2", "3", "4"):
            root.bind(f"<Key-{_k}>", on_gear)
        root.bind("<space>", on_space)

        # ---- 포커스 마커 (Q/W/E/R = 무선팔로우 테이프 마킹) ----
        # 없으면 현재 초점에 찍고, 있으면 그 위치로 트윈 이동(녹화 중 랙 포커스).
        # Shift+키 = 현재 초점으로 다시 찍기, Ctrl+키 = 마커 삭제.
        MARKC = {"q": "#ff6b6b", "w": "#ffd93d", "e": "#6bcbff", "r": "#b76bff"}

        def on_mark(e):
            if current["idx"] != 5 or S["pv"] is None:
                return
            ch = e.keysym.lower()
            if ch not in MARKC:
                return
            if e.state & 0x0004:               # Ctrl = 삭제
                if ch in S["marks"]:
                    del S["marks"][ch]
                    status.configure(text=f"마커 {ch.upper()} 삭제", fg=COL["sub"])
                    draw_wheel()
            elif e.state & 0x0001 or ch not in S["marks"]:   # Shift 또는 첫 지정 = 찍기
                S["marks"][ch] = S["ftg"]
                status.configure(text=f"마커 {ch.upper()} = 초점 {S['ftg']*100:.0f}%", fg=COL["sub"])
                draw_wheel(); save_proj()
            else:                              # 재호출 = 마커로 부드럽게 랙 포커스
                f1 = S["marks"][ch]
                if S["recording"]:
                    add_key(S["cur"], S["focus"])  # 랙 시작 키프레임
                dur = min(3.0, 0.15 + abs(f1 - S["focus"]) * 6.0 / max(sens_var.get(), 0.5))
                S["glide"] = (S["focus"], f1, time.perf_counter(), dur)   # 기어(감도) 비례 속도
                S["ftg"] = f1
                fval.configure(text=f"초점 {f1*100:4.0f}%  ({'가까움' if f1>0.6 else '멀리' if f1<0.4 else '중간'})")
                draw_wheel()
                if not S["tw"]:
                    S["tw"] = True
                    tween_tick()
                status.configure(text=f"마커 {ch.upper()}로 이동", fg=COL["sub"])
            return "break"
        for _k in ("q", "w", "e", "r"):
            root.bind(f"<Key-{_k}>", on_mark)
            root.bind(f"<Key-{_k.upper()}>", on_mark)

        # ---- A/D = 조리개 키 조절 (누르고 있으면 점진 가속) ----
        apk = {"t": 0.0, "n": 0}

        def on_apkey(e):
            if current["idx"] != 5:
                return
            ch = e.keysym.lower()
            if ch not in ("a", "d"):
                return
            now = time.perf_counter()
            apk["n"] = apk["n"] + 1 if now - apk["t"] < 0.15 else 0   # 연타/홀드 감지
            apk["t"] = now
            step = min(8.0, 1.0 + apk["n"] * 0.5)   # 한 번 탭=정확히 1칸, 홀드 시 가속
            v = float(round(min(100.0, max(0.0, ap_var.get() + (step if ch == "d" else -step)))))
            ap_var.set(v)
            on_ap(v)
            return "break"
        for _k in ("a", "d", "A", "D"):
            root.bind(f"<Key-{_k}>", on_apkey)

        # ---- 미리보기 렌더 (블러 스택 캐시 + 렌더 요청 병합) ----
        def _pv_slices(i):
            key = (i, round(S["hl"], 2))       # 프레임/하이라이트 바뀔 때만 재분해
            if S["stk_key"] != key:
                S["stk"] = dof_slices(S["pv"][i], S["pvd"][i], S["hl"])
                S["stk_key"] = key
            return S["stk"]

        def render_preview():
            if S["pv"] is None:
                return
            i = S["cur"]
            img, dp = S["pv"][i], S["pvd"][i]
            lo = S["playing"]                  # 재생 중엔 저해상(384) 프리뷰 — 매 프레임
            if lo:                             # 새 분해가 필요해 640으론 20fps가 안 나옴
                plw = 384
                plh = max(2, int(round(plw * S["ph"] / S["pw"])))
                img = cv2.resize(img, (plw, plh), interpolation=cv2.INTER_AREA)
                if isinstance(dp, tuple):
                    dp = (cv2.resize(dp[0], (plw, plh), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(dp[1], (plw, plh), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(dp[2], (plw, plh), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(dp[3], (plw, plh), interpolation=cv2.INTER_LINEAR))
                else:
                    dp = cv2.resize(dp, (plw, plh), interpolation=cv2.INTER_NEAREST)
            ap_now = float(ap_curve()[i]) if S["akeys"] else S["ap"]
            ap_px = (ap_now / 100.0 * 0.04) * img.shape[1]
            if ap_px < 0.6:
                out = img
            elif lo:
                out = dof_apply(img, dp, S["focus"], ap_px, fast=True,
                                shape=S["lshape"], ca=S["lca"], breath=S["lbr"])
            else:
                out = dof_apply(img, dp, S["focus"], ap_px,
                                slices=_pv_slices(i), fast=True,
                                shape=S["lshape"], ca=S["lca"], breath=S["lbr"])
            bgr = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            if lo:
                bgr = cv2.resize(bgr, (S["pw"], S["ph"]), interpolation=cv2.INTER_LINEAR)
            S["view"] = bgr                                  # 돋보기 소스 (연산 해상도)
            _present(bgr)
            frm_lbl.configure(text=f"{i+1}/{S['N']}")
            draw_graph()
            update_mag()

        def request_render():
            if S["rq"]:                        # 연속 휠/드래그 이벤트 병합
                return
            S["rq"] = True

            def _go():
                S["rq"] = False
                render_preview()
            root.after(15, _go)

        # ---- 돋보기 (Tab 토글, 마우스 위치 ×10) ----
        MAGX = 10

        def update_mag():
            m = S["mag"]
            if m is None or not m.winfo_exists() or S["view"] is None:
                return
            mcv = S["magcv"]
            mw = max(mcv.winfo_width(), 40); mh = max(mcv.winfo_height(), 30)
            view = S["view"]; vh, vw = view.shape[:2]
            sw = max(4, round(mw / MAGX)); sh = max(3, round(mh / MAGX))
            sw = min(sw, vw); sh = min(sh, vh)
            cx = min(max(int(S["mx"]), 0), vw - 1)
            cy = min(max(int(S["my"]), 0), vh - 1)
            x0 = min(max(cx - sw // 2, 0), vw - sw)
            y0 = min(max(cy - sh // 2, 0), vh - sh)
            big = cv2.resize(view[y0:y0 + sh, x0:x0 + sw], (sw * MAGX, sh * MAGX),
                             interpolation=cv2.INTER_NEAREST)
            ok, buf = cv2.imencode(".ppm", big)
            if not ok:
                return
            S["magphoto"] = tk.PhotoImage(data=buf.tobytes())
            mcv.delete("all")
            mcv.create_image(mw / 2, mh / 2, image=S["magphoto"])

        def close_mag():
            if S["mag"] is not None and S["mag"].winfo_exists():
                S["mag"].destroy()
            S["mag"] = None

        def toggle_mag(_=None):
            if current["idx"] != 5:            # 포커싱 탭에서만 — 다른 탭은 Tab 기본동작
                return
            if S["mag"] is not None and S["mag"].winfo_exists():
                close_mag()
                return "break"
            if S["pv"] is None:
                return "break"
            m = tk.Toplevel(root)
            m.title(f"돋보기 ×{MAGX} — Tab으로 닫기")
            m.configure(bg="#0a0b0c")
            m.attributes("-topmost", True)     # 이동·크기조절은 OS 창 그대로
            m.geometry(f"{UIS(320)}x{UIS(220)}+{root.winfo_pointerx() + 24}+{root.winfo_pointery() + 24}")
            m.minsize(UIS(120), UIS(90))
            mcv = tk.Canvas(m, bg="#0a0b0c", highlightthickness=0)
            mcv.pack(fill="both", expand=True)
            S["mag"], S["magcv"] = m, mcv
            m.bind("<Configure>", lambda e: update_mag())
            m.bind("<Tab>", toggle_mag)        # 돋보기 창에 포커스 있어도 Tab으로 닫힘
            for _k in ("1", "2", "3", "4"):    # 돋보기 창 포커스 중에도 기어/재생 단축키
                m.bind(f"<Key-{_k}>", on_gear)
            for _k in ("q", "w", "e", "r"):
                m.bind(f"<Key-{_k}>", on_mark)
                m.bind(f"<Key-{_k.upper()}>", on_mark)
            for _k in ("a", "d", "A", "D"):
                m.bind(f"<Key-{_k}>", on_apkey)
            m.bind("<space>", on_space)
            m.protocol("WM_DELETE_WINDOW", close_mag)
            update_mag()
            return "break"
        root.bind("<Tab>", toggle_mag)

        def on_move(e):
            # 표시 좌표 → 연산 해상도 좌표 (돋보기 샘플링용)
            dw = S.get("dw") or 1; dh = S.get("dh") or 1
            S["mx"] = (e.x - S.get("ox", 0)) * (S["pw"] or 1) / dw
            S["my"] = (e.y - S.get("oy", 0)) * (S["ph"] or 1) / dh
            if S["mag"] is not None:
                update_mag()
        prev_cv.bind("<Motion>", on_move)

        # ---- 로딩 / 뎁스 인식 ----
        def python_exe():
            if getattr(sys, "frozen", False):
                return shutil.which("python") or shutil.which("py") or "python"
            return sys.executable

        def depth_script():
            d = (os.path.dirname(os.path.abspath(sys.argv[0])) if getattr(sys, "frozen", False)
                 else os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(d, "depth_extract.py")

        def _ai_ready():
            """포커싱용 AI 스택(파이썬+torch) 사용 가능 여부 (결과 캐시)"""
            if S.get("ai_ok"):
                return True
            py = python_exe()
            if not py or not shutil.which(py) and not os.path.exists(py):
                return False
            try:
                r = subprocess.run([py, "-c", "import torch"], timeout=45,
                                   capture_output=True,
                                   creationflags=(0x08000000 if os.name == "nt" else 0))
                if r.returncode == 0:
                    S["ai_ok"] = True
                    return True
            except Exception:
                pass
            return False

        def _offer_install():
            """AI 스택이 없으면 동봉된 설치 스크립트를 자동 실행 (원클릭 설치)"""
            appdir = (os.path.dirname(os.path.abspath(sys.argv[0]))
                      if getattr(sys, "frozen", False)
                      else os.path.dirname(os.path.abspath(__file__)))
            bat = os.path.join(appdir, "install_ai.bat")
            if not os.path.exists(bat):
                status.configure(text="AI 구성요소 없음 — install_ai.bat이 폴더에 없습니다", fg=COL["err"])
                return
            if not messagebox.askyesno(
                    "포커싱 탭 첫 설정",
                    "포커싱 탭에 필요한 AI 구성요소(파이썬 + PyTorch, 약 6GB)가 없습니다.\n\n"
                    "지금 자동으로 설치할까요?  (인터넷 필요 · 15~30분 · 최초 1회)\n"
                    "설치 창이 뜨며, 완료되면 영상을 다시 넣으면 됩니다."):
                return
            subprocess.Popen(["cmd", "/c", "start", "", bat], shell=False)
            status.configure(text="AI 구성요소 설치 진행 중 — 설치 창 완료 후 영상을 다시 넣으세요", fg=COL["sub"])

        def load_video(path):
            if S["busy"]:
                return
            S["busy"] = True; S["playing"] = False; S["recording"] = False
            S["cnt"] = None
            rec_btn.set_text("● 녹화"); play_btn.set_text("▶"); set_lamp(COL["card2"])
            status.configure(text=f"불러오는 중: {os.path.basename(path)} …", fg=COL["sub"])

            def work():
                try:
                    if not _ai_ready():                 # AI 스택 없으면 원클릭 설치 제안
                        root.after(0, _offer_install)
                        return
                    cap = cv2.VideoCapture(path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    frames = []
                    while True:
                        ok, f = cap.read()
                        if not ok:
                            break
                        frames.append(f)
                    cap.release()
                    if not frames:
                        raise RuntimeError("프레임 없음")
                    H, W = frames[0].shape[:2]
                    pw = PVW; ph = int(round(pw * H / W))   # 연산은 저해상 고정 — 표시만 창 추종
                    pv = [cv2.resize(f, (pw, ph)).astype(np.float32) / 255.0 for f in frames]
                    root.after(0, lambda: status.configure(text="깊이 인식 중 … (torch/GPU)"))
                    # subprocess 뎁스 추출
                    tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                                       "._focus_depth.npz")
                    cmd = [python_exe(), depth_script(), path, tmp]
                    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, creationflags=(0x08000000 if os.name == "nt" else 0))
                    for line in pr.stdout:
                        line = line.strip()
                        if line.startswith("PROG"):
                            root.after(0, lambda l=line: status.configure(text=f"깊이 인식 {l.split()[1]}"))
                        elif line.startswith("STAB"):
                            root.after(0, lambda: status.configure(text="깊이 시간축 안정화 …"))
                        elif line.startswith("SNAP"):
                            root.after(0, lambda: status.configure(text="깊이 경계 정밀화 …"))
                        elif line.startswith("MATTE"):
                            root.after(0, lambda l=line: status.configure(text=f"털 매팅(ViTMatte) {l.split()[1]}"))
                        elif line.startswith("SAM"):
                            root.after(0, lambda: status.configure(text="피사체 분리(SAM2) …"))
                    pr.wait()
                    if pr.returncode != 0 or not os.path.exists(tmp):
                        raise RuntimeError("뎁스 추출 실패 (시스템 파이썬에 torch/transformers 필요)")
                    z = np.load(tmp)
                    depth16 = z["depth"]; N = int(z["n"])
                    dn16 = z["dnear"] if "dnear" in z.files else None
                    df16 = z["dfar"] if "dfar" in z.files else None
                    mt8 = z["matte"] if "matte" in z.files else None
                    pvd = []
                    for k in range(N):
                        dh = cv2.resize(depth16[k].astype(np.float32) / 65535.0, (pw, ph),
                                        interpolation=cv2.INTER_NEAREST)   # 절벽 보존
                        if mt8 is None:
                            pvd.append(dh)
                        else:                              # refine-edge 팩 (털 매트)
                            pvd.append((dh,
                                        cv2.resize(dn16[k].astype(np.float32) / 65535.0, (pw, ph),
                                                   interpolation=cv2.INTER_NEAREST),
                                        cv2.resize(df16[k].astype(np.float32) / 65535.0, (pw, ph),
                                                   interpolation=cv2.INTER_NEAREST),
                                        cv2.resize(mt8[k].astype(np.float32) / 255.0, (pw, ph),
                                                   interpolation=cv2.INTER_LINEAR)))
                    depth16 = (depth16, dn16, df16, mt8)
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    S.update({"frames": frames, "pv": pv, "depth16": depth16, "pvd": pvd,
                              "N": N, "W": W, "H": H, "fps": fps, "pw": pw, "ph": ph,
                              "cur": 0, "src": path, "fc": np.full(N, 0.5, np.float32),
                              "recmask": np.zeros(N, bool), "stk": None, "stk_key": None,
                              "marks": {}, "keys": [], "akeys": []})

                    def done():
                        tl_scale.configure(to=N - 1); pos_var.set(0)
                        if load_proj():
                            status.configure(text=f"준비완료 · {N}프레임 · 저장된 포커스 프로젝트 복원됨", fg=COL["ok"])
                            draw_wheel(); draw_graph()
                            set_focus(S["ftg"])
                        else:
                            status.configure(text=f"준비완료 · {N}프레임 {W}x{H} · 휠로 포커스 조절", fg=COL["ok"])
                            set_focus(0.5)
                    root.after(0, done)
                except Exception as e:
                    root.after(0, lambda: status.configure(text=f"실패: {e}", fg=COL["err"]))
                finally:
                    S["busy"] = False
            threading.Thread(target=work, daemon=True).start()

        def do_render():
            if S["frames"] is None or S["busy"]:
                return
            S["busy"] = True; S["playing"] = False
            status.configure(text="렌더 중 …", fg=COL["sub"])
            fc = S["fc"].copy()
            if S["recmask"].any():
                idx = np.where(S["recmask"])[0]
                fc = np.interp(np.arange(S["N"]), idx, fc[idx]).astype(np.float32)
                if S["N"] >= 5:
                    # 녹화 커브 정리 — 일시정지 중 휠 탐색 등으로 한 프레임에만 박힌
                    # 초점 스파이크가 렌더에서 단발 깜빡임이 됨: 미디언 디스파이크 +
                    # 가우시안 스무딩(실제 포커스 풀러의 손맛)
                    fc[1:-1] = np.median(np.stack([fc[:-2], fc[1:-1], fc[2:]]), axis=0)
                    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
                    fc = np.convolve(np.pad(fc, 3, mode="edge"), k / k.sum(),
                                     mode="valid").astype(np.float32)
            else:
                fc = np.full(S["N"], S["ftg"], np.float32)
            if S["akeys"]:
                ap_frac = (ap_curve() / 100.0 * 0.04).astype(np.float32)
            else:
                ap_frac = S["ap"] / 100.0 * 0.04
            hl = S["hl"]
            lshape, lca, lbr = S["lshape"], S["lca"], S["lbr"]

            def work():
                try:
                    def prog(v):
                        root.after(0, lambda: status.configure(text=f"렌더 중 … {v}%"))
                    dst = focus_render_video(S["frames"], S["depth16"], fc, ap_frac,
                                             S["fps"], S["src"], progress=prog, hl=hl,
                                             shape=lshape, ca=lca, breath=lbr)
                    root.after(0, save_proj)
                    root.after(0, lambda: status.configure(
                        text=f"완료: {os.path.basename(dst)} (+.mp4)", fg=COL["ok"]))
                except Exception as e:
                    root.after(0, lambda: status.configure(text=f"렌더 실패: {e}", fg=COL["err"]))
                finally:
                    S["busy"] = False
            threading.Thread(target=work, daemon=True).start()
        render_btn.bind("<Button-1>", lambda e: do_render())

        def open_dialog():
            p = filedialog.askopenfilename(filetypes=[("영상", "*.mp4 *.mov *.avi *.mkv *.webm")])
            if p:
                load_video(p)
        open_btn = canvas_button(top, "영상 열기", 90, 30, primary=True)
        open_btn.pack(side="right"); open_btn.bind("<Button-1>", lambda e: open_dialog())

        if dnd_ok:
            def on_drop(ev):
                try:
                    paths = root.tk.splitlist(ev.data)
                except Exception:
                    paths = [ev.data]
                if paths:
                    load_video(paths[0].strip("{}"))
            for w in (page, prev_cv):
                try:
                    w.drop_target_register(DND_FILES); w.dnd_bind("<<Drop>>", on_drop)
                except Exception:
                    pass

        draw_wheel()
        page.add_files = lambda files: (files and load_video(files[0]))
        return page


    def build_bgrm_page():
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)
        B = {"frames": None, "matte": None, "smask": None, "N": 0, "fps": 24.0,
             "src": None, "busy": False, "cur": 0, "photo": None,
             "mode": "alpha", "color": (64, 177, 0), "pts": [], "neg": [],
             "view": "result", "pmask": None, "pmask_i": -1}

        wrap = tk.Frame(page, bg=COL["bg"])
        wrap.pack(fill="both", expand=True, padx=22, pady=14)
        tk.Label(wrap, justify="left", bg=COL["bg"], fg=COL["sub"], font=(FONT, 10),
                 text="피사체(인물)만 남기고 배경을 제거합니다 — SAM2+매팅 기반, 로토 작업 불필요. "
                      "투명(ProRes 4444 알파)으로 뽑아 프리미어에 바로 얹거나, 임의 색 배경으로 합성합니다.\n"
                      "잘못 잡히면(얼굴 잘림 등): Tab(원본 보기)으로 전환 → 마스크 외곽선을 보며 "
                      "살릴 곳 클릭(＋) · 지울 곳 우클릭(×) → [다시 인식]"
                 ).pack(anchor="w", pady=(0, 8))
        top = tk.Frame(wrap, bg=COL["bg"]); top.pack(fill="x")
        status = tk.Label(top, text="영상을 드래그하거나 [영상 열기]", bg=COL["bg"],
                          fg=COL["dim"], font=(MONO, 10)); status.pack(side="left")

        PV2 = UIS(720)
        prev = tk.Canvas(wrap, width=PV2, height=int(PV2 * 9 / 16), bg="#0a0b0c",
                         highlightthickness=1, highlightbackground=COL["border"])
        prev.pack(pady=10, fill="x")            # 창 폭에 맞춰 늘어남

        def _pvw():
            """현재 미리보기 표시 폭 (창 크기 추종)"""
            return max(UIS(300), (prev.winfo_width() or PV2) - 2)

        def _on_prev_cfg(e):                    # 리사이즈 시 재렌더 (8px 이상 변할 때만)
            if B["frames"] is not None and abs(e.width - B.get("_lw", 0)) > 8:
                B["_lw"] = e.width
                render_pv()
        prev.bind("<Configure>", _on_prev_cfg)

        ctl = tk.Frame(wrap, bg=COL["bg"]); ctl.pack(fill="x")
        mode_btn = canvas_button(ctl, "배경: 투명(4444)", 150, 30, primary=False)
        mode_btn.pack(side="left", padx=(0, 6))
        colbox = tk.Canvas(ctl, width=30, height=30, bg="#00b140",
                           highlightthickness=1, highlightbackground=COL["border"])
        colbox.pack(side="left", padx=(0, 14))
        pos2 = tk.IntVar(value=0)

        def render_pv():
            if B["frames"] is None:
                return
            i = B["cur"]
            img = B["frames"][i].astype(np.float32) / 255.0
            h, w = img.shape[:2]
            pw = _pvw(); ph = int(round(pw * h / w))
            has_m = B["matte"] is not None and i < len(B["matte"])
            if has_m and B["view"] == "result":
                a, rgb = _bgrm_alpha(i, img)
                aP = cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)[..., None]
                rP = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
                if B["mode"] == "alpha":           # 투명 미리보기 = 체커보드
                    yy, xx = np.mgrid[0:ph, 0:pw]
                    chk = (((yy // 12) + (xx // 12)) % 2).astype(np.float32)
                    bg = (0.28 + 0.14 * chk)[..., None].repeat(3, 2)
                else:
                    bg = np.empty((ph, pw, 3), np.float32)
                    bg[:] = np.array(B["color"], np.float32)[::-1] / 255.0
                comp = np.clip((rP * aP + bg * (1.0 - aP)) * 255.0, 0, 255).astype(np.uint8)
            else:                                  # 원본 보기(Tab) / 인식 전 — 점 찍기용
                comp = np.clip(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
                               * 255.0, 0, 255).astype(np.uint8)
                pm = (B["pmask"] if B["pmask"] is not None and B["pmask_i"] == i
                      else None)
                mk = None
                if pm is not None:                 # 실시간 점 미리보기 마스크 우선
                    mk = (cv2.resize(pm, (pw, ph),
                                     interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
                elif has_m:
                    mk = (cv2.resize(_bgrm_a(i), (pw, ph),
                                     interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
                if mk is not None:                 # 마스크 외곽선 (AE 마스크처럼)
                    cs = cv2.findContours(mk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.polylines(comp, cs, True, (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.polylines(comp, cs, True, (255, 255, 255), 1, cv2.LINE_AA)
            ok2, buf = cv2.imencode(".ppm", comp)
            if not ok2:
                return
            B["photo"] = tk.PhotoImage(data=buf.tobytes())
            prev.configure(height=ph)
            prev.delete("all")
            prev.create_image(pw / 2, ph / 2, image=B["photo"])
            for x, y in B["pts"]:                  # 수동 포인트 마커
                cx, cy = x * pw / w, y * ph / h
                prev.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                 outline=COL["lime"], width=2)
                prev.create_line(cx - 3, cy, cx + 3, cy, fill=COL["lime"])
                prev.create_line(cx, cy - 3, cx, cy + 3, fill=COL["lime"])
            for x, y in B["neg"]:
                cx, cy = x * pw / w, y * ph / h
                prev.create_line(cx - 5, cy - 5, cx + 5, cy + 5, fill=COL["err"], width=2)
                prev.create_line(cx - 5, cy + 5, cx + 5, cy - 5, fill=COL["err"], width=2)
            frm2.configure(text=f"{i+1}/{B['N']}")

        def _bgrm_a(i):
            """전체 알파(내부=1, 밴드=매트, 밖=0)"""
            a = B["matte"][i].astype(np.float32) / 255.0   # ViTMatte 전체 알파(salpha)
            dil = cv2.dilate(B["smask"][i], np.ones((73, 73), np.uint8))
            return np.clip(np.where(dil > 0, a, 0.0), 0.0, 1.0).astype(np.float32)

        def _bgrm_alpha(i, img01):
            """전체 알파 + 경계 디스필 색(F 추정)"""
            a = _bgrm_a(i)
            # 디스필: 반투명 경계의 색에서 원본 배경 오염 제거 (순수 전경색 외삽)
            wF = np.clip((a - 0.7) / 0.3, 0.0, 1.0)
            k61 = (61, 61)
            Fx = (cv2.boxFilter(img01 * wF[..., None], -1, k61) /
                  np.maximum(cv2.boxFilter(wF, -1, k61), 1e-4)[..., None])
            np.clip(Fx, 0.0, 1.0, out=Fx)
            t = np.clip((a - 0.2) / 0.6, 0.0, 1.0)[..., None]
            rgb = img01 * t + Fx * (1.0 - t)
            return a, rgb

        def toggle_mode():
            B["mode"] = "color" if B["mode"] == "alpha" else "alpha"
            mode_btn.set_text("배경: 투명(4444)" if B["mode"] == "alpha" else "배경: 색상")
            render_pv()
        mode_btn.bind("<Button-1>", lambda e: toggle_mode())

        def pick_color(_=None):
            from tkinter import colorchooser
            c = colorchooser.askcolor(color="#%02x%02x%02x" % B["color"],
                                      title="배경 색 선택")
            if c and c[0]:
                B["color"] = tuple(int(v) for v in c[0])
                colbox.configure(bg="#%02x%02x%02x" % B["color"])
                if B["mode"] == "alpha":
                    toggle_mode()
                else:
                    render_pv()
        colbox.bind("<Button-1>", pick_color)

        redo_btn = canvas_button(ctl, "다시 인식", 100, 30, primary=False)
        redo_btn.pack(side="left", padx=(0, 6))
        redo_btn.bind("<Button-1>", lambda e: rerun())
        clr_btn = canvas_button(ctl, "점 초기화", 90, 30, primary=False)
        clr_btn.pack(side="left")
        view_btn = canvas_button(ctl, "원본 보기 (Tab)", 130, 30, primary=False)
        view_btn.pack(side="left", padx=(6, 0))

        def toggle_view(_=None):
            if B["frames"] is None:
                return
            B["view"] = "orig" if B["view"] == "result" else "result"
            view_btn.set_text("결과 보기 (Tab)" if B["view"] == "orig" else "원본 보기 (Tab)")
            render_pv()
        view_btn.bind("<Button-1>", lambda e: toggle_view())

        def on_tab6(e):
            if current["idx"] != 6:                # 배경 제거 탭에서만
                return
            toggle_view()
            return "break"
        root.bind("<Tab>", on_tab6, add="+")

        def clear_pts(_=None):
            if B["busy"]:
                return
            B["pts"] = []; B["neg"] = []
            B["pmask"] = None; B["pmask_i"] = -1
            status.configure(text="점 초기화 — [다시 인식]을 누르면 자동 인식으로 돌아갑니다",
                             fg=COL["sub"])
            render_pv()
        clr_btn.bind("<Button-1>", clear_pts)

        def _pv_click(ev, keep):
            """미리보기 클릭 → 원본 해상도 좌표의 수동 피사체 포인트 추가"""
            if B["frames"] is None or B["busy"]:
                return
            h, w = B["frames"][0].shape[:2]
            pw = _pvw()
            ph = int(round(pw * h / w))
            x = int(round(ev.x * w / pw))
            y = int(round(ev.y * h / ph))
            if not (0 <= x < w and 0 <= y < h):
                return
            (B["pts"] if keep else B["neg"]).append([x, y])
            if B["view"] == "result":              # 점 찍기 = 원본 보기로 자동 전환
                B["view"] = "orig"
                view_btn.set_text("결과 보기 (Tab)")
            status.configure(text=f"점 {len(B['pts']) + len(B['neg'])}개 — [다시 인식]으로 반영",
                             fg=COL["sub"])
            render_pv()
            _pv_seg_request()                      # 실시간 외곽선 미리보기
        prev.bind("<Button-1>", lambda e: _pv_click(e, True))
        prev.bind("<Button-3>", lambda e: _pv_click(e, False))

        rend_btn = canvas_button(ctl, "렌더", 90, 30, primary=True)
        rend_btn.pack(side="right")
        open2 = canvas_button(top, "영상 열기", 90, 30, primary=True)
        open2.pack(side="right")

        tl2 = tk.Frame(wrap, bg=COL["bg"]); tl2.pack(fill="x", pady=(8, 0))
        sc2 = tk.Scale(tl2, from_=0, to=0, orient="horizontal", variable=pos2,
                       command=lambda v: (B.update(cur=int(float(v))), render_pv()),
                       showvalue=False, length=UIS(560), bg=COL["card"], fg=COL["text"],
                       troughcolor=COL["card2"], highlightthickness=0, bd=0)
        sc2.pack(side="left", fill="x", expand=True, padx=(0, 8))
        frm2 = tk.Label(tl2, text="0/0", bg=COL["bg"], fg=COL["dim"], font=(MONO, 9))
        frm2.pack(side="left")

        # ---- 분석 로딩 오버레이 — 단계·퍼센트 카드 (프리뷰 위에 표시) ----
        ov = tk.Frame(wrap, bg=COL["card"], highlightthickness=1,
                      highlightbackground=COL["border"])
        _ovh = tk.Frame(ov, bg=COL["card"])
        _ovh.pack(fill="x", padx=UIS(18), pady=(UIS(14), UIS(8)))
        tk.Label(_ovh, text="영상 분석 중", bg=COL["card"], fg=COL["text"],
                 font=(FONT, 11, "bold")).pack(side="left")
        ov_pct = tk.Label(_ovh, text="0%", bg=COL["card"], fg=COL["sub"],
                          font=(MONO, 10))
        ov_pct.pack(side="right", padx=(UIS(30), 0))
        ov_bar = progress_bar(ov)
        ov_bar.configure(bg=COL["card"], width=UIS(320))
        ov_bar.pack(fill="x", padx=UIS(18))
        ov_stage = tk.Label(ov, text="", bg=COL["card"], fg=COL["sub"],
                            font=(FONT, 9), anchor="w", width=40)
        ov_stage.pack(fill="x", padx=UIS(18), pady=(UIS(6), UIS(10)))
        ov_cancel = canvas_button(ov, "취소", 90, 28, primary=False)
        ov_cancel.configure(bg=COL["card"])
        ov_cancel.pack(pady=(0, UIS(12)))

        PRG = {"plan": [], "idx": 0, "frac": 0.0, "text": "", "dots": 0,
               "animh": None, "job": 0, "cancel": False, "proc": None}

        def _ov_render():
            plan = PRG["plan"]
            if not plan:
                return
            tw = sum(w for _, w in plan) or 1
            done = (sum(w for _, w in plan[:PRG["idx"]])
                    + plan[PRG["idx"]][1] * PRG["frac"])
            f = min(1.0, done / tw)
            ov_bar.set_frac(f)
            ov_pct.configure(text=f"{int(round(f * 100))}%")
            ov_stage.configure(text=PRG["text"] + " " + "·" * (1 + PRG["dots"]))

        def _ov_set(key, frac, text):
            keys = [k for k, _ in PRG["plan"]]
            if key not in keys:
                return
            i = keys.index(key)
            if i < PRG["idx"]:
                return                          # 지난 단계로 회귀 금지 (지연/오류 라인)
            if i > PRG["idx"]:
                PRG["idx"] = i
                PRG["frac"] = 0.0
            if i == PRG["idx"] and frac is not None:
                PRG["frac"] = max(PRG["frac"], min(1.0, frac))
            PRG["text"] = text
            _ov_render()

        def ov_post(job, key, frac, text):
            """작업 스레드에서 진행상황 반영 (GUI 스레드로 마샬링 · 세대 검사)"""
            def _u():
                if PRG["job"] == job:          # 이전 작업의 지연 갱신 무시
                    _ov_set(key, frac, text)
            root.after(0, _u)

        def _ov_anim():
            PRG["dots"] = (PRG["dots"] + 1) % 3
            _ov_render()
            PRG["animh"] = root.after(500, _ov_anim)

        def _ov_show(plan, text):
            """오버레이 표시 — 작업 세대 번호를 돌려준다 (지연 콜백 무효화용)"""
            PRG["job"] += 1
            PRG.update(plan=plan, idx=0, frac=0.0, text=text,
                       dots=0, cancel=False, proc=None)
            ov_cancel.set_text("취소")
            ov.place(in_=prev, relx=0.5, rely=0.5, anchor="center")
            ov.lift()
            _ov_render()
            if PRG["animh"] is not None:
                try:
                    root.after_cancel(PRG["animh"])
                except Exception:
                    pass
            PRG["animh"] = root.after(500, _ov_anim)
            return PRG["job"]

        def _ov_hide(job=None):
            if job is not None and job != PRG["job"]:
                return                          # 이전 작업의 지연 콜백 — 무시
            if PRG["animh"] is not None:
                try:
                    root.after_cancel(PRG["animh"])
                except Exception:
                    pass
                PRG["animh"] = None
            PRG["plan"] = []
            ov.place_forget()

        def _ov_cancel(_=None):
            if PRG["cancel"] or not PRG["plan"]:
                return
            PRG["cancel"] = True
            ov_cancel.set_text("취소 중 …")
            p = PRG.get("proc")
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        ov_cancel.bind("<Button-1>", _ov_cancel)

        def py2():
            if getattr(sys, "frozen", False):
                return shutil.which("python") or shutil.which("py") or "python"
            return sys.executable

        def script2():
            d = (os.path.dirname(os.path.abspath(sys.argv[0])) if getattr(sys, "frozen", False)
                 else os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(d, "depth_extract.py")

        # ---- 실시간 점 미리보기 (SAM2 이미지 서버 상주) ----
        SRV = {"p": None, "ready": False, "pending": None, "thread": None, "fsent": None}
        pv_img = os.path.join(tempfile.gettempdir(), "nps_bgrm_frame.npy")
        pv_msk = os.path.join(tempfile.gettempdir(), "nps_bgrm_mask.npy")

        def _srv_ensure():
            if SRV["p"] is not None and SRV["p"].poll() is None:
                return True
            try:
                SRV["p"] = subprocess.Popen(
                    [py2(), script2(), "--serve"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True,
                    creationflags=(0x08000000 if os.name == "nt" else 0))
                SRV["ready"] = False
                SRV["fsent"] = None
                return True
            except Exception:
                SRV["p"] = None
                return False

        def _pv_seg_request():
            """현재 점으로 현재 프레임 마스크를 백그라운드 계산 (최신 요청만 유지)."""
            if B["frames"] is None:
                return
            SRV["pending"] = (B["cur"], list(B["pts"]), list(B["neg"]))
            if SRV["thread"] is not None and SRV["thread"].is_alive():
                return

            def work():
                if not _srv_ensure():
                    return
                p = SRV["p"]
                try:
                    if not SRV["ready"]:
                        root.after(0, lambda: status.configure(
                            text="실시간 미리보기 준비 중 (최초 1회) …", fg=COL["sub"]))
                        if not p.stdout.readline():        # READY 대기 (모델 로드)
                            return
                        SRV["ready"] = True
                    while SRV["pending"] is not None:
                        i2, pp, nn = SRV["pending"]
                        SRV["pending"] = None
                        if SRV["fsent"] != i2:
                            np.save(pv_img, B["frames"][i2])
                            SRV["fsent"] = i2
                        ps = ";".join(f"{x},{y}" for x, y in pp) or "-"
                        ns = ";".join(f"{x},{y}" for x, y in nn) or "-"
                        p.stdin.write(f"SEG\t{pv_img}\t{pv_msk}\t{ps}\t{ns}\n")
                        p.stdin.flush()
                        resp = (p.stdout.readline() or "").strip()
                        if not resp.startswith("OK"):
                            continue
                        try:
                            mk = np.load(pv_msk)
                        except Exception:
                            continue
                        B["pmask"] = mk
                        B["pmask_i"] = i2
                        root.after(0, lambda: (status.configure(
                            text="미리보기 갱신 — 전체 프레임 적용은 [다시 인식]",
                            fg=COL["sub"]), render_pv()))
                except Exception:
                    pass
            SRV["thread"] = threading.Thread(target=work, daemon=True)
            SRV["thread"].start()

        def _run_recog(job):
            """SAM2+매팅 인식 서브프로세스 실행 + 결과 로드 (작업 스레드에서 호출).
            B['pts']/B['neg']가 있으면 수동 포인트로, 없으면 자동 인식."""
            path = B["src"]
            tmp = os.path.join(os.path.dirname(os.path.abspath(path)), "._bgrm_depth.npz")
            cmd = [py2(), script2(), path, tmp, "--matteonly"]
            if B["pts"]:
                cmd += ["--pts", ";".join(f"{x},{y}" for x, y in B["pts"])]
            if B["neg"]:
                cmd += ["--neg", ";".join(f"{x},{y}" for x, y in B["neg"])]
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, creationflags=(0x08000000 if os.name == "nt" else 0))
            PRG["proc"] = pr
            if PRG["cancel"]:                  # 시작 직전에 취소 눌림 → 즉시 중단
                try:
                    pr.kill()
                except Exception:
                    pass
            ov_post(job, "eread", None, "엔진 시작 (torch 로드)")

            def _ij(l):
                """'TAG i/n' → (i, n) — 총계 미상이면 n=0"""
                try:
                    a, b = l.split()[1].split("/")
                    return int(a), int(b)
                except Exception:
                    return 0, 0
            skipmsg = None
            for line in pr.stdout:
                line = line.strip()
                if line.startswith("DEC "):
                    i, n = _ij(line)
                    ov_post(job, "eread", (i / n if n > 0 else None),
                            f"엔진 · 소스 읽기 {i}" + (f"/{n}" if n > 0 else ""))
                elif line.startswith("FRAMES"):
                    ov_post(job, "eread", 1.0, "엔진 · 소스 읽기 완료")
                elif line.startswith("LOAD"):
                    ov_post(job, "eread", None, "피사체 인식 모델 로드")
                elif line.startswith("PREP "):
                    i, n = _ij(line)
                    ov_post(job, "prep", (i / n * 0.7 if n else None),
                            f"SAM2 준비 · 프레임 변환 {i}/{n}")
                elif line.startswith("SAMLOAD"):
                    ov_post(job, "prep", 0.75, "SAM2 모델 로드")
                elif line.startswith("SAMINIT"):
                    ov_post(job, "prep", 0.85, "SAM2 세션 초기화")
                elif line.startswith("SAMP "):
                    i, n = _ij(line)
                    ov_post(job, "samp", (i / n if n else None), f"피사체 분리(SAM2) {i}/{n}")
                elif line.startswith("MATTE "):
                    i, n = _ij(line)
                    ov_post(job, "matte", (i / n if n else None), f"경계 매팅(ViTMatte) {i}/{n}")
                elif line.startswith("MLOAD"):
                    ov_post(job, "matte", 0.0, "매팅 모델 로드")
                elif line.startswith("SAVE"):
                    ov_post(job, "save", None, "결과 저장 중")
                elif line.startswith("SAMSKIP"):       # SAM2 실패 — 원인 보존
                    skipmsg = (line.split(None, 1)[1] if " " in line else "SAM2 실패")[:120]
                elif line.startswith("SAM"):           # SAM/SAMPT — 전파 준비 단계
                    ov_post(job, "prep", None, "피사체 분리 준비")
            pr.wait()
            if PRG["cancel"]:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                raise RuntimeError("취소됨")
            if pr.returncode != 0 or not os.path.exists(tmp):
                raise RuntimeError("인식 실패 (포커싱 탭에서 AI 구성요소를 먼저 설치하세요)")
            ov_post(job, "save", 0.5, "결과 불러오는 중")
            z = np.load(tmp)
            if "salpha" not in z.files:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                raise RuntimeError("피사체를 못 찾음 — 미리보기에서 피사체를 클릭 후 [다시 인식]"
                                   + (f" · 원인: {skipmsg}" if skipmsg else ""))
            matte = z["salpha"]; smask = z["smask"]; N = int(z["n"])
            try:
                os.remove(tmp)
            except Exception:
                pass
            if PRG["cancel"]:                  # 결과 로드 중 취소 눌림
                raise RuntimeError("취소됨")
            B.update({"matte": matte, "smask": smask, "N": N,
                      "cur": min(B["cur"], N - 1)})

            def done():
                sc2.configure(to=N - 1); pos2.set(B["cur"])
                B["view"] = "result"
                B["pmask"] = None; B["pmask_i"] = -1
                view_btn.set_text("원본 보기 (Tab)")
                status.configure(text=f"준비완료 · {N}프레임 · 확인 후 [렌더] — 잘렸으면 "
                                      f"Tab(원본 보기)에서 클릭 보정", fg=COL["ok"])
                render_pv()
            root.after(0, done)

        def _fail_status(e):
            """작업 실패/취소 → 상태 표시 (예외 메시지를 값으로 캡처)"""
            msg = str(e)
            root.after(0, lambda m=msg: status.configure(
                text=("취소됨" if m == "취소됨" else f"실패: {m}"),
                fg=(COL["sub"] if m == "취소됨" else COL["err"])))

        def load2(path):
            if B["busy"]:
                return
            B["busy"] = True
            status.configure(text=f"불러오는 중: {os.path.basename(path)} …", fg=COL["sub"])
            job = _ov_show([("read", 14), ("eread", 14), ("prep", 12),
                            ("samp", 22), ("matte", 34), ("save", 4)], "영상 읽기")

            def work():
                try:
                    cap = cv2.VideoCapture(path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    frames = []
                    while True:
                        ok2, f = cap.read()
                        if not ok2:
                            break
                        frames.append(f)
                        if PRG["cancel"]:
                            raise RuntimeError("취소됨")
                        if len(frames) % 10 == 0:
                            ov_post(job, "read", (len(frames) / total if total > 0 else None),
                                    f"영상 읽기 {len(frames)}"
                                    + (f"/{total}" if total > 0 else ""))
                    cap.release()
                    if not frames:
                        raise RuntimeError("프레임 없음")
                    B.update({"frames": frames, "matte": None, "smask": None,
                              "N": len(frames), "fps": fps, "src": path,
                              "cur": 0, "pts": [], "neg": [], "view": "result",
                              "pmask": None, "pmask_i": -1})
                    SRV["fsent"] = None
                    root.after(0, lambda: (sc2.configure(to=len(frames) - 1),
                                           pos2.set(0),
                                           view_btn.set_text("원본 보기 (Tab)"),
                                           render_pv()))
                    _run_recog(job)
                except Exception as e:
                    _fail_status(e)
                finally:
                    B["busy"] = False
                    root.after(0, lambda j=job: _ov_hide(j))
            threading.Thread(target=work, daemon=True).start()

        def rerun():
            if B["frames"] is None or B["busy"]:
                return
            B["busy"] = True
            status.configure(text="다시 인식 중 …", fg=COL["sub"])
            job = _ov_show([("eread", 14), ("prep", 12), ("samp", 22),
                            ("matte", 34), ("save", 4)], "엔진 시작")

            def work():
                try:
                    _run_recog(job)
                except Exception as e:
                    _fail_status(e)
                finally:
                    B["busy"] = False
                    root.after(0, lambda j=job: _ov_hide(j))
            threading.Thread(target=work, daemon=True).start()

        def do_render2():
            if B["frames"] is None or B["matte"] is None or B["busy"]:
                return
            B["busy"] = True
            status.configure(text="렌더 중 …", fg=COL["sub"])
            mode, color = B["mode"], B["color"]

            def work():
                try:
                    ff = get_ffmpeg()
                    src2 = B["src"]
                    h, w = B["frames"][0].shape[:2]
                    base2, _ = os.path.splitext(src2)
                    flags = 0x08000000 if os.name == "nt" else 0
                    if mode == "alpha":
                        dst = base2 + "_bgrm.mov"
                        cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
                               "-s", f"{w}x{h}", "-r", f"{B['fps']}", "-i", "-",
                               "-i", src2, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
                               "-c:v", "prores_ks", "-profile:v", "4444",
                               "-pix_fmt", "yuva444p10le", dst]
                    else:
                        dst = base2 + "_bgrm.mov"
                        cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{w}x{h}", "-r", f"{B['fps']}", "-i", "-",
                               "-i", src2, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
                               "-c:v", "prores_ks", "-profile:v", "3",
                               "-pix_fmt", "yuv422p10le", dst]
                    p2 = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=flags)
                    bgc = np.array(color, np.float32)[::-1] / 255.0
                    for i in range(B["N"]):
                        img = B["frames"][i].astype(np.float32) / 255.0
                        a, rgb = _bgrm_alpha(i, img)
                        if mode == "alpha":
                            out = np.dstack([np.clip(rgb * 255, 0, 255).astype(np.uint8),
                                             np.clip(a * 255, 0, 255).astype(np.uint8)])
                        else:
                            comp = rgb * a[..., None] + bgc * (1.0 - a[..., None])
                            out = np.clip(comp * 255, 0, 255).astype(np.uint8)
                        p2.stdin.write(out.tobytes())
                        if i % 5 == 0:
                            root.after(0, lambda v=int((i + 1) / B["N"] * 100):
                                       status.configure(text=f"렌더 중 … {v}%"))
                    p2.stdin.close(); p2.wait()
                    if p2.returncode != 0:
                        raise RuntimeError("FFmpeg 인코딩 실패")
                    root.after(0, lambda: status.configure(
                        text=f"완료: {os.path.basename(dst)}", fg=COL["ok"]))
                except Exception as e:
                    msg = str(e)
                    root.after(0, lambda m=msg: status.configure(
                        text=f"렌더 실패: {m}", fg=COL["err"]))
                finally:
                    B["busy"] = False
            threading.Thread(target=work, daemon=True).start()
        rend_btn.bind("<Button-1>", lambda e: do_render2())

        def open_dlg2():
            p = filedialog.askopenfilename(filetypes=[("영상", "*.mp4 *.mov *.avi *.mkv *.webm")])
            if p:
                load2(p)
        open2.bind("<Button-1>", lambda e: open_dlg2())
        if dnd_ok:
            def on_drop2(ev):
                try:
                    paths = root.tk.splitlist(ev.data)
                except Exception:
                    paths = [ev.data]
                if paths:
                    load2(paths[0].strip("{}"))
            for wdg in (page, prev):
                try:
                    wdg.drop_target_register(DND_FILES); wdg.dnd_bind("<<Drop>>", on_drop2)
                except Exception:
                    pass
        page.add_files = lambda files: (files and load2(files[0]))
        return page

    def build_vcam_page():
        """TEST · 가상 카메라 — 완성 클립에 뎁스 기반 카메라 무빙(펀치인/리프레임) 추가."""
        page = tk.Frame(body, bg=COL["bg"])
        page.place(x=0, y=0, relwidth=1, relheight=1)
        V = {"pv": None, "pvd": None, "frames": None, "depth16": None,
             "pva": None, "bgp": None, "salpha": None, "smask": None,
             "N": 0, "fps": 24.0, "W": 0, "H": 0, "cur": 0, "busy": False,
             "src": None, "playing": False, "view": None, "photo": None,
             "edit": "A", "par": 0.7, "grid": None,
             "A": {"z": 1.0, "cx": 0.5, "cy": 0.5},
             "B": {"z": 1.35, "cx": 0.5, "cy": 0.5},
             "dw": 1, "dh": 1, "ox": 0, "oy": 0}

        wrap = tk.Frame(page, bg=COL["bg"])
        wrap.pack(fill="both", expand=True, padx=22, pady=14)
        tk.Label(wrap, justify="left", bg=COL["bg"], fg=COL["sub"], font=(FONT, 10),
                 text="TEST · 완성된 클립에 뎁스 기반 카메라 무빙을 더합니다 — 와이드 하나로 푸시인·클로즈업 커버리지 파생.\n"
                      "미리보기 드래그=프레이밍 · 휠=줌 · [A 시작]/[B 끝] 카메라를 잡으면 클립 전체에 A→B 무빙. "
                      "이동시차 0%는 단순 펀치인, 올릴수록 돌리(달리)처럼 가까운 것이 더 크게 움직입니다."
                 ).pack(anchor="w", pady=(0, 8))
        top = tk.Frame(wrap, bg=COL["bg"]); top.pack(fill="x")
        status = tk.Label(top, text="영상을 드래그하거나 [영상 열기]", bg=COL["bg"],
                          fg=COL["dim"], font=(MONO, 10)); status.pack(side="left")
        open_b = canvas_button(top, "영상 열기", 90, 30, primary=True)
        open_b.pack(side="right")
        pbar3 = progress_bar(wrap)
        pbar3.pack(fill="x", pady=(6, 0))

        def _setp(f):
            root.after(0, lambda: pbar3.set_frac(f))

        prev = tk.Canvas(wrap, height=UIS(420), bg="#0a0b0c", cursor="fleur",
                         highlightthickness=1, highlightbackground=COL["border"])
        prev.pack(fill="both", expand=True, pady=10)

        # ---- GDI 고속 표시 (포커싱 탭과 동일 기법, 페이지 로컬) ----
        import ctypes as _c2

        class _BIH2(_c2.Structure):
            _fields_ = [("biSize", _c2.c_uint32), ("biWidth", _c2.c_int32),
                        ("biHeight", _c2.c_int32), ("biPlanes", _c2.c_uint16),
                        ("biBitCount", _c2.c_uint16), ("biCompression", _c2.c_uint32),
                        ("biSizeImage", _c2.c_uint32), ("biXPelsPerMeter", _c2.c_int32),
                        ("biYPelsPerMeter", _c2.c_int32), ("biClrUsed", _c2.c_uint32),
                        ("biClrImportant", _c2.c_uint32)]

        _g2 = {"ok": os.name == "nt"}
        if _g2["ok"]:
            try:
                _u2 = _c2.windll.user32; _gd2 = _c2.windll.gdi32
                _u2.GetDC.restype = _c2.c_void_p
                _u2.GetDC.argtypes = [_c2.c_void_p]
                _u2.ReleaseDC.argtypes = [_c2.c_void_p, _c2.c_void_p]
                _gd2.SetStretchBltMode.argtypes = [_c2.c_void_p, _c2.c_int]
                _gd2.SetBrushOrgEx.argtypes = [_c2.c_void_p, _c2.c_int, _c2.c_int,
                                               _c2.c_void_p]
                _gd2.StretchDIBits.argtypes = ([_c2.c_void_p] + [_c2.c_int] * 8 +
                                               [_c2.c_void_p, _c2.c_void_p,
                                                _c2.c_uint, _c2.c_uint32])
            except Exception:
                _g2["ok"] = False

        def _present2(bgr):
            cw = max(prev.winfo_width(), 8); chh = max(prev.winfo_height(), 8)
            h, w = bgr.shape[:2]
            sc = min(cw / w, chh / h)
            dw = max(2, int(w * sc)); dh = max(2, int(h * sc))
            V["dw"], V["dh"] = dw, dh
            V["ox"], V["oy"] = (cw - dw) // 2, (chh - dh) // 2
            if _g2["ok"]:
                try:
                    data = np.ascontiguousarray(bgr)
                    hdc = _u2.GetDC(prev.winfo_id())
                    if hdc:
                        try:
                            _gd2.SetStretchBltMode(hdc, 4)
                            _gd2.SetBrushOrgEx(hdc, 0, 0, None)
                            bi = _BIH2(biSize=40, biWidth=w, biHeight=-h, biPlanes=1,
                                       biBitCount=24, biCompression=0)
                            _gd2.StretchDIBits(hdc, V["ox"], V["oy"], dw, dh, 0, 0, w, h,
                                               data.ctypes.data_as(_c2.c_void_p),
                                               _c2.byref(bi), 0, 0x00CC0020)
                            return
                        finally:
                            _u2.ReleaseDC(prev.winfo_id(), hdc)
                except Exception:
                    _g2["ok"] = False
            disp = cv2.resize(bgr, (dw, dh))
            ok2, buf = cv2.imencode(".ppm", disp)
            if ok2:
                V["photo"] = tk.PhotoImage(data=buf.tobytes())
                prev.delete("all")
                prev.create_image(cw / 2, chh / 2, image=V["photo"])

        prev.bind("<Expose>", lambda e: V["view"] is not None and _present2(V["view"]))
        prev.bind("<Configure>", lambda e: V["view"] is not None and _present2(V["view"]))

        # ---- 카메라/워프 ----
        def _grid(w, h):
            if V["grid"] is None or V["grid"][0].shape != (h, w):
                V["grid"] = np.meshgrid(np.arange(w, dtype=np.float32),
                                        np.arange(h, dtype=np.float32))
            return V["grid"]

        def cam_at(t):
            e = t * t * (3.0 - 2.0 * t)            # smoothstep 이징
            A, B = V["A"], V["B"]
            return {k: A[k] + (B[k] - A[k]) * e for k in ("z", "cx", "cy")}

        def smooth_dn(dn):
            """워프용 뎁스 정리 — 근경 팽창(max) + 블러.
            깊이 절벽이 있으면 backward warp에서 실루엣 단층(찢어짐)이 생기므로,
            피사체를 최근경으로 한 덩어리로 굳히고 경사를 완만하게 만든다."""
            w2 = dn.shape[1]
            k = max(3, int(w2 * 0.030) * 2 + 1)
            d = cv2.dilate(dn, np.ones((k, k), np.uint8))
            d = cv2.GaussianBlur(d, (0, 0), w2 * 0.035)
            # 대비 압축(뎁스 요철 물결 억제) — 기준점은 고정 0.5.
            # 프레임별 평균을 쓰면 평균 변동이 확대율을 흔들어 깜빡임이 생긴다.
            return 0.5 + (d - 0.5) * 0.7

        def warp(img, dn, cam, par):
            """뎁스 가중 줌 리프레임 — (cx,cy)의 소스 지점이 화면 중앙에 오도록.
            par>0이면 가까운 픽셀이 더 크게 확대되어 돌리 이동시차가 생긴다."""
            h, w = img.shape[:2]
            gx, gy = _grid(w, h)
            z = max(1.0, float(cam["z"]))
            zc = _zcov(cam, w, h)
            zeff = np.maximum(1.0 + (z - 1.0) * (1.0 - par + par * dn), zc)
            mapx = cam["cx"] * w + (gx - w * 0.5) / zeff
            mapy = cam["cy"] * h + (gy - h * 0.5) / zeff
            return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

        def _zcov(cam, w, h):
            """패닝 시 샘플 창이 원본 밖으로 나가지 않는 최소 줌(커버리지 하한).
            밖을 보면 가장자리 픽셀이 늘어나 물결 줄무늬가 생긴다."""
            cxp, cyp = cam["cx"] * w, cam["cy"] * h
            mx = max(min(cxp, w - cxp), 1e-3)
            my = max(min(cyp, h - cyp), 1e-3)
            return max(1.0, w / (2.0 * mx), h / (2.0 * my))

        def render_pv():
            if V["pv"] is None:
                return
            i = V["cur"]
            t = i / max(1, V["N"] - 1)
            out = warp(V["pv"][i], V["pvd"][i], cam_at(t), V["par"])
            V["view"] = out
            _present2(out)
            st = V[V["edit"]]
            frm_l.configure(text=f"{i+1}/{V['N']} · 편집 {V['edit']} · z {st['z']:.2f}")

        # ---- 컨트롤 ----
        ctl = tk.Frame(wrap, bg=COL["bg"]); ctl.pack(fill="x")
        ab_b = canvas_button(ctl, "편집: A (시작)", 120, 30, primary=False)
        ab_b.pack(side="left", padx=(0, 6))
        cpy_b = canvas_button(ctl, "B ← A 복사", 100, 30, primary=False)
        cpy_b.pack(side="left", padx=(0, 6))
        rst_b = canvas_button(ctl, "초기화", 70, 30, primary=False)
        rst_b.pack(side="left", padx=(0, 14))
        tk.Label(ctl, text="이동시차", bg=COL["bg"], fg=COL["sub"],
                 font=(FONT, 9)).pack(side="left")
        par_v = tk.IntVar(value=70)
        tk.Scale(ctl, from_=0, to=100, orient="horizontal", variable=par_v,
                 command=lambda v: (V.update(par=int(v) / 100.0), render_pv()),
                 showvalue=True, length=UIS(150), bg=COL["card"], fg=COL["text"],
                 troughcolor=COL["card2"], highlightthickness=0, bd=0,
                 font=(MONO, 8)).pack(side="left", padx=(6, 0))
        rend_b = canvas_button(ctl, "렌더", 90, 30, primary=True)
        rend_b.pack(side="right")

        tl = tk.Frame(wrap, bg=COL["bg"]); tl.pack(fill="x", pady=(8, 0))
        play_b = canvas_button(tl, "▶", 40, 30, primary=False); play_b.pack(side="left")
        pos_v = tk.IntVar(value=0)
        sc3 = tk.Scale(tl, from_=0, to=0, orient="horizontal", variable=pos_v,
                       command=lambda v: (V.update(cur=int(float(v)), playing=False),
                                          render_pv()),
                       showvalue=False, bg=COL["card"], fg=COL["text"],
                       troughcolor=COL["card2"], highlightthickness=0, bd=0)
        sc3.pack(side="left", fill="x", expand=True, padx=8)
        frm_l = tk.Label(tl, text="0/0", bg=COL["bg"], fg=COL["dim"], font=(MONO, 9))
        frm_l.pack(side="left")

        def set_edit(which):
            V["edit"] = which
            ab_b.set_text("편집: A (시작)" if which == "A" else "편집: B (끝)")
            V["cur"] = 0 if which == "A" else max(0, V["N"] - 1)
            pos_v.set(V["cur"])
            render_pv()
        ab_b.bind("<Button-1>", lambda e: set_edit("B" if V["edit"] == "A" else "A"))

        def copy_ab(_=None):
            V["B"] = dict(V["A"])
            set_edit("B")
        cpy_b.bind("<Button-1>", lambda e: copy_ab())

        def reset_cam(_=None):
            V["A"] = {"z": 1.0, "cx": 0.5, "cy": 0.5}
            V["B"] = {"z": 1.35, "cx": 0.5, "cy": 0.5}
            set_edit("A")
        rst_b.bind("<Button-1>", lambda e: reset_cam())

        drag = {"x": 0, "y": 0}

        def on_press(e):
            drag["x"], drag["y"] = e.x, e.y

        def on_drag(e):
            if V["pv"] is None:
                return
            st = V[V["edit"]]
            z = max(st["z"], 1.0)
            st["cx"] = min(0.83, max(0.17, st["cx"] - (e.x - drag["x"]) / (V["dw"] * z)))
            st["cy"] = min(0.83, max(0.17, st["cy"] - (e.y - drag["y"]) / (V["dh"] * z)))
            drag["x"], drag["y"] = e.x, e.y
            render_pv()

        def on_wheel(e):
            if V["pv"] is None:
                return
            st = V[V["edit"]]
            st["z"] = min(3.0, max(1.0, st["z"] * (1.08 if e.delta > 0 else 1 / 1.08)))
            render_pv()
        prev.bind("<Button-1>", on_press)
        prev.bind("<B1-Motion>", on_drag)
        prev.bind("<MouseWheel>", on_wheel)

        def toggle_play(_=None):
            if V["pv"] is None:
                return
            V["playing"] = not V["playing"]
            play_b.set_text("❚❚" if V["playing"] else "▶")
            if V["playing"]:
                _tick()
        play_b.bind("<Button-1>", toggle_play)

        def _tick():
            if not V["playing"] or V["pv"] is None:
                return
            V["cur"] = (V["cur"] + 1) % V["N"]
            pos_v.set(V["cur"])
            render_pv()
            root.after(max(15, int(1000 / max(V["fps"], 1.0))), _tick)

        # ---- 로드 (뎁스 파이프라인 재사용) ----
        def py3():
            if getattr(sys, "frozen", False):
                return shutil.which("python") or shutil.which("py") or "python"
            return sys.executable

        def scr3():
            d = (os.path.dirname(os.path.abspath(sys.argv[0])) if getattr(sys, "frozen", False)
                 else os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(d, "depth_extract.py")

        def load3(path):
            if V["busy"]:
                return
            V["busy"] = True
            V["playing"] = False
            pbar3.set_frac(0.0)
            status.configure(text=f"불러오는 중: {os.path.basename(path)} …", fg=COL["sub"])

            def work():
                try:
                    cap = cv2.VideoCapture(path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    frames = []
                    while True:
                        ok3, f = cap.read()
                        if not ok3:
                            break
                        frames.append(f)
                    cap.release()
                    if not frames:
                        raise RuntimeError("프레임 없음")
                    H, W = frames[0].shape[:2]
                    tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                                       "._vcam_depth.npz")
                    pr = subprocess.Popen([py3(), scr3(), path, tmp],
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True,
                                          creationflags=(0x08000000 if os.name == "nt" else 0))
                    for line in pr.stdout:
                        line = line.strip()
                        if line.startswith("PROG"):
                            root.after(0, lambda l=line: status.configure(
                                text=f"깊이 인식 {l.split()[1]}"))
                            try:
                                a3, b3 = line.split()[1].split("/")
                                _setp(0.55 * int(a3) / max(1, int(b3)))
                            except Exception:
                                pass
                        elif line.startswith(("STAB", "SNAP")):
                            root.after(0, lambda: status.configure(text="깊이 안정화 …"))
                            _setp(0.58)
                        elif line.startswith("MATTE"):
                            root.after(0, lambda l=line: status.configure(
                                text=f"피사체 매팅 {l.split()[1]}"))
                            try:
                                a3, b3 = line.split()[1].split("/")
                                _setp(0.62 + 0.28 * int(a3) / max(1, int(b3)))
                            except Exception:
                                pass
                        elif line.startswith("SAM"):
                            root.after(0, lambda: status.configure(text="피사체 분리(SAM2) …"))
                            _setp(0.60)
                    pr.wait()
                    if pr.returncode != 0 or not os.path.exists(tmp):
                        raise RuntimeError("뎁스 추출 실패 (포커싱 탭에서 AI 구성요소 먼저 설치)")
                    z = np.load(tmp)
                    depth16 = z["depth"]; N = int(z["n"])
                    salpha = z["salpha"] if "salpha" in z.files else None
                    smask = z["smask"] if "smask" in z.files else None
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    pw = 640; ph = int(round(pw * H / W))
                    pv = [cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
                          for f in frames]
                    rd = [cv2.resize(depth16[k].astype(np.float32) / 65535.0, (pw, ph),
                                     interpolation=cv2.INTER_LINEAR) for k in range(N)]
                    if smask is not None:
                        # 피사체 뎁스 평탄화 — 클립 전체에서 하나의 상수 깊이로 고정.
                        # 피사체는 강체로 확대(내부 늘어남 없음)되고, 합성·인페인트가
                        # 없으므로 깜빡임 원인 자체가 없다.
                        root.after(0, lambda: status.configure(text="뎁스 정리(피사체 평탄화) …"))
                        rm = [cv2.resize(smask[k], (pw, ph),
                                         interpolation=cv2.INTER_NEAREST) for k in range(N)]
                        cs = []
                        for k in range(0, N, max(1, N // 12)):
                            m = rm[k] > 0
                            if m.any():
                                cs.append(float(np.percentile(rd[k][m], 85)))
                        c_subj = float(np.median(cs)) if cs else None
                        ks = max(3, int(9 * pw / W) | 1)
                        k9s = np.ones((ks, ks), np.uint8)
                        pvd = []
                        for k in range(N):
                            if k % 8 == 0:
                                _setp(0.90 + 0.08 * k / max(1, N))
                            d2 = rd[k]
                            if c_subj is not None:
                                mm = cv2.dilate(rm[k], k9s)
                                d2 = np.where(mm > 0, c_subj, d2).astype(np.float32)
                            pvd.append(smooth_dn(d2))
                    else:
                        pvd = [smooth_dn(rd[k]) for k in range(N)]
                    if N >= 3:                     # 시간축 3탭 블렌딩 (미세 요동 제거)
                        pvd = ([pvd[0]] +
                               [0.25 * pvd[k - 1] + 0.5 * pvd[k] + 0.25 * pvd[k + 1]
                                for k in range(1, N - 1)] + [pvd[-1]])
                    V.update({"frames": frames, "pv": pv, "pvd": pvd, "depth16": depth16,
                              "pva": None, "bgp": None, "salpha": None, "smask": None,
                              "N": N, "fps": fps, "W": W, "H": H, "src": path, "cur": 0})

                    def done():
                        sc3.configure(to=N - 1)
                        pbar3.set_frac(1.0)
                        status.configure(text=f"준비완료 · {N}프레임 · A/B 카메라 잡고 재생으로 확인",
                                         fg=COL["ok"])
                        reset_cam()
                    root.after(0, done)
                except Exception as e:
                    root.after(0, lambda: status.configure(text=f"실패: {e}", fg=COL["err"]))
                finally:
                    V["busy"] = False
            threading.Thread(target=work, daemon=True).start()

        # ---- 렌더 (원본 해상도) ----
        def do_render3():
            if V["frames"] is None or V["busy"]:
                return
            V["busy"] = True
            V["playing"] = False
            status.configure(text="렌더 중 …", fg=COL["sub"])
            par = V["par"]

            def work():
                try:
                    ff = get_ffmpeg()
                    W, H, N = V["W"], V["H"], V["N"]
                    base3, _ = os.path.splitext(V["src"])
                    dst = base3 + "_vcam.mov"
                    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", f"{V['fps']}", "-i", "-",
                           "-i", V["src"], "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
                           "-c:v", "prores_ks", "-profile:v", "3",
                           "-pix_fmt", "yuv422p10le", dst]
                    p3 = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=(0x08000000 if os.name == "nt" else 0))
                    for i in range(N):
                        t = i / max(1, N - 1)
                        db = cv2.resize(V["pvd"][i], (W, H),
                                        interpolation=cv2.INTER_LINEAR)
                        out = warp(V["frames"][i], db, cam_at(t), par)
                        p3.stdin.write(out.tobytes())
                        if i % 5 == 0:
                            _setp((i + 1) / N)
                            root.after(0, lambda v=int((i + 1) / N * 100):
                                       status.configure(text=f"렌더 중 … {v}%"))
                    p3.stdin.close(); p3.wait()
                    if p3.returncode != 0:
                        raise RuntimeError("FFmpeg 인코딩 실패")
                    root.after(0, lambda: status.configure(
                        text=f"완료: {os.path.basename(dst)}", fg=COL["ok"]))
                except Exception as e:
                    root.after(0, lambda: status.configure(text=f"렌더 실패: {e}", fg=COL["err"]))
                finally:
                    V["busy"] = False
            threading.Thread(target=work, daemon=True).start()
        rend_b.bind("<Button-1>", lambda e: do_render3())

        def open_dlg3(_=None):
            p = filedialog.askopenfilename(filetypes=[("영상", "*.mp4 *.mov *.avi *.mkv *.webm")])
            if p:
                load3(p)
        open_b.bind("<Button-1>", open_dlg3)
        if dnd_ok:
            def on_drop3(ev):
                try:
                    paths = root.tk.splitlist(ev.data)
                except Exception:
                    paths = [ev.data]
                if paths:
                    load3(paths[0].strip("{}"))
            for wdg in (page, prev):
                try:
                    wdg.drop_target_register(DND_FILES)
                    wdg.dnd_bind("<<Drop>>", on_drop3)
                except Exception:
                    pass
        page.add_files = lambda files: (files and load3(files[0]))
        return page

    pages[0] = build_page(
        "밴딩(계단현상) 제거 — 이미지는 디테일 보존 스무딩, 영상은 여러 프레임을 "
        "스태킹해 진짜 10비트 계조를 복원(색보정 버팀). 영상은 ProRes .mov(그레이딩용)+"
        ".mp4(미리보기), 이미지는 .png 로 원본 폴더에 저장 ",
        "_debanded", deband, "_debanded", media=True)
    pages[1] = build_photoreal_page()
    pages[2] = build_removegpt_page()
    pages[3] = build_sizzle_page()
    pages[4] = build_detailboost_page()
    pages[5] = build_focus_page()
    pages[6] = build_bgrm_page()
    pages[7] = build_vcam_page()

    pages[0].tkraise()

    if initial_files:
        if "--tab5" in sys.argv:   # UI 테스트용: 디테일 보강 탭으로 바로 로드
            select_tab(4)
            root.attributes("-topmost", True)
            root.after(200, pages[4].add_files, initial_files)
        elif "--tab4" in sys.argv:   # UI 테스트용: 우글거림 수정 탭으로 바로 로드
            select_tab(3)
            root.attributes("-topmost", True)
            root.after(200, pages[3].add_files, initial_files)
        elif "--tab2" in sys.argv:   # UI 테스트용: 실사 질감 탭으로 바로 로드
            select_tab(1)
            if "--autorun" in sys.argv:
                root.attributes("-topmost", True)
            root.after(200, pages[1].add_files, initial_files)
        elif "--tab3" in sys.argv:   # UI 테스트용: Remove GPT 탭으로 바로 로드
            select_tab(2)
            root.after(200, pages[2].add_files, initial_files)
            if "--autorun" in sys.argv:
                root.attributes("-topmost", True)
                root.after(900, pages[2].run_convert)
        else:
            root.after(200, pages[0].add_files, initial_files)

    check_update_async(root)        # 사일런트 자동 업데이트 (EXE만)

    if "--buildtest" in sys.argv:   # 모든 탭 생성 검증 후 자동 종료
        root.after(600, root.destroy)

    root.mainloop()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    silent = "--silent" in sys.argv

    if "--selftest" in sys.argv:
        try:
            from tkinterdnd2 import TkinterDnD
            r = TkinterDnD.Tk()
            r.withdraw()
            r.update()
            r.destroy()
            print("SELFTEST OK dnd=1")
        except Exception as e:
            print("SELFTEST FAIL", e)
            sys.exit(1)
        sys.exit(0)

    if silent:
        if "--removegpt-strong" in sys.argv:
            fn, suffix = (lambda im, progress=None: remove_gpt(im, True, progress)), "_nogpt_strong"
        elif "--removegpt" in sys.argv:
            fn, suffix = (lambda im, progress=None: remove_gpt(im, False, progress)), "_nogpt"
        elif "--photoreal" in sys.argv:
            fn, suffix = photoreal, "_photoreal"
        else:
            fn, suffix = deband, "_debanded"
        code = 0
        for p in args:
            try:
                if "--detailboost" in sys.argv and p.lower().endswith(VEXTS):
                    dst = detail_boost_video(p)
                elif "--sizzlefix" in sys.argv and p.lower().endswith(VEXTS):
                    dst = fix_sizzle_video(p)
                elif p.lower().endswith(VEXTS):
                    dst = stack_deband_video(p)
                else:
                    dst = process_file(p, fn, suffix)
                print("OK", dst)
            except Exception as e:
                print("FAIL", p, e)
                code = 1
        sys.exit(code)

    try:
        run_gui(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
