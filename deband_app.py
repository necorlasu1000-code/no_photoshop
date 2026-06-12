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
import queue
import threading
import time
import traceback

import numpy as np
import cv2

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
VEXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
APP_TITLE = "no photoshop — 이미지 후처리"
APP_VER = "ver 0.2"

COL = {
    "bg":     "#0e0f10",
    "card":   "#161719",
    "card2":  "#1b1c1f",
    "border": "#2c2e31",
    "dash":   "#3a3d41",
    "lime":   "#d6f25e",
    "lime2":  "#c4e348",
    "text":   "#e9ebed",
    "sub":    "#9a9da1",
    "dim":    "#6f7378",
    "ok":     "#8fd964",
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
    from tkinter import filedialog

    dnd_ok = False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        dnd_ok = True
    except Exception:
        root = tk.Tk()

    root.title(APP_TITLE)
    root.geometry("1020x780")
    root.minsize(780, 640)
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

    # ---------- 세그먼트 탭 ----------
    tabbar = tk.Canvas(root, height=58, bg=COL["bg"], highlightthickness=0)
    tabbar.pack(fill="x", padx=24, pady=(18, 0))
    pages = {}
    current = {"idx": 0}
    TABS = ["밴딩 제거", "실사 질감", "Remove GPT", "우글거림 수정", "디테일 보강"]
    TAB_W, TAB_H, PILL_PAD = 122, 40, 4

    def draw_tabs(_=None):
        tabbar.delete("all")
        total_w = TAB_W * len(TABS) + PILL_PAD * 2
        rrect(tabbar, 0, 6, total_w, 6 + TAB_H + PILL_PAD * 2, 14,
              fill=COL["card"], outline=COL["border"])
        for i, name in enumerate(TABS):
            x0 = PILL_PAD + i * TAB_W
            y0 = 6 + PILL_PAD
            if i == current["idx"]:
                rrect(tabbar, x0, y0, x0 + TAB_W, y0 + TAB_H, 11,
                      fill=COL["lime"], outline="")
                fg = "#101204"
            else:
                fg = COL["sub"]
            tabbar.create_text(x0 + TAB_W / 2, y0 + TAB_H / 2, text=name,
                               fill=fg, font=(FONT, 11, "bold"))
        # 우측 끝 버전 표시
        bw = tabbar.winfo_width() or 900
        tabbar.create_text(bw - 6, 6 + PILL_PAD + TAB_H / 2, text=APP_VER,
                           fill=COL["dim"], font=(MONO, 9), anchor="e")
    tabbar.bind("<Configure>", draw_tabs)

    def on_tab_click(e):
        total_w = TAB_W * len(TABS) + PILL_PAD * 2
        if 0 <= e.x <= total_w:
            idx = min(len(TABS) - 1, max(0, int((e.x - PILL_PAD) // TAB_W)))
            if idx != current["idx"]:
                current["idx"] = idx
                draw_tabs()
                pages[idx].tkraise()

    tabbar.bind("<Button-1>", on_tab_click)

    body = tk.Frame(root, bg=COL["bg"])
    body.pack(fill="both", expand=True)

    def canvas_button(parent, text, width, height, primary=True):
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
                rrect(cv, 1, 1, width - 1, height - 1, 12, fill=fill, outline="")
            else:
                fill = COL["card2"] if state["hover"] else COL["card"]
                fg = COL["text"] if state["enabled"] else COL["dim"]
                rrect(cv, 1, 1, width - 1, height - 1, 12, fill=fill,
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
            mode_cv = tk.Canvas(mode_row, width=176, height=40, bg=COL["bg"],
                                highlightthickness=0, cursor="hand2")
            mode_cv.pack(side="left")
            MW, MH, MP = 82, 30, 4

            def draw_mode():
                mode_cv.delete("all")
                rrect(mode_cv, 0, 1, MW * 2 + MP * 2, 1 + MH + MP * 2, 11,
                      fill=COL["card"], outline=COL["border"])
                for i, name in enumerate(["IMG", "VIDEO"]):
                    x0 = MP + i * MW
                    active = (i == 1) == mode["video"]
                    if active:
                        rrect(mode_cv, x0, 1 + MP, x0 + MW, 1 + MP + MH, 9,
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

        drop = tk.Canvas(page, bg=COL["bg"], highlightthickness=0, height=250)
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
                        dst = deband_video(p, progress=prog)
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
                         width=280)
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
        drop = tk.Canvas(page, bg=COL["bg"], highlightthickness=0, height=150)
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

        drop = tk.Canvas(page, bg=COL["bg"], highlightthickness=0, height=150)
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

    pages[0] = build_page(
        "그라데이션 밴딩(계단현상)을 부드럽게 풀어줘요 — 디테일은 그대로. 원본 폴더에 ",
        "_debanded.png / .mp4", deband, "_debanded", media=True)
    pages[1] = build_photoreal_page()
    pages[2] = build_removegpt_page()
    pages[3] = build_sizzle_page()
    pages[4] = build_detailboost_page()

    current["idx"] = 0
    draw_tabs()
    pages[0].tkraise()

    if initial_files:
        if "--tab5" in sys.argv:   # UI 테스트용: 디테일 보강 탭으로 바로 로드
            current["idx"] = 4
            draw_tabs()
            pages[4].tkraise()
            root.attributes("-topmost", True)
            root.after(200, pages[4].add_files, initial_files)
        elif "--tab4" in sys.argv:   # UI 테스트용: 우글거림 수정 탭으로 바로 로드
            current["idx"] = 3
            draw_tabs()
            pages[3].tkraise()
            root.attributes("-topmost", True)
            root.after(200, pages[3].add_files, initial_files)
        elif "--tab2" in sys.argv:   # UI 테스트용: 실사 질감 탭으로 바로 로드
            current["idx"] = 1
            draw_tabs()
            pages[1].tkraise()
            if "--autorun" in sys.argv:
                root.attributes("-topmost", True)
            root.after(200, pages[1].add_files, initial_files)
        elif "--tab3" in sys.argv:   # UI 테스트용: Remove GPT 탭으로 바로 로드
            current["idx"] = 2
            draw_tabs()
            pages[2].tkraise()
            root.after(200, pages[2].add_files, initial_files)
            if "--autorun" in sys.argv:
                root.attributes("-topmost", True)
                root.after(900, pages[2].run_convert)
        else:
            root.after(200, pages[0].add_files, initial_files)

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
                    dst = deband_video(p)
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
