# -*- coding: utf-8 -*-
"""
depth_extract.py — 영상에서 시간축 안정화된 뎁스맵을 뽑아 .npz로 저장.
No photoshop 포커싱 탭이 subprocess로 호출 (torch가 있는 시스템 파이썬에서 실행).

출력 npz: depth (uint16 [N,H,W], 전역 정규화 0..65535, 1=가까움), fps, n, h, w
진행상황은 stdout에 'PROG i/n' 로 출력.

Usage: python depth_extract.py <video> <out.npz> [--model small|base]
"""
import sys, os, time
import numpy as np
import cv2


def _hf(fn, *a, **kw):
    """HF 로드: 로컬 캐시 우선(허브 접속 없음 — 네트워크가 막히면 무한대기하므로),
    캐시에 없을 때만 온라인 다운로드 시도."""
    try:
        return fn(*a, local_files_only=True, **kw)
    except Exception:
        return fn(*a, **kw)


def _guided_color(I, p, r, eps):
    """3채널 컬러 가이드 guided filter (He et al.) — 털/머리카락 가닥까지
    컬러 대비로 매트를 정렬한다 (그레이 가이드는 가닥을 못 잡음)."""
    k = (2 * r + 1, 2 * r + 1)
    bf = lambda x: cv2.boxFilter(x, -1, k)
    mI = [bf(I[..., c]) for c in range(3)]
    mp = bf(p)
    cov = [bf(I[..., c] * p) - mI[c] * mp for c in range(3)]
    S = {}
    for i in range(3):
        for j in range(i, 3):
            S[(i, j)] = bf(I[..., i] * I[..., j]) - mI[i] * mI[j] + (eps if i == j else 0.0)
    a, b, c_, d_, e_, f_ = S[(0, 0)], S[(0, 1)], S[(0, 2)], S[(1, 1)], S[(1, 2)], S[(2, 2)]
    A = d_ * f_ - e_ * e_; B = c_ * e_ - b * f_; C = b * e_ - c_ * d_
    D = a * f_ - c_ * c_; E = b * c_ - a * e_; F = a * d_ - b * b
    det = a * A + b * B + c_ * C
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    ax = (A * cov[0] + B * cov[1] + C * cov[2]) / det
    ay = (B * cov[0] + D * cov[1] + E * cov[2]) / det
    az = (C * cov[0] + E * cov[1] + F * cov[2]) / det
    bb = mp - ax * mI[0] - ay * mI[1] - az * mI[2]
    return bf(ax) * I[..., 0] + bf(ay) * I[..., 1] + bf(az) * I[..., 2] + bf(bb)


def refine_matte(d, img01):
    """refine-edge 데이터 생성 — 물체 경계의 불확실 밴드를 이진 커밋 대신
    (전경깊이 dnear, 배경깊이 dfar, 소프트 매트 M)로 반환. 털 가닥처럼
    픽셀이 전/후경에 걸친 곳은 M(0..1)이 가닥 텍스처를 담고, DoF 엔진이
    그 비율로 이중 소속 렌더링한다 (AE 로토브러쉬 refine edge와 같은 원리).
    d: float32 0..1 (1=가까움), img01: float32 BGR 0..1."""
    r = max(5, min(d.shape) // 120)
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    emin = cv2.erode(d, k)
    emax = cv2.dilate(d, k)
    rng = emax - emin
    m0 = ((d - emin) > rng * 0.5).astype(np.float32)   # 러프 FG
    M = np.clip(_guided_color(img01, m0, 2, 1e-4), 0.0, 1.0)
    zone = np.clip((rng - 0.08) / 0.06, 0.0, 1.0)      # 큰 절벽 주변만
    zone = cv2.GaussianBlur(zone, (0, 0), 1.5)
    dnear = d * (1.0 - zone) + emax * zone
    dfar = d * (1.0 - zone) + emin * zone
    M = (1.0 - zone) + M * zone                        # 존 밖 = 단일 소속(M=1)
    # 호환/보조용 커밋 뎁스 (엔진의 경계존 맵 등에 사용) — 작은 얼룩 제거
    dec = (M > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dec, 8)
    for l in range(1, n):
        if st[l, cv2.CC_STAT_AREA] < 4:
            dec[lab == l] = 0
    inv = (1 - dec).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 8)
    for l in range(1, n):
        if st[l, cv2.CC_STAT_AREA] < 4:
            dec[lab == l] = 1
    dhard = np.where(dec > 0, dnear, dfar)
    return dhard, dnear, dfar, M


def _subject_band(d):
    """근경 후보 밴드 — 강한 뎁스 절벽(실루엣) 주변 12~48px의 근경 픽셀.
    바닥처럼 절벽 없이 이어지는 면은 자연 배제."""
    H, W = d.shape
    r = max(5, min(H, W) // 120)
    kk = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    rng = cv2.dilate(d, kk) - cv2.erode(d, kk)
    zone = (rng > 0.15).astype(np.uint8)
    ring = (cv2.dilate(zone, np.ones((97, 97), np.uint8)) -
            cv2.dilate(zone, np.ones((25, 25), np.uint8)))
    return ((d > np.percentile(d, 75)) & (ring > 0)).astype(np.uint8)


def _auto_subject_points(d, k=3):
    """피사체 자동 선택 포인트 — 근경 밴드의 최대 덩어리(중앙 가중)에서 내부점 추출."""
    H, W = d.shape
    cand = _subject_band(d)
    n, lab, st, cent = cv2.connectedComponentsWithStats(cand, 8)
    best, bs = None, -1.0
    for l in range(1, n):
        area = st[l, cv2.CC_STAT_AREA]
        if area < H * W * 0.005:
            continue
        cw = 1.0 - 0.5 * abs(cent[l][0] - W / 2) / (W / 2)   # 중앙 가중
        if area * cw > bs:
            bs, best = area * cw, (lab == l).astype(np.uint8)
    if best is None:
        return []
    dt = cv2.distanceTransform(best, cv2.DIST_L2, 5)
    pts = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        if dt[y, x] < 5:
            break
        pts.append([int(x), int(y)])
        cv2.circle(dt, (int(x), int(y)), 120, 0, -1)
    return pts


def sam2_subject_mattes(frames, d0, dev, pts=None, neg=None):
    """SAM2 비디오 전파(시간축 메모리 → 프레임 간 안정) + ViTMatte(트라이맵 매팅
    → 머리카락/털 낱올 알파). 피사체 경계 밴드의 (알파, 밴드가중) 리스트 반환.
    pts/neg(수동 클릭 포인트, 원본 해상도 좌표)가 있으면 자동 선택을 건너뛴다.
    실패/피사체 없음 → None (가이디드 매트 폴백)."""
    import torch
    from transformers import (Sam2VideoModel, Sam2VideoProcessor,
                              VitMatteForImageMatting, VitMatteImageProcessor)
    from PIL import Image
    manual = bool(pts)
    if not manual:
        pts = _auto_subject_points(d0)
        if not pts:
            return None
    pts = [list(p) for p in pts]
    neg = [list(p) for p in (neg or [])]
    H, W = frames[0].shape[:2]
    rgb = []
    for k, f in enumerate(frames):
        rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if (k + 1) % 24 == 0:
            print(f"PREP {k + 1}/{len(frames)}", flush=True)
    MID = "facebook/sam2.1-hiera-small"
    print("SAMLOAD", flush=True)
    sproc = _hf(Sam2VideoProcessor.from_pretrained, MID)
    smodel = _hf(Sam2VideoModel.from_pretrained, MID, dtype=torch.bfloat16).to(dev).eval()
    print("SAMINIT", flush=True)
    sess = sproc.init_video_session(video=rgb, inference_device=dev, dtype=torch.bfloat16)

    def _prompt():
        ap = pts + neg
        sproc.add_inputs_to_inference_session(sess, frame_idx=0, obj_ids=1,
                                              input_points=[[ap]],
                                              input_labels=[[[1] * len(pts) + [0] * len(neg)]])
    _prompt()
    if manual:
        smodel(inference_session=sess, frame_idx=0)
    else:
        # 커버리지 보강 — 초기 점이 한 부위(예: 갑옷)에 몰리면 SAM2가 부위 마스크만
        # 반환해 얼굴 등이 잘린다. 근경 밴드 중 마스크 인접 미커버 영역에 점을
        # 추가해 인물 전체 마스크로 승격시킨다.
        band0 = _subject_band(d0)
        for _ in range(3):
            o0 = smodel(inference_session=sess, frame_idx=0)
            m0 = (sproc.post_process_masks([o0.pred_masks],
                  original_sizes=[[H, W]])[0][0, 0] > 0).cpu().numpy().astype(np.uint8)
            near = cv2.dilate(m0, np.ones((151, 151), np.uint8))
            miss = ((band0 > 0) & (near > 0) &
                    (cv2.dilate(m0, np.ones((25, 25), np.uint8)) == 0)).astype(np.uint8)
            n2, lab2, st2, _c = cv2.connectedComponentsWithStats(miss, 8)
            big = sorted((l for l in range(1, n2)
                          if st2[l, cv2.CC_STAT_AREA] >= H * W * 0.004),
                         key=lambda l: -st2[l, cv2.CC_STAT_AREA])
            add = []
            for l in big[:2]:
                dt = cv2.distanceTransform((lab2 == l).astype(np.uint8), cv2.DIST_L2, 5)
                y, x = np.unravel_index(np.argmax(dt), dt.shape)
                if dt[y, x] >= 4:
                    add.append([int(x), int(y)])
            if not add:
                break
            pts = pts + add
            print(f"SAMPT {len(pts)}", flush=True)
            _prompt()
    masks = {}
    for o in smodel.propagate_in_video_iterator(sess):
        mk = sproc.post_process_masks([o.pred_masks], original_sizes=[[H, W]])[0]
        masks[o.frame_idx] = (mk[0, 0] > 0).cpu().numpy().astype(np.uint8)
        if len(masks) % 4 == 0 or len(masks) == len(frames):
            print(f"SAMP {len(masks)}/{len(frames)}", flush=True)
    del smodel, sess
    torch.cuda.empty_cache()

    VID = "hustvl/vitmatte-small-composition-1k"
    print("MLOAD", flush=True)
    vproc = _hf(VitMatteImageProcessor.from_pretrained, VID)
    vmodel = _hf(VitMatteForImageMatting.from_pretrained, VID).to(dev).eval()
    kb = np.ones((25, 25), np.uint8)
    kd = np.ones((73, 73), np.uint8)   # 바깥쪽 ±36px — 긴 털도 불확실 영역에 포함
    out = []
    N = len(frames)
    for i in range(N):
        m = masks.get(i)
        if m is None or m.sum() < 64:
            out.append(None)
            continue
        fg = cv2.erode(m, kb)
        dil = cv2.dilate(m, kd)
        tri = np.zeros((H, W), np.uint8)
        tri[(dil - fg) > 0] = 128
        tri[fg > 0] = 255
        inp = vproc(images=Image.fromarray(rgb[i]), trimaps=Image.fromarray(tri),
                    return_tensors="pt").to(dev)
        with torch.no_grad():
            alpha = vmodel(**inp).alphas[0, 0].cpu().numpy()[:H, :W]
        band = cv2.GaussianBlur(((dil - fg) > 0).astype(np.float32), (0, 0), 3.0)
        out.append((np.clip(alpha, 0.0, 1.0), np.clip(band, 0.0, 1.0), m))
        print(f"MATTE {i + 1}/{N}", flush=True)
    del vmodel
    torch.cuda.empty_cache()
    return out


def serve_points():
    """실시간 포인트 미리보기 서버 — 배경 제거 탭이 점을 찍을 때마다 현재
    프레임의 SAM2 이미지 마스크를 즉시 계산해 돌려준다 (전파/매팅 없음).
    stdin 프로토콜(탭 구분): SEG\t<frame.npy>\t<mask.npy>\t<pts x,y;..|->\t<neg|->
    응답: 모델 로드 완료 시 READY 1회, 이후 요청마다 OK 또는 ERR <msg>."""
    import torch
    from transformers import Sam2Model, Sam2Processor
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    MID = "facebook/sam2.1-hiera-small"
    proc = _hf(Sam2Processor.from_pretrained, MID)
    model = _hf(Sam2Model.from_pretrained, MID).to(dev).eval()
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            f = line.rstrip("\n").split("\t")
            if f[0] != "SEG" or len(f) < 5:
                continue
            img = np.load(f[1])                    # BGR uint8
            H, W = img.shape[:2]
            pp = [[int(v) for v in p.split(",")] for p in f[3].split(";") if p and p != "-"]
            nn = [[int(v) for v in p.split(",")] for p in f[4].split(";") if p and p != "-"]
            if not pp:
                np.save(f[2], np.zeros((H, W), np.uint8))
                print("OK", flush=True)
                continue
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inp = proc(images=pil, input_points=[[pp + nn]],
                       input_labels=[[[1] * len(pp) + [0] * len(nn)]],
                       return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model(**inp, multimask_output=False)
            pm = o.pred_masks.float().cpu()
            while pm.dim() > 3:                    # (…,M,h,w) → (M,h,w)
                pm = pm[0]
            m = torch.nn.functional.interpolate(pm[None], size=(H, W), mode="bilinear",
                                                align_corners=False)[0, 0]
            np.save(f[2], (m > 0).numpy().astype(np.uint8) * 255)
            print("OK", flush=True)
        except Exception as e:
            print(f"ERR {e}", flush=True)


def main():
    if "--serve" in sys.argv:
        serve_points()
        return
    src = sys.argv[1]
    out = sys.argv[2]
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 기본값: GPU면 large(공간감·부위분리 최상, VRAM 부족 시 base 폴백), CPU는 small
    model_key = "small"
    if dev == "cuda":
        try:
            free = torch.cuda.mem_get_info()[0] / 1e9
        except Exception:
            free = 8.0
        model_key = "large" if free > 3.5 else "base"
    if "--model" in sys.argv:
        model_key = sys.argv[sys.argv.index("--model") + 1]
    size = 770 if dev == "cuda" else 518
    if "--size" in sys.argv:
        size = int(sys.argv[sys.argv.index("--size") + 1])

    def _parse_pts(flag):
        """--pts/--neg "x,y;x,y" → [[x,y],...] (원본 해상도 좌표)"""
        if flag not in sys.argv:
            return None
        s = sys.argv[sys.argv.index(flag) + 1]
        return [[int(float(v)) for v in p.split(",")] for p in s.split(";") if p]
    upts = _parse_pts("--pts")
    uneg = _parse_pts("--neg")
    matteonly = "--matteonly" in sys.argv   # 배경 제거 탭: 비디오 뎁스 생략
    if matteonly:
        model_key = "small"                 # 자동 포인트용 프레임0 뎁스만 필요

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    MODELS = {"small": "depth-anything/Depth-Anything-V2-Small-hf",
              "base": "depth-anything/Depth-Anything-V2-Base-hf",
              "large": "depth-anything/Depth-Anything-V2-Large-hf"}
    proc = model = None
    if not (matteonly and upts):
        print(f"LOAD {MODELS[model_key]} {dev} {size}px", flush=True)
        proc = _hf(AutoImageProcessor.from_pretrained, MODELS[model_key],
                   size={"height": size, "width": size})
        model = _hf(AutoModelForDepthEstimation.from_pretrained, MODELS[model_key]).to(dev).eval()
        if dev == "cuda":
            model = model.half()

    cap = cv2.VideoCapture(src)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    frames, grays = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if not matteonly:              # 배경 제거 경로는 grays 미사용 — RAM 절약
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if len(frames) % 12 == 0:
            print(f"DEC {len(frames)}/{total}", flush=True)
    cap.release()
    N = len(frames)
    if N == 0:
        print("ERR no frames", flush=True); sys.exit(1)
    print(f"FRAMES {N} {W}x{H}", flush=True)

    if matteonly:
        # 배경 제거 탭 전용 고속 경로 — 비디오 뎁스/안정화/refine 전부 생략,
        # SAM2 전파 + ViTMatte 매팅만 수행해 salpha/smask를 저장한다.
        d0 = None
        if not upts:
            rgb0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            inp = proc(images=[rgb0], return_tensors="pt").to(dev)
            if dev == "cuda":
                inp = {k: (v.half() if v.dtype == torch.float32 else v)
                       for k, v in inp.items()}
            with torch.no_grad():
                dd = model(**inp).predicted_depth
            dd = torch.nn.functional.interpolate(
                dd[:, None].float(), size=(H, W),
                mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
            lo0, hi0 = np.percentile(dd, 1), np.percentile(dd, 99)
            d0 = np.clip((dd - lo0) / (hi0 - lo0 + 1e-6), 0, 1).astype(np.float32)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
        try:
            print("SAM", flush=True)
            sm = sam2_subject_mattes(frames, d0, dev, pts=upts, neg=uneg)
        except Exception as e:
            print(f"SAMSKIP {e}", flush=True)
            sm = None
        extra = {}
        if sm is not None and any(s is not None for s in sm):
            smask = np.zeros((N, H, W), np.uint8)
            salpha = np.zeros((N, H, W), np.uint8)
            for i in range(N):
                if sm[i] is not None:
                    smask[i] = sm[i][2]
                    salpha[i] = (sm[i][0] * 255.0 + 0.5).astype(np.uint8)
            extra = {"smask": smask, "salpha": salpha}
        print("SAVE", flush=True)
        np.savez_compressed(out, fps=np.float32(fps), n=np.int32(N),
                            h=np.int32(H), w=np.int32(W), **extra)
        print(f"DONE {out}", flush=True)
        return

    depths = None
    if "--novda" not in sys.argv and dev == "cuda":
        try:
            # Video-Depth-Anything: 비디오 네이티브 시간축 일관 뎁스 (경계 숨쉬임↓)
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vda"))
            from video_depth_anything.video_depth import VideoDepthAnything
            from huggingface_hub import hf_hub_download
            print("LOAD Video-Depth-Anything-Small cuda", flush=True)
            vdam = VideoDepthAnything(encoder="vits", features=64,
                                      out_channels=[48, 96, 192, 384])
            try:
                ck = hf_hub_download("depth-anything/Video-Depth-Anything-Small",
                                     "video_depth_anything_vits.pth", local_files_only=True)
            except Exception:
                ck = hf_hub_download("depth-anything/Video-Depth-Anything-Small",
                                     "video_depth_anything_vits.pth")
            vdam.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
            vdam = vdam.to(dev).eval()
            rgb_all = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], 0)
            dts, _ = vdam.infer_video_depth(rgb_all, float(fps), input_size=518, device=dev)
            depths = np.array(dts, dtype=np.float32)
            del vdam, rgb_all
            torch.cuda.empty_cache()
            print(f"PROG {N}/{N}", flush=True)
        except Exception as e:
            print(f"VDASKIP {e}", flush=True)
            depths = None

    vda_ok = depths is not None
    if depths is None:
        depths = np.empty((N, H, W), np.float32)
    B = 8
    for s in ([] if vda_ok else range(0, N, B)):
        rgb = [cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB) for k in range(s, min(s + B, N))]
        inp = proc(images=rgb, return_tensors="pt").to(dev)
        if dev == "cuda":
            inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
        with torch.no_grad():
            d = model(**inp).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None].float(), size=(H, W),
                                            mode="bicubic", align_corners=False)[:, 0]
        depths[s:s + len(rgb)] = d.cpu().numpy()
        print(f"PROG {min(s + B, N)}/{N}", flush=True)

    # 시간축 안정화 (플로우 정렬 EMA, 양방향) — VDA는 자체 일관성이 있어 생략
    if not vda_ok:
        print("STAB", flush=True)
    _rng = (np.percentile(depths, 99) - np.percentile(depths, 1)) + 1e-6
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def warp(prev, fl):
        return cv2.remap(prev, gx + fl[..., 0], gy + fl[..., 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if not vda_ok:
        for i in range(1, N):
            fl = dis.calc(grays[i], grays[i - 1], None); wp = warp(depths[i - 1], fl)
            a = np.clip(0.45 + 0.55 * np.clip(np.abs(wp - depths[i]) / _rng / 0.10, 0, 1), 0.45, 1.0)
            depths[i] = a * depths[i] + (1 - a) * wp
        for i in range(N - 2, -1, -1):
            fl = dis.calc(grays[i], grays[i + 1], None); wn = warp(depths[i + 1], fl)
            a = np.clip(0.6 + 0.4 * np.clip(np.abs(wn - depths[i]) / _rng / 0.10, 0, 1), 0.6, 1.0)
            depths[i] = a * depths[i] + (1 - a) * wn

    lo, hi = np.percentile(depths, 1), np.percentile(depths, 99)
    dn = np.clip((depths - lo) / (hi - lo + 1e-6), 0, 1)   # 1=near 0=far

    if "--nosnap" not in sys.argv:                         # 경계 정밀화 (refine edge)
        print("SNAP", flush=True)
        dnear = np.empty_like(dn)
        dfar = np.empty_like(dn)
        matf = np.empty((N, H, W), np.float32)
        for i in range(N):
            dh, dnr, dfr, M = refine_matte(dn[i], frames[i].astype(np.float32) / 255.0)
            dnear[i] = dnr; dfar[i] = dfr; matf[i] = M
        if "--nosam" not in sys.argv and dev == "cuda":
            try:                                           # 피사체 SAM2+ViTMatte 매팅
                print("SAM", flush=True)
                sm = sam2_subject_mattes(frames, dn[0], dev)
            except Exception as e:
                print(f"SAMSKIP {e}", flush=True)
                sm = None
            smask = salpha = None
            if sm is not None:
                smask = np.zeros((N,) + dn.shape[1:], np.uint8)
                salpha = np.zeros((N,) + dn.shape[1:], np.uint8)   # ViTMatte 원본 알파
                for i in range(N):
                    if sm[i] is not None:
                        smask[i] = sm[i][2]
                        salpha[i] = (sm[i][0] * 255.0 + 0.5).astype(np.uint8)
                kbox = (51, 51)
                for i in range(N):
                    if sm[i] is None:
                        continue
                    alpha, band, msk = sm[i]
                    # M의 의미는 '로컬 근경(dnear) 소속 비율'인데 SAM2 알파는 '피사체
                    # 비율'이다. 피사체보다 가까운 물체(예: 들고 있는 해머) 위로 밴드가
                    # 지나가면 방향이 반대가 되어 흐린 물체 안에 딱딱한 경계가 생기므로,
                    # 주변 피사체 깊이와 dnear/dfar를 비교해 알파 방향을 정렬한다.
                    m8 = msk.astype(np.float32)
                    ds = (cv2.boxFilter(dn[i] * m8, -1, kbox) /
                          np.maximum(cv2.boxFilter(m8, -1, kbox), 1e-4))
                    near_subj = np.abs(ds - dnear[i]) <= np.abs(ds - dfar[i])
                    a2 = np.where(near_subj, alpha, 1.0 - alpha)
                    # max 병합: ViTMatte(부드러운 경계) + 가이디드 매트(낱가닥 감도)
                    matf[i] = matf[i] * (1.0 - band) + np.maximum(a2, matf[i]) * band
        if N >= 3 and "--nomstab" not in sys.argv:
            # 매트 시간축 안정화 (플로우 정렬 EMA, 양방향) — 매트는 프레임별 독립
            # 계산이라 피사체가 미세하게 움직이면 경계가 1px씩 어긋나며 떨리고,
            # 그게 렌더에서 실루엣 보석·엣지의 깜빡임이 된다.
            print("MSTAB", flush=True)
            for i in range(1, N):
                fl = dis.calc(grays[i], grays[i - 1], None)
                matf[i] = 0.55 * matf[i] + 0.45 * warp(matf[i - 1], fl)
            for i in range(N - 2, -1, -1):
                fl = dis.calc(grays[i], grays[i + 1], None)
                matf[i] = 0.6 * matf[i] + 0.4 * warp(matf[i + 1], fl)
        mat = np.empty((N, H, W), np.uint8)
        for i in range(N):
            M = np.clip(matf[i], 0.0, 1.0)
            mat[i] = (M * 255 + 0.5).astype(np.uint8)
            # 커밋 뎁스도 안정화된 매트 기준으로 재계산
            dn[i] = np.where(M > 0.5, dnear[i], dfar[i])
        extra = ({"smask": smask, "salpha": salpha} if smask is not None else {})
        np.savez_compressed(out, depth=(dn * 65535 + 0.5).astype(np.uint16),
                            dnear=(dnear * 65535 + 0.5).astype(np.uint16),
                            dfar=(dfar * 65535 + 0.5).astype(np.uint16),
                            matte=mat, fps=np.float32(fps),
                            n=np.int32(N), h=np.int32(H), w=np.int32(W), **extra)
        print(f"DONE {out}", flush=True)
        return

    depth16 = (dn * 65535 + 0.5).astype(np.uint16)
    np.savez_compressed(out, depth=depth16, fps=np.float32(fps),
                        n=np.int32(N), h=np.int32(H), w=np.int32(W))
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
