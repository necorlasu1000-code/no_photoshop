# No photoshop

AI 생성 이미지·영상의 "AI 티"를 지우는 무설치 후처리 도구. **AI/API 없이** 고전 영상처리만으로 동작하고, 인터넷 연결 없이 오프라인에서 실행됩니다.

`No photoshop.exe` 파일 하나만 전달하면 별도 설치 없이 바로 쓸 수 있습니다 (Windows).

## 기능 (ver 0.2)

| 탭 | 설명 | 출력 |
|---|---|---|
| **밴딩 제거** | 그라데이션 밴딩·계단현상 제거 (IMG/VIDEO). 멀티스케일 엣지 보존 디밴딩 + 재밴딩 방지 디더 | `_debanded.png` / `.mp4` |
| **실사 질감** | 실시간 미리보기 + 효과별 슬라이더로 실사 카메라 룩 (블랙 리프트, 하이라이트 롤오프, 블룸, 색수차, 비네트, 그레인). ARRI·8mm 등 프리셋 | `_photoreal.png` |
| **Remove GPT** | GPT 이미지 특유의 도트 노이즈 제거 (표준/강력), 전후 비교 슬라이더 | `_nogpt.png` |
| **우글거림 수정** | AI 영상의 파티클 끓는 텍스처를 모션보상 시간 안정화 + 모션블러 + 장면 적응 파티클 합성으로 수정 | `_particlefix.mp4` |
| **디테일 보강** | AI 풀샷의 뭉개진 디테일 복원(하늘·구름 자동 보호) + 텍스처 우글거림 고정 + 카메라 저더 보정 + 180° 셔터 모션블러 | `_detail.mp4` |

영상 출력은 10-bit HEVC(yuv420p10le)로 인코딩되고 원본 오디오를 그대로 복사합니다.

## 소스로 실행

```
pip install opencv-python numpy tkinterdnd2 imageio-ffmpeg
python deband_app.py
```

CLI 모드 (GUI 없이):

```
python deband_app.py --silent 이미지.png                # 디밴딩
python deband_app.py --silent --photoreal 이미지.png    # 실사 질감
python deband_app.py --silent --removegpt 이미지.png    # 도트 제거
python deband_app.py --silent --sizzlefix 영상.mp4      # 우글거림 수정
python deband_app.py --silent --detailboost 영상.mp4    # 디테일 보강
```

## EXE 빌드

```
pip install pyinstaller
python -m PyInstaller "No photoshop.spec" --noconfirm
```

빌드 결과는 `dist/No photoshop.exe` (FFmpeg 포함, 단일 파일).
