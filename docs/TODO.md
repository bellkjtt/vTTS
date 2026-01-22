# vTTS TODO List

## 🎯 Phase 1: 핵심 기능 구현 (현재)

### ✅ 완료
- [x] 프로젝트 구조 설계
- [x] 베이스 엔진 인터페이스 (`BaseTTSEngine`)
- [x] 엔진 레지스트리 시스템
- [x] Supertonic 엔진 기본 구현
- [x] FastAPI 서버 구조
- [x] OpenAI 호환 API 엔드포인트
- [x] Python 클라이언트
- [x] CLI 인터페이스
- [x] 기본 문서

### 🚧 진행 중
- [ ] **CosyVoice3 엔진 구현** (우선순위: 높음)
  - [ ] 모델 로더
  - [ ] Zero-shot 음성 복제
  - [ ] 다국어 지원
  - [ ] 방언/억양 지원
  
- [ ] **GPT-SoVITS 엔진 구현** (우선순위: 높음)
  - [ ] 모델 로더
  - [ ] Few-shot 음성 복제
  - [ ] 참조 오디오 처리

- [ ] **Supertonic 엔진 완성** (우선순위: 중간)
  - [ ] 실제 supertonic 패키지 연동
  - [ ] ONNX Runtime 최적화
  - [ ] 다국어 테스트

## 🚀 Phase 2: 고급 기능

### 스트리밍
- [ ] 서버 사이드 스트리밍 구현
- [ ] 청크 단위 오디오 생성
- [ ] WebSocket 지원 (선택)
- [ ] Server-Sent Events (SSE) 지원

### 성능 최적화
- [ ] 배치 추론 지원
- [ ] 모델 캐싱 전략
- [ ] GPU 메모리 관리
- [ ] 동적 배치 크기 조절
- [ ] KV 캐시 최적화

### 모델 관리
- [ ] 자동 모델 다운로드
- [ ] 모델 버전 관리
- [ ] 모델 핫 스왑 (무중단 모델 교체)
- [ ] 멀티 모델 동시 서빙
- [ ] 모델 양자화 지원

## 🧪 Phase 3: 품질 & 안정성

### 테스트
- [ ] 단위 테스트 (pytest)
- [ ] 통합 테스트
- [ ] 성능 벤치마크
- [ ] 부하 테스트 (locust)
- [ ] 엔진별 테스트

### CI/CD
- [ ] GitHub Actions 설정
- [ ] 자동 테스트
- [ ] Docker 이미지 빌드
- [ ] PyPI 자동 배포

### 모니터링
- [ ] Prometheus 메트릭
- [ ] 로깅 개선
- [ ] 에러 추적 (Sentry)
- [ ] 성능 프로파일링

## 📦 Phase 4: 배포 & 확장

### Docker
- [ ] 멀티 스테이지 빌드
- [ ] 이미지 크기 최적화
- [ ] GPU/CPU 별도 이미지
- [ ] Docker Compose 예제

### Kubernetes
- [ ] Helm 차트
- [ ] HPA (Horizontal Pod Autoscaling)
- [ ] Istio 통합

### 클라우드 배포
- [ ] AWS 배포 가이드
- [ ] GCP 배포 가이드
- [ ] Azure 배포 가이드

## 🎨 Phase 5: 사용성 개선

### CLI 개선
- [ ] 인터랙티브 모드
- [ ] 진행률 표시
- [ ] 컬러 출력 개선
- [ ] 자동 완성 (argcomplete)

### 문서
- [ ] API 문서 자동 생성
- [ ] 튜토리얼 비디오
- [ ] 블로그 포스트
- [ ] 다국어 문서 (영어, 한국어)

### UI
- [ ] Gradio 웹 UI (선택)
- [ ] Streamlit 데모 (선택)
- [ ] Swagger UI 커스터마이징

## 🌟 Phase 6: 추가 기능

### 새 엔진 지원
- [ ] StyleTTS2
- [ ] XTTS-v2
- [ ] Bark
- [ ] VALL-E X
- [ ] NaturalSpeech 3

### 고급 기능
- [ ] 음성 감정 조절
- [ ] 피치/톤 조절
- [ ] 배경음악 합성
- [ ] 다중 화자 대화 생성
- [ ] 음성 편집 (잘라내기, 붙이기)

### 통합
- [ ] LangChain 통합
- [ ] LlamaIndex 통합
- [ ] ComfyUI 노드
- [ ] Blender 플러그인

## 📊 메트릭 & 목표

### 성능 목표
- [ ] RTF (Real-time Factor) < 0.1
- [ ] P99 레이턴시 < 500ms
- [ ] 동시 요청 > 100
- [ ] GPU 활용률 > 80%

### 품질 목표
- [ ] 테스트 커버리지 > 80%
- [ ] 문서 완성도 > 90%
- [ ] Zero critical bugs

## 🐛 알려진 이슈

1. **Supertonic 엔진**: 실제 supertonic 패키지 API 확인 필요
2. **오디오 인코딩**: pydub 대신 더 빠른 라이브러리 고려
3. **모델 다운로드**: 대용량 모델 다운로드 시간 최적화 필요

## 💡 아이디어

- [ ] 모델 앙상블 (여러 TTS 모델 결합)
- [ ] 자동 언어 감지
- [ ] 텍스트 전처리 개선 (숫자, 특수문자 등)
- [ ] 발음 사전 지원
- [ ] SSML (Speech Synthesis Markup Language) 지원

---

**업데이트**: 2026-01-20
**담당자**: vTTS Team
