# v0.1.0 릴리스 체크리스트

## 📋 릴리스 전 준비

### 코드 정리
- [ ] 모든 파일에서 `YOUR_USERNAME` 교체
- [ ] 버전 번호 확인 (pyproject.toml, setup.py)
- [ ] LICENSE 파일 확인
- [ ] .gitignore 확인

### 문서화
- [ ] README.md 완성도 확인
- [ ] QUICKSTART.md 확인
- [ ] INSTALL.md 확인
- [ ] TODO.md 업데이트
- [ ] 예제 코드 동작 확인

### 테스트
- [ ] 로컬에서 설치 테스트
  ```bash
  pip install -e .
  vtts --help
  ```
- [ ] Kaggle 노트북 테스트
- [ ] 기본 기능 동작 확인

## 🚀 GitHub 설정

### 저장소 생성
- [ ] GitHub에 vTTS 저장소 생성
- [ ] Repository 설명 작성
- [ ] Topics 추가: `tts`, `stt`, `speech`, `ai`, `machine-learning`, `openai`, `huggingface`

### 코드 업로드
```bash
# 1. Git 초기화
git init
git add .
git commit -m "Initial commit: vTTS v0.1.0"

# 2. GitHub 연결
git remote add origin https://github.com/YOUR_USERNAME/vTTS.git
git branch -M main
git push -u origin main

# 3. 태그 생성
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### 스폰서 설정
- [ ] `.github/FUNDING.yml` 수정
- [ ] GitHub Sponsors 활성화
- [ ] 스폰서 티어 설정
- [ ] Ko-fi 계정 연결 (선택)

### GitHub Actions
- [ ] `.github/workflows/test.yml` 확인
- [ ] `.github/workflows/release.yml` 확인
- [ ] Actions 실행 확인

## 📦 릴리스 생성

### GitHub Release
- [ ] Releases → Create a new release
- [ ] Tag: v0.1.0
- [ ] Title: vTTS v0.1.0 - Initial Release
- [ ] 릴리스 노트 작성
- [ ] Publish release

### PyPI 배포 (선택적)
```bash
# 1. PyPI 계정 생성
# https://pypi.org/account/register/

# 2. API 토큰 생성
# https://pypi.org/manage/account/token/

# 3. 빌드
python -m build

# 4. 업로드
python -m twine upload dist/*
```

## 🧪 Kaggle 테스트

### 노트북 업로드
- [ ] Kaggle에 노트북 업로드
- [ ] Public으로 설정
- [ ] 노트북 실행 확인
- [ ] 결과 검증

### Dataset 생성 (선택)
- [ ] vTTS 폴더 압축
- [ ] Kaggle Dataset 업로드
- [ ] 노트북에서 테스트

## 📣 홍보

### README 배지 추가
- [ ] Release 배지
- [ ] License 배지
- [ ] Python 버전 배지
- [ ] Sponsors 배지

### 커뮤니티 공유
- [ ] Reddit (r/MachineLearning)
- [ ] Hacker News
- [ ] Twitter/X
- [ ] GeekNews
- [ ] MLOps Korea

### 문서 사이트 (선택)
- [ ] MkDocs 설정
- [ ] GitHub Pages 배포
- [ ] 도메인 연결

## ✅ 릴리스 후

### 모니터링
- [ ] GitHub Issues 모니터링
- [ ] Discussions 활성화
- [ ] Pull Request 검토
- [ ] Stars/Forks 추적

### 다음 버전 계획
- [ ] v0.2.0 로드맵 작성
- [ ] Issue 템플릿 생성
- [ ] Contributing 가이드 작성

## 🎯 성공 지표

- [ ] 100+ GitHub Stars
- [ ] 첫 번째 외부 Contributor
- [ ] 첫 번째 스폰서
- [ ] 100+ PyPI 다운로드 (배포 시)

---

**릴리스 담당자**: ___________
**릴리스 날짜**: ___________
**체크리스트 완료**: [ ]
