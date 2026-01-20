# 스폰서 설정 가이드

## 🎯 현재 상태

- ✅ `.github/FUNDING.yml` 파일 생성 완료
- ⏳ **GitHub Sponsors 승인 대기 필요** (1-3일)
- 💡 즉시 사용 가능한 대안 있음

## 📋 GitHub Sponsors 신청 방법

### 1. 신청 페이지 접속
https://github.com/sponsors

### 2. 필수 정보 입력

#### A. 개인 정보
```
- Legal name: [실명]
- Email: [이메일]
- Country: South Korea
- Organization type: Individual
```

#### B. 세금 정보 (W-8BEN)
```
- 한국 거주자는 W-8BEN 양식 작성
- 미국 세금 미적용 확인
- 서명 필요
```

#### C. 지급 정보
```
- Stripe 계정 연결
- 은행 계좌 정보:
  * 은행명
  * 계좌번호
  * 예금주
- 신분증 업로드 (경우에 따라)
```

### 3. 프로필 작성

```markdown
## Headline
오픈소스 TTS/STT 개발자

## Bio
한국어 음성 AI 기술을 모두에게 접근 가능하게 만들고 있습니다.

## Featured work
- vTTS: Universal TTS/STT Serving System
- OpenAI compatible API
```

### 4. 티어 설정

```
Tier 1: $2/month - Coffee ☕
Tier 2: $10/month - Individual 🚀
Tier 3: $25/month - Pro 💎
Tier 4: $100/month - Organization 🏢
```

### 5. 제출 및 대기

```
✅ Submit application
⏳ 승인 대기: 1-3일 (보통 24시간)
📧 승인 이메일 수신
🎉 활성화!
```

---

## 🚀 즉시 사용 가능한 대안

### 옵션 1: Ko-fi (추천)

**장점**:
- ✅ 가입 즉시 사용 가능 (1분)
- ✅ 수수료 0% (PayPal/Stripe 수수료만)
- ✅ One-time & 월간 후원 모두 지원
- ✅ 한국어 지원

**설정**:
1. https://ko-fi.com 접속
2. "Sign Up" 클릭
3. Username: `bellkjtt` 설정
4. `.github/FUNDING.yml` 업데이트:
   ```yaml
   ko_fi: bellkjtt
   ```
5. Git push
6. 즉시 "Sponsor" 버튼 표시!

### 옵션 2: Buy Me a Coffee

**장점**:
- ✅ 매우 간단한 UI
- ✅ 즉시 사용 가능
- ✅ 모바일 앱 지원

**설정**:
1. https://buymeacoffee.com 접속
2. 가입 및 프로필 설정
3. `.github/FUNDING.yml` 업데이트:
   ```yaml
   custom: ["https://buymeacoffee.com/bellkjtt"]
   ```

### 옵션 3: Patreon

**장점**:
- ✅ 크리에이터 플랫폼으로 유명
- ✅ 다양한 티어 및 혜택 관리

**단점**:
- ❌ 수수료 높음 (5-12%)
- ❌ 설정 복잡

---

## 💡 추천 전략

### Phase 1: 즉시 (지금)
```yaml
# .github/FUNDING.yml
github: bellkjtt      # 승인 대기 중
ko_fi: bellkjtt       # 즉시 활성화
```

**액션**:
1. ✅ GitHub Sponsors 신청 제출
2. ✅ Ko-fi 가입 (1분)
3. ✅ FUNDING.yml 업데이트
4. ✅ "Sponsor this project" 즉시 활성화!

### Phase 2: 승인 후 (1-3일 후)
```yaml
# .github/FUNDING.yml  
github: bellkjtt      # ✅ 승인 완료 - 주 플랫폼
# ko_fi: bellkjtt     # 백업으로 유지 (선택)
```

**액션**:
1. ✅ GitHub Sponsors 승인 완료
2. ✅ 티어 설정 완료
3. ✅ Ko-fi는 백업으로 유지 또는 제거

---

## 🎯 지금 바로 할 것

### 즉시 실행 (5분):

```bash
# 1. Ko-fi 가입
https://ko-fi.com → Sign Up → Username: bellkjtt

# 2. FUNDING.yml 업데이트
# 파일 편집 후:
git add .github/FUNDING.yml
git commit -m "Add Ko-fi for immediate sponsorship"
git push

# 3. 확인
https://github.com/bellkjtt/vTTS 
→ "Sponsor this project" 섹션 확인!
```

### 병렬 실행 (10분):

```
☐ GitHub Sponsors 신청
  → https://github.com/sponsors
  → 정보 입력 (10분)
  → 승인 대기 (1-3일)

☐ Ko-fi 설정
  → https://ko-fi.com
  → 가입 (1분)
  → FUNDING.yml 업데이트
  → 즉시 활성화!
```

---

## ✅ 체크리스트

### GitHub Sponsors 신청
- [ ] https://github.com/sponsors 접속
- [ ] 개인 정보 입력
- [ ] W-8BEN 작성
- [ ] Stripe 계정 연결
- [ ] 은행 계좌 등록
- [ ] 신청 제출
- [ ] 승인 이메일 대기

### Ko-fi 설정 (대안)
- [ ] https://ko-fi.com 가입
- [ ] Username 설정
- [ ] `.github/FUNDING.yml` 업데이트
- [ ] Git push
- [ ] GitHub에서 확인

---

## 🆘 문제 해결

### Q: GitHub Sponsors 승인이 늦어지면?
**A**: Ko-fi나 Buy Me a Coffee를 사용하세요. 승인 후 변경 가능합니다.

### Q: 한국 은행 계좌 연결 가능한가?
**A**: 네, Stripe를 통해 한국 은행 계좌 연결 가능합니다.

### Q: 수수료는 얼마인가?
**A**: 
- GitHub Sponsors: 0% (GitHub 무료)
- Ko-fi: 0% (PayPal/Stripe 수수료만)
- Patreon: 5-12%

### Q: 여러 플랫폼 동시 사용 가능?
**A**: 네, FUNDING.yml에 여러 개 추가 가능합니다.

---

## 📞 도움

문제가 있으면:
- GitHub Sponsors 지원: https://support.github.com
- Ko-fi 지원: https://help.ko-fi.com
