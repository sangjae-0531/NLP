# 인간지표 팩터 (The "Human Indicator" Factor)

경제 유튜버 영상 자막에서 투자 발언을 추출해 유튜버별 "인간지표" 점수를 매기고, 그 점수로 만든 롱숏 포트폴리오가 시장 노출을 통제한 뒤에도 알파를 내는지 검정하는 프로젝트입니다.

금융공학고급논제(자연어) 파이널 프로젝트 · Team 5 (이상재, 정선호)

## 연구 질문

1. **발언 추출 (NLP):** 한국어 자동 자막에서 투자 발언(대상, 방향, 기간)을 자동으로 추출할 수 있는가? 수작업 라벨과 비교해 정확도를 잰다.
2. **실력 vs 역실력:** 유튜버의 방향 예측 실력은 시간이 지나도 지속되는가? 꾸준히 틀리는 "역지표" 유튜버가 있는가?
3. **롱숏 알파:** 상위 분위(Q1)는 따라 하고 하위 분위(Q5)는 반대로 하는 포트폴리오가 시장 베타를 통제한 뒤에도 유의한 알파를 내는가?

지속성이 없다는 결과도 "유튜버 실력은 재현되지 않는다"는 결론으로 의미가 있습니다.

## 파이프라인

```
수집 → 발언 추출 → ETF 매핑 → 점수·분위 → 포트폴리오 → 알파 검정
```

### 1. 데이터 수집
- **채널:** 경제 유튜버 구독자 상위 50개 (사전 테스트는 3~10개)
- **메타데이터:** video_id, 채널, 제목, 업로드 시각(KST, 분 단위), 조회수 — `yt-dlp`
- **자막:** 한국어 자동 자막 — `yt-dlp`, `youtube-transcript-api`, 품질이 나쁘면 `mlx-whisper`로 보완
- **가격:** 국내 상장 ETF 수정종가 — `pykrx`, `FinanceDataReader`
- **기간:** 점수 형성 6~8개월 + 검정 12개월 이상 (사전 테스트 후 확정)

### 2. 발언 추출 (NLP 핵심)
1. 자막을 2~3분 청크로 나누고 regex로 숫자·지수명·종목명 정규화
2. 방향어(오른다, 빠진다, 매수 등)가 없는 청크는 건너뜀
3. LLM으로 JSON 추출: `video_id, speaker, t_start, quote, target, target_level, direction, horizon, conviction, conditional`
4. 사전 기반 규칙과 Naive Bayes 베이스라인과 비교
5. 두 명이 200개 청크를 각자 라벨링해 Cohen's kappa로 일치도를 보고, precision·recall을 보고

LLM은 대상과 방향만 추출하고 맞았는지는 판단하지 않습니다. 업로드 이후 시장 정보는 프롬프트에 넣지 않습니다 (look-ahead bias 방지).

### 3. 포지션 변환과 ETF 매핑
발언 하나는 (ETF, 부호, 진입일, 청산일) 포지션 하나가 됩니다.

| 발언 대상 | 매핑 ETF |
|---|---|
| 코스피 | KODEX 200 |
| 코스닥 | KODEX 코스닥150 |
| S&P500 | TIGER 미국S&P500 |
| 나스닥 | TIGER 미국나스닥100 |
| 반도체 | KODEX 반도체 |
| 금 / 유가 | KODEX 골드선물(H) / KODEX WTI원유선물(H) |
| 개별 종목 | 해당 섹터 ETF (확장 분석에서 종목 자체) |

- **보유기간:** 단기·언급 없음 1거래일, 분기·하반기 20거래일, 중장기 60거래일
- **진입:** 업로드 후 첫 종가 (15:30 전 업로드면 당일, 이후면 다음 거래일)
- **숏:** 수익률 부호를 반대로 함 (인버스 ETF는 쓰지 않음)

### 4. 인간지표 점수와 5분위
상승장 편향을 없애기 위해 같은 방향을 찍었을 때의 기대 적중률을 뺀 **초과 적중률**에, 발언 수가 적을수록 0으로 당기는 **shrinkage**를 적용합니다.

$$\tilde{S}_{i,t} = \frac{n_{i,t}}{n_{i,t}+k} \cdot \frac{1}{n_{i,t}} \sum_j (\text{Hit}_j - p_j), \quad k = 10$$

- 매월 말 t에 **채점이 끝난 발언만** 써서 점수를 매김 (point-in-time)
- 채점 10건 이상인 유튜버만 5분위로 분류
- 지속성 검정: Spearman 순위 상관, 분위 전이행렬

### 5. 포트폴리오와 알파 검정
- 캘린더 타임 롱숏 $R_{LS,t} = R_{P(1),t} - R_{P(5),t}$, 포지션별 1/σ 가중
- 3팩터 회귀: $R_{LS,t} = \alpha + \beta_1 R_{KOSPI} + \beta_2 R_{SPX} + \beta_3 R_{Gold} + \varepsilon_t$
- Treynor-Mazuy 타이밍 분해 ($\gamma > 0$이면 타이밍 실력)
- Newey-West t값 (lag 5), 순열 검정 1,000회, 분위 단조성, 역인과 통제
- 거래비용 왕복 0.1%·0.2% 반영

## 강건성 분석

3분위 · 12개월 롤링 형성 창 · k = 5, 20 · 고정 분할 · 사전 기반 추출 · 개별 종목 직접 매매 · 발언 단위 A안/B안

## 일정

| 기간 | 작업 |
|---|---|
| 10/24–10/28 | 사전 테스트 |
| 10/27–10/29 | 프로젝트 피치 |
| 10/30–11/15 | 수집·추출·검증 |
| 11/20–12/6 | 점수·포트폴리오·회귀 |
| 12/7–12/13 | 강건성·발표 자료 |
| 12/15–12/17 | 최종 발표 |

## 참고 문헌

- Antweiler & Frank (2004), *Journal of Finance*
- Tetlock (2007), *Journal of Finance*
- Engelberg, Sasseville & Williams (2012), *Management Science*
- Chen, Hwang & Peng (2025), *Review of Financial Studies*
- Kakhbod et al. (2024), AFA/AEA Working Paper
- Kirschbaum et al. (2026), *Finance Research Letters*
- Lopez-Lira & Tang (2026), Levy (2026) — LLM과 look-ahead bias
