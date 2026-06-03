---
frontmatter-version: 1
title: Template — issue-note
section: templates
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Template — issue-note

`GOVERNMENT/Working-Pad/issue-notes/` 하위 파일에 사용하는 템플릿.

---

## 디렉토리 구조

```
GOVERNMENT/Working-Pad/issue-notes/
  map-issue-notes.md
  open/      ← 활성 이슈 (status: draft | in-review)
  closed/    ← 종결 이슈 (status: closed)
```

---

## 유형 (`issue-type`)

| 유형 | 언제 사용하는가 |
|---|---|
| `discussion` | 방향이 열려 있는 주제를 분석·논의. 결론이 없고 검토가 진행 중인 경우. |
| `idea` | 아직 개발되지 않은 아이디어 메모. 논의 시작 전, 가능성 탐색 단계. |
| `review` | 기존 문서·시스템·결정을 재검토. 현황을 진단하고 개선 방향을 도출. |
| `problem` | 발생한 문제를 추적·진단. 기술 결함부터 프로세스 문제까지. |
| `research` | 의사결정 전 정보·사례 수집. 외부 리서치 또는 내부 탐색. |

> **issue-notes vs proposals**: 방향이 열려 있는 논의·분석 → `issue-notes/open/`.
> 구체적 변경안이 있고 승인이 필요한 경우 → `GOVERNMENT/Working-Pad/vault-staging/`.

---

## 파일명 규칙

```
YYMMDD-[brief-kebab-topic].md
```

- `YYMMDD`: 이슈 생성일 (예: `260521`)
- `[brief-kebab-topic]`: 주제를 나타내는 짧은 kebab-case 문자열 (3–5 단어)
- type은 파일명에 포함하지 않음 — frontmatter `issue-type` 으로 구분

예시: `260521-human-wiki-sync.md`, `260521-frontmatter-checker-externalization.md`

---

## 공통 frontmatter

```yaml
frontmatter-version: 1
title: [이슈 제목]
section: issue-notes/open           # 종결 시 issue-notes/closed 로 변경
issue-type: discussion              # discussion | idea | review | problem | research
status: draft
last-edited-by: claude
created: YYYY-MM-DD
updated: YYYY-MM-DD
# 선택:
# related: [관련 파일 경로 — 레포지토리 루트 기준]
# resolution: resolved | rejected | abandoned | superseded   ← closed 시
# outcome: [결과 문서 경로]                                   ← closed 시
```

---

## 공통 본문 구조

> **작성 원칙**: 배경과 문제 정의는 나중에 읽어도 당시 맥락을 재구성할 수 있어야 한다.
> 독자는 사용자와 에이전트 모두이며, 문서가 닫힌 후에도 참조될 수 있다.

```markdown
## 배경

[이 문서가 왜 작성됐는가. 당시 상황, 동기, 계기.
이 섹션만 읽어도 "왜 이 이슈가 존재하는지" 이해할 수 있어야 한다.]

## 문제 정의

[해결하려는 문제 또는 답해야 할 질문을 1–3문장으로.
"~가 문제다" 또는 "~를 결정해야 한다"처럼 명제 형태로 쓰면 명확해진다.]

## 본론

[유형에 따라 내용이 달라진다. 아래 유형별 가이드 참조.]

## 결론 / 미결 사항

[결론이 있으면: 결정 사항과 다음 액션.
미결이면: 진전하기 위해 해소해야 할 것.]

## 참조

[관련 파일과 문서. 경로는 레포지토리 루트 기준 상대 경로로 통일.]
- `GOVERNMENT/...` — [운영 문서]
- `research-space/...` — [이론·연구 문서]
- `code-space/...` — [코드]
```

---

## 유형별 가이드

### `discussion` — 논의

본론 구조는 자유. 아래를 참고해 필요한 항목만 사용한다.

```markdown
## 본론

[검토한 옵션, 트레이드오프, 고려 사항을 자유롭게 서술.
여러 옵션을 비교할 경우 소제목을 사용해도 좋다.]

### 옵션 A — [이름]
...

### 옵션 B — [이름]
...

### 트레이드오프
...
```

---

### `idea` — 아이디어

본론 구조는 자유. 아래를 참고해 필요한 항목만 사용한다.

```markdown
## 본론

### 핵심 아이디어
[무엇을 하고자 하는가. 간결하게.]

### 가능성과 한계
[왜 될 것 같은가, 어떤 제약이 있는가.]

### 열린 질문
[발전시키려면 먼저 확인해야 할 것들.]
```

---

### `review` — 리뷰

본론 서브섹션 권장. 리뷰 흐름(대상→발견→권고)이 일관되게 유지되어야 한다.

```markdown
## 본론

### 리뷰 대상 현황
[무엇을 리뷰했는가. 현재 상태 스냅샷.]

### 발견 사항
[리뷰를 통해 확인된 문제, 갭, 개선 가능 영역.]

### 권고
[변경하거나 다음에 할 것들. 우선순위 포함.]
```

---

### `problem` — 문제

본론 서브섹션 **강권장**. 증상과 원인을 분리하지 않으면 진단이 흐트러진다.

```markdown
## 본론

### 증상
[무엇이 잘못되고 있는가. 관찰 가능한 현상.]

### 재현 조건
[어떤 상황에서 발생하는가. 재현에 필요한 전제 조건.]

### 근본 원인 분석
[왜 발생했는가. 표면 원인 뒤의 구조적 원인.]

### 해결 방안
[가능한 해결책 후보. 각 방안의 트레이드오프.]
```

---

### `research` — 탐색

본론 구조는 자유. 아래를 참고해 필요한 항목만 사용한다.

```markdown
## 본론

### 탐색 범위
[무엇을 조사했는가. 범위와 출처.]

### 발견 사항
[주요 발견, 사례, 데이터.]

### 시사점
[이 탐색 결과가 현재 프로젝트에 주는 의미.]
```

---

## 생성 후 처리

1. 파일 위치: `GOVERNMENT/Working-Pad/issue-notes/open/`
2. `status: draft`, 날짜 frontmatter 기입.
3. `map-issue-notes.md` "열린 이슈" 목차에 한 줄 추가.

## idea → discussion 전환

1. `open/` 에 새 discussion 파일 생성.
2. idea 파일 frontmatter: `outcome: GOVERNMENT/Working-Pad/issue-notes/open/[새-파일명].md` 추가.
3. idea 파일: `status: closed`, `resolution: superseded`, `section: issue-notes/closed`.
4. idea 파일을 `closed/` 로 이동.
5. `map-issue-notes.md`: idea 항목을 종결 이슈 섹션으로 이동, discussion 항목 신규 등록.

## 종결 처리

- 결론 도출 시: 결정 내용 → `GOVERNMENT/Court-Precedents/`, 재사용 절차·분석 → `GOVERNMENT/Agents-Bylaws/`.
- 파일 frontmatter: `status: closed`, `resolution` 기입, `section: issue-notes/closed`.
- 파일을 `open/` → `closed/` 로 이동.
- `map-issue-notes.md`: 항목을 종결 이슈 섹션으로 이동.
