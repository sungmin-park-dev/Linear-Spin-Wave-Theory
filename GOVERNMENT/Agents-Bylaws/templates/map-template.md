---
frontmatter-version: 1
template-version: 1
title: Template — map-*.md
section: templates
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-30
must-read: GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md
---

# Template — map-*.md

`map-*.md`는 에이전트가 폴더를 탐색할 때 읽는 네비게이션 파일이며, navigation, source routing, 파일 역할 경계만 담당한다. **100줄 이내**를 권장한다.
목차 표의 파일 항목은 Obsidian에서도 열 수 있도록 wikilink alias 형식으로 작성한다.

---

## 구조 (복사해서 사용)

```markdown
---
frontmatter-version: 1
template-version: 1
title: Map — [폴더명]
section: [layer 내 경로. layer prefix 제외. 예: issue-notes, templates]
status: in-review
last-edited-by: agent-id
created: YYYY-MM-DD
updated: YYYY-MM-DD
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---

# Map — [폴더명]

- [이 폴더가 무엇인지 — 1줄]
- [주요 역할 또는 구성 기준 — 1줄]
- [에이전트가 알아야 할 핵심 맥락 — 필요 시 1줄 추가]

## 목차 <!-- 기본형 | 택일: 기본형 / 라이프사이클형 -->
<!--
- 정적 디렉토리. 직접 자식을 단순 나열.
- 현재 폴더의 직접 하위 문서·폴더만 넣는다.
- 하위 폴더 내부 문서는 해당 하위 폴더의 README 또는 map에서 관리한다.
- 라이프사이클형(open/closed 등 상태 이동)인 경우 아래 기본형 대신 라이프사이클형 사용.
-->

| 항목 | 역할 |
|---|---|
| [[path/to/file\|file]] | 설명 |
| `subfolder/` | 설명 |

---

## 목차 <!-- 라이프사이클형 | 택일: 기본형 / 라이프사이클형 -->
<!--
- open/closed 등의 상태로 파일이 이동하는 워크플로우 디렉토리에 사용.
- `## 목차`에 상태별 서브섹션을 작성하고, 컬럼을 유형·상태·비고 등으로 확장한다.
-->

### `open/` — [활성 항목 설명]

| 항목 | [유형/상태 등] | 역할 | [상태] |
|---|---|---|---|
| [[path/to/open-file\|open-file]] | [유형/상태 등] | 설명 | [상태] |

### `closed/` — [종결 항목 설명]

| 항목 | [필드] | [결과] |
|---|---|---|
| [[path/to/closed-file\|closed-file]] | [필드] | [결과] |

---

### Remarks

[선택. 하위 폴더 내용, source-of-truth routing, 주의사항 등. 필요 없으면 섹션 전체 생략.]

## 에이전트 지침

- [이 폴더에서 지킬 규칙 — 상위 AGENTS.md와 중복 금지]
- [새 파일 생성 전 `GOVERNMENT/Agents-Bylaws/templates/[해당]-template.md` 읽을 것.]

## 참고 문서
<!--
- 이 map 또는 아래 문서를 수정할 때 함께 확인·갱신해야 하는 문서.
- 참고 문서에도 이 map으로 돌아오는 링크를 둔다.
-->

- [[GOVERNMENT/Agents-Bylaws/templates/map-template|map-template]] — map 작성 기준
```

---

## frontmatter 필드 안내

| 필드 | 값 규칙 |
|---|---|
| `frontmatter-version` | frontmatter **스키마** 버전. 전역 공통값(현재 `1`). 이 template에서 임의로 바꾸지 않는다 |
| `template-version` | 이 map이 따르는 map-template **내용** 버전. 새 map은 생성 시점의 template-version을 그대로 적는다. §`작성 규칙` 참조 |
| `status` | 에이전트가 생성한 직후 `in-review`. 사용자 검토 후 `accepted`로 전환 |
| `section` | layer prefix 없이 layer 내 경로만. 예: `issue-notes` (❌ `GOVERNMENT/Working-Pad/issue-notes`) |
| `must-read` | 이 폴더 파일을 편집할 때 선행 필독 문서가 있는 경우만 기재. 전역 AGENTS.md는 제외 |

전체 frontmatter 정책: `GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md`

---

## 작성 규칙

- **인트로 bullet**: 폴더의 정의·역할·배경 맥락. 3줄 이내.
- **map 범위**: map은 navigation, source routing, 파일 역할 경계를 담당한다. 상세 계획, 조사 계획, batch plan, stop rule, 상세 운영 원칙은 별도 named document나 policy/template에 두고 map에는 링크만 둔다.
- **목차**: 각 항목의 핵심 역할을 담는다. 구조만으로 파악하기 어려운 추가 맥락은 `### Remarks`에 명시.
- **목차 표 범위**: 현재 폴더의 직접 하위 문서·폴더만 넣는다. 하위 폴더 내부 문서는 해당 하위 폴더의 README 또는 map에서 관리한다.
- **파일 항목 표기**: 파일은 `[[path/to/file\|file]]`처럼 Obsidian wikilink alias로 쓴다. Markdown table 안에서는 alias 구분자 `|`를 `\|`로 escape한다.
- **폴더 항목 표기**: 폴더는 wikilink가 아니라 `subfolder/`처럼 inline code 경로로 쓴다.
- **Remarks**: 필요한 경우만 추가. 하위 폴더 내용, source-of-truth routing, 주의사항 등을 적고 없으면 섹션 전체를 생략.
- **에이전트 지침**: 이 폴더에만 해당하는 규칙만. 상위 AGENTS.md 반복 금지.
- **참고 문서**: 이 template 변경 시 repo 안의 모든 `map-*.md`를 하나씩 확인한다. 고정 목록은 두지 않고, 현재 대상은 `rg --files | rg '(^|/)map-[^/]+\.md$'`로 조회한다.
- **legacy heading 금지**: 새 map에 `Read First`, `Source Of Truth`, `Belongs Here`, `Does Not Belong Here`, `Key Documents`, `Folder Roles`, `Agent Notes`, `Rules` 같은 확장형 heading을 추가하지 않는다.
- **100줄 초과 시**: 별도 문서 도입 검토 후 map에서 링크로 참조.
- **에이전트 자율 업데이트 주의**: 에이전트가 단독으로 내용을 갱신한 경우 사용자 검토 권장.

## template-version 관리

`template-version`은 frontmatter **스키마** 버전인 `frontmatter-version`과 별개로, 이 **map-template의 내용(구조·규칙) 버전**을 추적한다.

- **이 파일(map-template.md)의 `template-version`**: 현재 이 template의 기준 버전이다.
- **각 `map-*.md`의 `template-version`**: 그 map이 마지막으로 reconcile된 template 버전. 자신의 `must-read` template(= 이 문서) 버전과 짝을 이룬다.
- **drift**: map의 `template-version`이 이 파일의 값보다 낮거나 없으면 reconcile 대상이다.

### 버전을 올리는 기준

- map의 **구조·규칙·필드**가 바뀌어 기존 map들이 reconcile되어야 하는 변경에서만 올린다.
- 오탈자·표현 다듬기 등 기존 map에 영향이 없는 변경은 올리지 않는다.
- 버전을 올리면 이 파일 frontmatter의 `template-version`과 위 `## 구조` 복사블록의 `template-version`을 함께 올린다.

> 현재는 `map-*.md`에만 적용한다. 다른 template으로의 확장과 checker 자동 drift 검출은 후속 작업이다.

## 업데이트 절차

이 template이 업데이트되면 repo 안의 모든 `map-*.md`를 하나씩 확인한다. 각 map 문서에 template 변경 사항을 적용하고 `template-version`을 현재 값으로 맞춘다. 적용하지 않는다면 예외로 둘 근거가 있는지 확인한다.

대상 목록은 이 문서에 고정하지 않는다. 확인 시점의 실제 파일 목록을 아래 명령으로 조회한다.

```bash
rg --files | rg '(^|/)map-[^/]+\.md$'
```

## 참고 문서

- [[GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy|frontmatter-policy]] — 운영 문서 frontmatter 기준
