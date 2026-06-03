---
frontmatter-version: 1
title: Template — map-*.md
section: templates
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
must-read: GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md
---

# Template — map-*.md

`map-*.md`는 에이전트가 폴더를 탐색할 때 읽는 네비게이션 파일이다. **100줄 이내**를 권장한다.

---

## 구조 (복사해서 사용)

```markdown
---
frontmatter-version: 1
title: Map — [폴더명]
section: [GOVERNMENT 내부 상대 경로. 예: issue-notes/open, templates]
status: in-review
last-edited-by: claude
created: YYYY-MM-DD
updated: YYYY-MM-DD
# must-read: [이 폴더 파일 편집 시 반드시 먼저 읽어야 할 template 또는 핵심 지침 경로]
---

# Map — [폴더명]

- [이 폴더가 무엇인지 — 1줄]
- [주요 역할 또는 구성 기준 — 1줄]
- [에이전트가 알아야 할 핵심 맥락 — 필요 시 1줄 추가]

## 목차

[**기본형** — 정적 디렉토리. 직접 자식을 단순 나열.]
| 항목 | 역할 |
|---|---|
| `file.md` | 설명 |
| `subfolder/` | 설명 |

[**라이프사이클 확장형** — open/closed 등의 상태로 파일이 이동하는 워크플로우 디렉토리에 사용.
`## 목차` 대신 상태별 서브섹션으로 대체하고, 컬럼을 유형·상태·비고 등으로 확장한다.]

### `open/` — [활성 항목 설명]

| 항목 | [유형/상태 등] | 역할 | [상태] |
|---|---|---|---|

### `closed/` — [종결 항목 설명]

| 항목 | [필드] | [결과] |
|---|---|---|

### Remarks

[선택. child 폴더 내용 언급, 주의사항, 참조 문서 등. 필요 없으면 섹션 전체 생략.]

## 에이전트 지침

- [이 폴더에서 지킬 규칙 — 상위 AGENTS.md와 중복 금지]
- [새 파일 생성 전 `GOVERNMENT/Agents-Bylaws/templates/[해당]-template.md` 읽을 것.]
```

---

## frontmatter 필드 안내

| 필드 | 값 규칙 |
|---|---|
| `status` | 에이전트가 생성한 직후 `in-review`. 사용자 검토 후 `accepted`로 전환 |
| `section` | `GOVERNMENT/` prefix 없이 내부 경로만. 예: `issue-notes/open` (❌ `GOVERNMENT/Working-Pad/issue-notes/open`) |
| `must-read` | 이 폴더 파일을 편집할 때 선행 필독 문서가 있는 경우만 기재. 전역 AGENTS.md는 제외 |

전체 frontmatter 정책: `GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md`

---

## 작성 규칙

- **인트로 bullet**: 폴더의 정의·역할·배경 맥락. 3줄 이내.
- **목차**: 각 항목의 핵심 역할을 담는다. 구조만으로 파악하기 어려운 추가 맥락은 `### Remarks`에 명시.
- **Remarks**: 필요한 경우만 추가. 없으면 섹션 전체 생략.
- **에이전트 지침**: 이 폴더에만 해당하는 규칙만. 상위 AGENTS.md 반복 금지.
- **100줄 초과 시**: 별도 문서 도입 검토 후 map에서 링크로 참조.
- **에이전트 자율 업데이트 주의**: 에이전트가 단독으로 내용을 갱신한 경우 사용자 검토 권장. LLM이 생성한 context 파일은 성능 저하를 유발할 수 있다 (ETH Zurich, 2026).
