# 보존본

더 고치지 않는 것을 둔다. 이 안의 문서에 적힌 경로는 쓸 때의 위치 그대로다. 지금 위치는 아래 표로 찾는다.

| 폴더 | 내용 |
|---|---|
| [`2026-10-08-lswt-markdown/`](2026-10-08-lswt-markdown/README.md) | LSWT 일반 이론 Markdown, 2026-10-08 상태 (옛 `docs/lswt`). 지금 정본은 연구노트 `workbench/topics/*/lswt-*` |
| [`2026-10-08-development-beamer/`](2026-10-08-development-beamer/README.md) | 패키지 개발 설계와 결정 기록 Beamer, 2026-10-08 상태 (옛 `docs/development`). 지금 정본은 `dev-*` 연구노트 |
| [`2026-10-11-lswt-latex-notes/`](2026-10-11-lswt-latex-notes/) | Markdown 전환 직전의 LaTeX 이론 연구노트와 변환 도구 |
| [`legacy/`](legacy/) | 원본 legacy 코드(`modules/`, `scripts/`)와 과거 연구노트. 테스트 몇 개가 legacy 수치와 대조한다 |
| [`government/`](government/) | 옛 GOVERNMENT의 닫힌 이슈·인계 기록, 형식 규칙(frontmatter·이름·폴더 안내)과 서식, map 문서, User-Constitution(원칙은 연구 공통 규칙 6·7번으로 옮김) |

## 옛 경로 → 새 경로 (2026-10-13)

연구 저장소 폴더 표준(research-workspace 결정 "연구 저장소 폴더", 2026-10-13)에 맞춰 옮겼다. 살아 있는 파일(코드, 테스트, README·AGENTS, 노트, 규칙, 열린 할 일)의 경로는 함께 고쳤고, 기록(일지, 코멘트, 결정, 닫힌 기록, 이 폴더 안, `data-space/`)은 그대로 두었다.

| 옛 경로 | 새 경로 |
|---|---|
| `examples/` | `code-space/examples/` |
| `model/` | `code-space/model/` |
| `docs/sources/00-primary-source/`, `01-editable-notes/`, `README.md` | `sources/` 아래 같은 이름 |
| `docs/sources/03-verification-notes/` | `docs/verification-notes/` |
| `docs/archive/<이름>/` | `archive/<이름>/` |
| `legacy/` | `archive/legacy/` |
| `workbench/notes/<노트>/` | `workbench/topics/<주제>/<노트>/` (주제는 아래 표, `note.yaml` `topics:`의 첫째) |
| `workbench/research.yaml`의 주제 정보 | `workbench/topics/<주제>/topic.yaml` (순서는 `research.yaml` `topics:`) |
| `GOVERNMENT/Court-Precedents/<날짜-이름>.md` | `workbench/decisions/<날짜-이름>.md` |
| `GOVERNMENT/Court-Precedents/map-decisions.md` | `archive/government/Court-Precedents/map-decisions.md` |
| `GOVERNMENT/Agents-Bylaws/policies/lswt-writing-style.md` | `workbench/rules/lswt-writing-style.md` |
| `GOVERNMENT/Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md` | `workbench/rules/lswt-canonical-document-lifecycle.md` |
| `GOVERNMENT/Agents-Bylaws/` (그 밖) | `archive/government/Agents-Bylaws/` |
| `GOVERNMENT/Working-Pad/issue-notes/closed/`, `handoff/closed/`, `inbox/`, `vault-staging/` | `archive/government/Working-Pad/` 아래 같은 이름 |
| `GOVERNMENT/User-Constitution/` (10/13 공통 규칙 뒤) | `archive/government/User-Constitution/`. 단일 정본 원칙은 개인 저장소 `rules/research.md` 6·7번 |

남은 `GOVERNMENT/`(TASK-QUEUE, 열린 이슈·인계, 제안, project-review, roadmap)는 손댈 때 하나씩 `workbench/`로 옮긴다.

| 주제 | 노트 |
|---|---|
| `fundamental-theory-of-spin-wave-theory` | `lswt-boson-hamiltonian`, `lswt-classical-order`, `lswt-paraunitary-diagonalization`, `lswt-spin-hamiltonian` |
| `spin-wave-observables` | `lswt-magnon-thermodynamics`, `lswt-magnon-topology`, `lswt-spin-correlations` |
| `beyond-linear-spin-waves` | `lswt-nonlinear-spin-waves` |
| `worked-examples` | `lswt-example-honeycomb-topological-ferromagnet`, `lswt-example-square-antiferromagnet`, `lswt-example-square-ferromagnet`, `lswt-example-triangular-antiferromagnet` |
| `toolkit-development` | `dev-decisions`, `dev-design`, `dev-verification` |
