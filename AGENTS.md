# AGENTS.md

> **먼저 읽을 것:** 모든 연구 저장소에 통하는 [연구 공통 규칙](https://github.com/sungmin-park-dev/research-workspace-backup/blob/main/rules/research.md)(맥 사본 `~/.config/research-workspace/personal/rules/research.md`). 이 파일은 그 위에 이 저장소만의 규칙을 더한다. 둘이 부딪히면 공통 규칙을 따르고 사용자에게 알린다.

> LSWT 이론 문서 정립과 Python 패키지 개발 시 에이전트(Claude·Codex)가 따르는 저장소 지침. `CLAUDE.md`는 이 파일을 불러온다(`@AGENTS.md`).

---

## 프로젝트 개요

2D 스핀 모델의 Linear Spin Wave Theory 계산을 위한 Python 라이브러리.
공개 배포 목표. NBCP(Na₂BaCo(PO₄)₂) 모형(arXiv:2505.06398; npj Quantum Materials 2022, DOI 10.1038/s41535-022-00500-3)을 주요 검증 사례로 쓴다. NBCP 연구 자체는 별도 저장소 `nbcp-spin-supersolid`에 있다.

**핵심 기능**: 격자 정의 → LSWT 대각화 → 물리량 계산 (열역학, 위상, 상관함수)

**설계 원칙**:
- 명확성 우선 — 물리적 의미가 코드에 드러나야 함
- 관심사 분리 — 시스템 정의 / 솔버 / 관측량(observables) / 시각화
- 범위는 반고전 방법: 고전 질서와 그 위의 요동(LSWT 중심, 1/S 보정, 고전 MC·스핀 동역학). 양자 다체 계산(DMRG·QMC·TN)은 다루지 않고, ED는 스핀파 결과 검증용으로만 둔다 ([2026-10-10 결정](workbench/decisions/2026-10-10-package-scope-semiclassical.md))
- `SpinSystem`은 solver-agnostic — LSWT 전용 로직을 넣지 않음
- Python config 사용 (YAML 아님) — exchange matrix(3×3)를 numpy로 직접 정의

---

## 현재 디렉토리 구조

> LSWT 일반 이론과 개발 설계는 `workbench/topics/` 연구노트(2026-10-08부터; `archive/2026-10-08-lswt-markdown/`·`archive/2026-10-08-development-beamer/`는 보존본), 실행 예제와 검증 스크립트는 `code-space/examples/`에서 관리한다.
> 설계 사본은 `archive/government/Working-Pad/idea-proposals/2026-05-30-project-knowledge-philosophy.md`에 둔다.

```
project-root/
├── AGENTS.md · CLAUDE.md · README.md · pyproject.toml
│
├── workbench/                   # research-workspace 앱이 읽고 쓰는 곳 (형식: 앱의 docs/repo-format.md)
│   ├── research.yaml            # 프로젝트 정보와 주제 순서
│   ├── topics/<주제>/           # topic.yaml(주제 정보)과 그 주제가 주인인 연구노트 폴더 (2026-10-08부터 내용 원본)
│   ├── decisions/               # 결정 기록 (옛 GOVERNMENT/Court-Precedents)
│   ├── rules/                   # 이 프로젝트의 긴 규칙: 이론 문체, 정본 문서 절차
│   ├── tasks/                   # 작업 파일, 일 하나에 파일 하나 (앱 작업 탭; 옛 TASK-QUEUE·열린 이슈·인계)
│   ├── comments/ · log/ · figures/
│   └── STATUS.md                # 앱이 쓰는 요약 (git 제외)
│
├── code-space/                  # 코드
│   ├── spintoolkit/             # system · states · methods · definitions · observables · visualization
│   ├── lswt/                    # 옛 이름 `lswt` 호환 패키지(DeprecationWarning)
│   ├── tests/                   # pytest 테스트
│   ├── examples/                # quickstart.py, gallery.py, tutorials/(튜토리얼 스크립트), worked_examples/(예제 노트의 계산), verification/(검증 스크립트)
│   └── model/nbcp/              # NBCP 모형 정의 (패키지 회귀 테스트용 고정본)
│
├── docs/                        # 우리가 쓴 글
│   ├── README.md
│   ├── reviews/                 # 검토·계획 문서 (옛 project-review, 2026-10-01 이론 검토 안내)
│   ├── tutorials/               # 패키지 사용법 영어 튜토리얼 5편
│   └── verification-notes/      # 수식·basis convention과 코드의 내부 검증 자료
│
├── sources/                     # LSWT 원자료: 원본 PDF와 TeX 전사본. 고치지 않는다
├── data-space/                  # 검토하고 정돈한 결과 데이터
│
├── archive/                     # 더 고치지 않는 보존본. 옛 경로 → 새 경로 표는 archive/README.md
│   ├── 2026-10-08-lswt-markdown/      # 옛 docs/lswt
│   ├── 2026-10-08-development-beamer/ # 옛 docs/development
│   ├── 2026-10-11-lswt-latex-notes/   # Markdown 전환 직전 LaTeX 이론 노트와 변환 도구
│   ├── legacy/                  # 원본 legacy 코드와 과거 연구노트
│   └── government/              # 옛 GOVERNMENT: 닫힌 기록, 형식 규칙과 서식, map 문서, User-Constitution, TASK-QUEUE·roadmap·제안(2026-10-15)
```

---

## 이론 문서와 source authority

LSWT 이론 내용은 [2026-10-08 결정](workbench/decisions/2026-10-08-lswt-development-research-notes.md)에 따라 `workbench/topics/`의 영어 연구노트에 정리한다. 2026-10-11부터 이론 연구노트의 본문 형식은 Markdown(`note.md`)이고, 옛 LaTeX 원본은 `archive/2026-10-11-lswt-latex-notes/`에 보존한다([결정](workbench/decisions/2026-10-11-lswt-notes-markdown.md)). 2026-10-09부터 주제별 7개(`lswt-spin-hamiltonian`, `lswt-classical-order`, `lswt-boson-hamiltonian`, `lswt-paraunitary-diagonalization`, `lswt-magnon-thermodynamics`, `lswt-spin-correlations`, `lswt-magnon-topology`)다(처음 3개 `lswt-foundations`·`lswt-derivation`·`lswt-observables`를 나눔). 2026-10-12에 1/S 보정 노트 `lswt-nonlinear-spin-waves`를 더해 8개가 됐고, 같은 날 예제 노트 `lswt-example-*` 4개를 더했다. 예제 노트마다 `code-space/examples/worked_examples/`에 계산 스크립트가 있고, `code-space/tests/test_worked_examples.py`가 노트에 적은 숫자를 확인한다. 앱의 주제(`workbench/topics/<주제>/topic.yaml`, 순서는 `workbench/research.yaml` `topics:`)는 기초 이론 · 관측량 · 선형 너머 · 예제 · 패키지 개발 다섯이다. `archive/2026-10-08-lswt-markdown/` Markdown은 2026-10-08 상태로 보존하고 더 고치지 않는다. 개발 설계와 결정 기록도 같은 결정에 따라 `dev-design`, `dev-verification`, `dev-decisions` 연구노트가 원본이며 `archive/2026-10-08-development-beamer/` Beamer는 보존본이다. 원본의 주장, 수식과 논리 연결을 추적할 수 있도록 보존하면서 문서를 개념별 owner로 나눈다. 재배치와 제외·보류 기록은 lifecycle 절차를 따르며, 물리적 의미나 수학적 타당성이 불명확한 부분을 임의로 고치지 않는다.

단일 정본 원칙은 [2026-08-01 결정](workbench/decisions/2026-08-01-lswt-markdown-source-authority.md), 현재 작성 경로와 자료 위치는 [2026-09-16 통합 기록](archive/government/Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md)을 따랐고, 정본의 위치와 작성 경로는 2026-10-08 결정이 연구노트로 바꿨다. 이는 사용자 승인 구조이며, 이론 내용의 acceptance를 변경하지 않는다. 전체 문서 진입점은 [docs/README.md](docs/README.md), LSWT 읽기 순서는 [archive/2026-10-08-lswt-markdown/README.md](archive/2026-10-08-lswt-markdown/README.md)다.

| 경로 | 역할 |
|---|---|
| `sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` | 원래 기록, 수식과 review annotation을 판정하는 primary evidence |
| `sources/01-editable-notes/note_lswt_reviewed.tex` | 원문 section 순서를 유지한 editable transcription |
| `sources/01-editable-notes/note_lswt_restructured.tex` | 재배치와 일부 편집이 포함된 structural reference |
| 논문 라이브러리 (research-library, 10/7 `02-reference-papers/`에서 옮김) | 유도와 convention을 확인하는 외부 참고문헌 (Toth–Lake: `tothLinear2015`) |
| `docs/verification-notes/` | 수식, basis convention과 코드의 내부 검증 자료 |
| 사용자 승인 `workbench/topics/*/lswt-*` 연구노트 | 현재 LSWT 이론 claim을 소유하는 유일한 theory canon (2026-10-08 전에는 `archive/2026-10-08-lswt-markdown/` Markdown) |

상세 source inventory와 원문 section-to-docs 대응은 `sources/README.md`에서 확인한다. `reviewed`와 `restructured` TeX를 새로운 단일 TeX master로 자동 병합하지 않으며, 유효한 내용은 primary PDF와 대조한 뒤 해당 LSWT 연구노트에서 통합한다. Source가 충돌하거나 부호, index, conjugation, normalization 또는 적용 조건이 불명확하면 `Unknown` 또는 open review item으로 남긴다. 자동 검사나 코드 수치 일치는 사용자의 Human Physics and Mathematics Review를 대체하지 않는다.

NBCP 연구는 2026-10-09 별도 저장소 `sungmin-park-dev/nbcp-spin-supersolid`로 옮겼다([결정](workbench/decisions/2026-10-09-nbcp-repository-split.md)). 연구노트·보조 노트·원고·계산 스크립트와 결과는 그 저장소가 소유한다. 이 저장소의 `code-space/model/nbcp/`와 `data-space/verification/260912-pseudo-goldstone/`는 패키지 회귀 테스트가 쓰는 고정본이다.

---

## 핵심 API

```python
from spintoolkit import SpinSystem, LSWTSolver   # 관례 별칭: import spintoolkit as stk

# 시스템 정의 — builder 패턴 (권장)
system = SpinSystem(lattice_vectors=[[1, 0], [0.5, np.sqrt(3)/2]])
system.add_site("A", [0, 0], spin=0.5, angles=[θ, φ], magnetic_field=[0, 0, h])
system.add_site("B", [0.5, 0.5], spin=0.5, angles=[θ, φ], magnetic_field=[0, 0, h])
system.add_coupling("A", "B", J_matrix, displacement=[1, 0])

# list 기반 생성도 지원 (하위 호환)
system = SpinSystem(sites=[...], couplings=[...], lattice_vectors=[[...], [...]])

# 접근
system.site("A").position       # label 또는 index로 접근
system.get_couplings("A", "B")  # 필터링된 coupling 리스트

# 솔버 실행 — bz_type은 solver에서 지정
solver = LSWTSolver(system, bz_type="Hex_60")
result = solver.solve(N=10)

# 결과 (SolverResult)
result.ground_state_energy   # float
result.eigenvalues           # np.ndarray (num_k, num_bands)
result.method                # str
result.data                  # dict (솔버별 고유 데이터)
```

**Legacy 호환**: `LSWTSolver`는 legacy dict 형식도 받음. `SpinSite`, `Coupling` alias 유지 (점진적 제거 예정).

**범위 (2026-10-10 결정)**: 같은 `SpinModel` 정의에서 고전 상태 탐색 → `solve_lswt` → `solve_nlswt`(1/S) 순서로 쌓는다. 새 방법은 이 반고전 흐름 안에서만 더하고, ED는 검증용으로 둔다.

---

## 코딩 컨벤션

- **Naming**: 모듈 `snake_case.py`, 클래스 `PascalCase`, 함수 `snake_case`, 상수 `UPPER_SNAKE_CASE`
- **Docstring**: NumPy 스타일
- **Import 순서**: 표준 라이브러리 → 서드파티 (`numpy`, `scipy`) → 로컬 (`spintoolkit.*`)
- **Type hints**: 사용 권장

---

## 작업 원칙

> 이 섹션은 에이전트(Claude·Codex)가 이론 문서와 코드 작업 시 반드시 따라야 하는 규칙이다.

1. **인터페이스 변경은 토의 후 결정**: API(함수명, 데이터 구조, 클래스 인터페이스) 변경은 선 제안 → 토의 → 성민 확정 순서를 따름. 임의로 결정하지 않음.
2. **영향 범위 명시**: 모듈 간 의존성 변경이 생기면 영향받는 모듈을 명시할 것.
3. **작업 종류에 맞는 검증**: 코드 구현은 구현 → legacy 수치 대비 검증 → 성민 확인 → 다음 단계 순서로 진행하며, 검증 전 다음 구현 단계에 착수하지 않는다. 이론 문서는 `workbench/rules/lswt-canonical-document-lifecycle.md`에 따라 문체 교정, 원문 대조와 사용자 물리·수학 검토를 구분한다. 문체 교정에 코드 수치 검증을 일괄 요구하지 않는다.
4. **이론 source authority 준수**: 위 `이론 문서와 source authority` 구분을 따른다. LSWT 일반 이론과 개발 노트는 `workbench/topics/` 연구노트를 직접 편집한다. 이론 노트 `lswt-*`는 `note.md`, 개발 노트 `dev-*`는 아직 TeX다. PDF는 그로부터 생성한다(Markdown 이론 노트의 공개용 PDF 경로는 열린 항목이다). LSWT 이론의 공개본은 사용자가 accepted로 승인한 연구노트에서만 만들고, 검토용 preview는 draft에서도 만든다. PDF 등 파생물을 직접 수정하지 않는다.
5. **이론 문체 preflight**: LSWT 이론 연구노트(`workbench/topics/*/lswt-*`)를 작성하거나 교정하기 전에 `workbench/rules/lswt-writing-style.md`를 읽는다. `status: in-review`이면 current working guidance로 적용하되 accepted policy로 보고하지 않으며, 문체 검토를 물리·수학적 acceptance로 간주하지 않는다.

현재 작업, 우선순위와 진행 상태는 `workbench/tasks/`의 작업 파일(앱 작업 탭, `STATUS.md` "맡긴 일" 절)에서 확인한다. `AGENTS.md`에는 완료 이력, backlog 또는 issue 상태를 중복 기록하지 않는다.
