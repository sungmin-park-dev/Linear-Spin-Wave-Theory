# CLAUDE.md

> LSWT 이론 문서 정립과 Python 패키지 개발 시 Claude가 따르는 저장소 지침.

---

## 프로젝트 개요

2D 스핀 모델의 Linear Spin Wave Theory 계산을 위한 Python 라이브러리.
NBCP(Na₂BaCo(PO₄)₂) 관련 논문(arXiv:2505.06398; npj Quantum Materials 2022, DOI 10.1038/s41535-022-00500-3) 결과를 재현할 수 있도록 공개 배포 목표.

**핵심 기능**: 격자 정의 → LSWT 대각화 → 물리량 계산 (열역학, 위상, 상관함수)

**설계 원칙**:
- 명확성 우선 — 물리적 의미가 코드에 드러나야 함
- 관심사 분리 — 시스템 정의 / 솔버 / 관측량(observables) / 시각화
- `AbstractSolver` 인터페이스로 다른 방법론(real-space BdG, ED, TN 등) 확장 가능
- `SpinSystem`은 solver-agnostic — LSWT 전용 로직을 넣지 않음
- Python config 사용 (YAML 아님) — exchange matrix(3×3)를 numpy로 직접 정의

---

## 현재 디렉토리 구조

> LSWT 일반 이론은 `docs/lswt/`, NBCP 연구 노트는 `docs/nbcp/`, 실행 예제와 검증 스크립트는 `examples/`에서 관리한다.
> 설계 사본은 `GOVERNMENT/Working-Pad/idea-proposals/2026-05-30-project-knowledge-philosophy.md`에 둔다.

```
project-root/
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml
│
├── GOVERNMENT/                  # 운영 지식
│   ├── User-Constitution/
│   │   ├── map-user-constitution.md
│   │   └── single-knowledge-canon.md
│   ├── Court-Precedents/
│   │   ├── map-decisions.md
│   │   ├── 2026-08-01-lswt-markdown-source-authority.md
│   │   └── 2026-08-09-lswt-docs-authoring-surface.md
│   ├── Agents-Bylaws/
│   │   ├── map-agents-bylaws.md
│   │   └── procedures/
│   │       ├── map-procedures.md
│   │       ├── lswt-canonical-document-lifecycle.md
│   │       └── theory_code_verification_plan.md
│   └── Working-Pad/
│       ├── TASK-QUEUE.md
│       ├── map-working-pad.md
│       ├── idea-proposals/
│       ├── issue-notes/
│       └── vault-staging/
│
├── code-space/                  # 코드 구현
│   ├── spintoolkit/             # system · states · methods · definitions · observables · visualization
│   ├── lswt/                    # 옛 이름 `lswt` 호환 패키지(DeprecationWarning)
│   └── tests/                   # pytest 테스트
│
├── docs/                        # 문서와 참고자료의 통합 진입점
│   ├── README.md
│   ├── development/             # Toolkit 개발 설계 Beamer; 구현·검증 기록은 부록
│   ├── lswt/                    # LSWT 일반 이론 Markdown
│   │   ├── 00-foundations/
│   │   ├── 01-derivation/
│   │   ├── 02-observables/
│   │   ├── 03-examples/
│   │   ├── 04-appendices/
│   │   └── sources/             # 원본·전사본·외부 논문·검증 근거
│   ├── nbcp/                    # NBCP 연구
│   │   ├── main.tex              # NBCP LaTeX 원본 진입점
│   │   ├── chapters/             # 본문 9장
│   │   ├── appendices/           # 상세 유도와 수치 검증
│   │   ├── references.tex        # 문헌과 source 역할
│   │   ├── preamble.tex          # 공통 서식
│   │   ├── metadata.tex          # 제목·날짜·검토 상태
│   │   ├── output/              # 생성 PDF와 생성 기록
│   │   └── sources/             # Overleaf 원문 대조 기록
│   └── archive/                 # 집필을 종료한 문서
│
├── model/                       # 모델별 정의·간단한 계산·원시 및 중간 결과
│   └── nbcp/                    # NBCP 모델 구성
│
├── examples/                    # 실행 예제와 검증 스크립트
│   ├── nbcp_ground_state.py
│   └── nbcp_hamiltonian_check.py
│
├── legacy/                      # 원본 legacy 코드 아카이브
└── data-space/                  # 검토하고 정돈한 결과 데이터
```

---

## 이론 문서와 source authority

LSWT 이론 내용은 `docs/lswt/`에 영어 Markdown으로 정리한다. 원본의 주장, 수식과 논리 연결을 추적할 수 있도록 보존하면서 문서를 개념별 owner로 나눈다. 재배치와 제외·보류 기록은 lifecycle 절차를 따르며, 물리적 의미나 수학적 타당성이 불명확한 부분을 임의로 고치지 않는다.

Markdown 단일 정본 원칙은 [2026-08-01 결정](GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md), 현재 작성 경로와 자료 위치는 [2026-09-16 통합 기록](GOVERNMENT/Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md)을 따른다. 이는 2026-08-09 결정의 경로 조항을 갱신한 사용자 승인 구조이며, 이론 내용의 acceptance를 변경하지 않는다. 전체 문서 진입점은 [docs/README.md](docs/README.md), LSWT 읽기 순서는 [docs/lswt/README.md](docs/lswt/README.md)다.

| 경로 | 역할 |
|---|---|
| `docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` | 원래 기록, 수식과 review annotation을 판정하는 primary evidence |
| `docs/lswt/sources/01-editable-notes/note_lswt_reviewed.tex` | 원문 section 순서를 유지한 editable transcription |
| `docs/lswt/sources/01-editable-notes/note_lswt_restructured.tex` | 재배치와 일부 편집이 포함된 structural reference |
| `docs/lswt/sources/02-reference-papers/` | 유도와 convention을 확인하는 외부 참고문헌 |
| `docs/lswt/sources/03-verification-notes/` | 수식, basis convention과 코드의 내부 검증 자료 |
| 사용자 승인 `docs/lswt/00-*`–`04-*`의 이론 Markdown | 현재 LSWT 이론 claim을 소유하는 유일한 theory canon |

상세 source inventory와 원문 section-to-docs 대응은 `docs/lswt/sources/README.md`에서 확인한다. `reviewed`와 `restructured` TeX를 새로운 단일 TeX master로 자동 병합하지 않으며, 유효한 내용은 primary PDF와 대조한 뒤 해당 `docs/lswt/` Markdown owner에서 통합한다. Source가 충돌하거나 부호, index, conjugation, normalization 또는 적용 조건이 불명확하면 `Unknown` 또는 open review item으로 남긴다. 자동 검사나 코드 수치 일치는 사용자의 Human Physics and Mathematics Review를 대체하지 않는다.

NBCP 연구는 [2026-09-18 결정](GOVERNMENT/Court-Precedents/2026-09-18-nbcp-latex-source-authority.md)에 따라 `docs/nbcp/main.tex`와 여기서 포함하는 `chapters/`, `appendices/`, `references.tex`를 유일한 내용 편집 원본으로 사용한다. `preamble.tex`는 공통 서식, `metadata.tex`는 제목·날짜와 검토 상태를 관리한다. 본문은 TeX에서 직접 수정하며 semantic label과 참조를 유지한다. `examples/nbcp_research_export.py`가 XeLaTeX로 `docs/nbcp/output/`에 PDF와 생성 기록만 저장하고, 빌드 중간 파일은 임시 폴더에서 처리한다. 기존 `research-note.md`는 안내만 제공하며 이전 Markdown·변환기는 날짜가 붙은 archive로 보존한다. 이 NBCP 한정 결정은 LSWT 일반 이론의 Markdown 정본 원칙이나 물리·수학 검토 상태를 바꾸지 않는다. 참고자료·navigation·archive·output은 내용 편집 원본이 아니다.

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

**향후 목표**:
```
SpinSystem ──┬── LSWTSolver(system).solve()  → SolverResult
             ├── EDSolver(system).solve()    → SolverResult  (미구현)
             └── BdGSolver(system).solve()   → SolverResult  (미구현)
```

---

## 코딩 컨벤션

- **Naming**: 모듈 `snake_case.py`, 클래스 `PascalCase`, 함수 `snake_case`, 상수 `UPPER_SNAKE_CASE`
- **Docstring**: NumPy 스타일
- **Import 순서**: 표준 라이브러리 → 서드파티 (`numpy`, `scipy`) → 로컬 (`spintoolkit.*`)
- **Type hints**: 사용 권장
- **언어**: 코드·주석·docstring은 영어, 사용자 대화는 한국어

---

## 작업 원칙

> 이 섹션은 Claude가 이론 문서와 코드 작업 시 반드시 따라야 하는 규칙이다.

1. **제안 우선**: 파일 생성·수정·삭제 전 반드시 변경 계획을 먼저 제시하고 승인 대기.
2. **인터페이스 변경은 토의 후 결정**: API(함수명, 데이터 구조, 클래스 인터페이스) 변경은 선 제안 → 토의 → 성민 확정 순서를 따름. 임의로 결정하지 않음.
3. **영향 범위 명시**: 모듈 간 의존성 변경이 생기면 영향받는 모듈을 명시할 것.
4. **물리적 의도 불명확 시 질문**: legacy 로직의 물리적 의미가 불분명하면 임의 해석하지 말고 반드시 질문할 것.
5. **작업 종류에 맞는 검증**: 코드 구현은 구현 → legacy 수치 대비 검증 → 성민 확인 → 다음 단계 순서로 진행하며, 검증 전 다음 구현 단계에 착수하지 않는다. 이론 문서는 `GOVERNMENT/Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md`에 따라 문체 교정, 원문 대조와 사용자 물리·수학 검토를 구분한다. 문체 교정에 코드 수치 검증을 일괄 요구하지 않는다.
6. **이론 source authority 준수**: 위 `이론 문서와 source authority` 구분을 따른다. LSWT 일반 이론의 공개 TeX·PDF·HTML은 accepted Markdown에서 생성하고, 사용자 검토용 preview는 draft/in-review Markdown에서도 생성한다. NBCP는 위에서 정한 canonical TeX에서 PDF를 생성하며 TeX 원본을 직접 편집한다. PDF 등 파생물을 직접 수정하지 않는다.
7. **이론 문체 preflight**: `docs/lswt/`의 reader-facing theory 문서를 작성하거나 교정하기 전에 `GOVERNMENT/Agents-Bylaws/policies/lswt-writing-style.md`를 읽는다. `status: in-review`이면 current working guidance로 적용하되 accepted policy로 보고하지 않으며, 문체 검토를 물리·수학적 acceptance로 간주하지 않는다.

현재 작업, 우선순위와 진행 상태는 `GOVERNMENT/Working-Pad/TASK-QUEUE.md`에서 확인한다. `CLAUDE.md`에는 완료 이력, backlog 또는 issue 상태를 중복 기록하지 않는다.
