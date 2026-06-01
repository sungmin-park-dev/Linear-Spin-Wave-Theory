---
frontmatter-version: 1
title: Project Knowledge Philosophy — 프로젝트 정의 및 지식 관리 체계 (초안)
section: idea-proposals
status: draft
last-edited-by: claude
created: 2026-05-30
updated: 2026-05-31
destination: GOVERNMENT/User-Constitution/project-definition/projects/  ← 확정 후 이동 대상
related: idea-proposals/2026-05-30-layer-aware-root-system.md
---

# Project Knowledge Philosophy — 프로젝트 정의 및 지식 관리 체계

> **이 문서의 위상**: AAD 대시보드에서 "프로젝트란 무엇인가"를 정의하는 기초 철학 문서.
> 내용 확정 후 `GOVERNMENT/User-Constitution/project-definition/projects/`로 이동한다.
> 현재는 초안 — 하단 [미논의 사항] 섹션에 추가 논의가 필요한 항목이 구분돼 있다.

---

## 1. 핵심 원칙: 두 종류의 지식
 **(user-comment: 지식에 대한 두 구분과 에이전트의 역할, 분리 원칙 등은 현재 agentic ai의 최첨단의 방법론과 철학을 반영하는지, 또 잠재적인 문재는 없는지 매우 매우 엄밀하게 검토할 것. 이 검토과정은 웹 검색을 이요한 조사과정도 포함해야한다.)**

> ⚠️ **보완 필요 — 웹 조사 결과**
> 1. **분리 원칙에 보안 이유 추가**: OWASP LLM Top 10 2025 1위가 prompt injection. 콘텐츠(project content)와 지침(agent operating knowledge)이 같은 레이어에 혼재하면 보안 공격 표면이 생긴다. "혼동 방지" 외에 보안 근거도 원칙에 포함시킬 것.
> 2. **2-Type 분류의 추상화 레벨 명시 필요**: 최신 연구(CoALA, MIRIX)는 에이전트 메모리를 episodic / semantic / procedural / resource 등 4–6 유형으로 분류한다. 이 문서의 2-Type 구분은 **파일 조직 구조** 레벨이지 에이전트 인지 아키텍처 레벨이 아님을 명시해야 혼동을 막을 수 있다.

프로젝트의 지식은 — 파일 조직 구조 레벨에서든, 지식의 본질 레벨에서든 — 두 종류로 구분된다.
1. **Project Content Knowledge**: 프로젝트의 목표, 배경, 맥락, 구현물, 필요 지식 등.
2. **Agent Operating Knowledge**: 에이전트가 그 프로젝트에서 **무엇을 어떻게 일하는가**에 대한 내용.
위 구분은 이후의 섹션에서 구체화한다.


AAD의 프로젝트에서 관리하는 프로젝트는 다음의 원칙을 이야기한다.
- **분리 원칙**: 
	- 에이전트 운영 지식은 프로젝트 콘텐츠와 반드시 다른 레이어에 존재해야 한다. 
	- 이는 에이전트가 "무엇을 만들어야 하는가"와 "어떻게 일하는가"를 혼동하지 않도록 하기 위함이다.
- **에이전트의 역할**: 
	- **운영실무자**: 에이전트는 사용자의 의도와 목표를 실현하는 과정에서 보조하는 worker이자 steward다. 
	- 자율적 권한은 없고, 사용자의 승인과 검토에 따라 움직인다.
	- 단, 제안은 자유롭게 할 수 있고, 사전에 허가된 특정 영역 또는 특정 작업에 대해서도 자유롭게 작업할 수 있다. (Working-Pad, Agents-Bylaws가 해당.)
	- 이 원칙이 지식 구조의 모든 설계 결정에 우선한다.

## 2. 프로젝트 폴더 구조

### 개요
- **이 프로젝트에 어떤 폴더 구조가 필요한가**에서 출발한다. 
- 프로젝트 유형/특성은 프로젝트마다 정의될 정도로 무의미할 정도로 다양하기 때문이다.

### 고려할 점 및 원칙
1. 폴더 구조는 사용자의 편의와 에이전트의 효율성을 모두 고려해야한다.
2. 폴더 구조(tree, name)는 개념적으로 모호함이 없어야하고, 사용자가 해당 프로젝트가 무엇을 하는지 알 수 있을 정도로 선명해야한다. 이를 위해서는 폴더가 다루는 내용을 모호하지 않게 이름에 반영되어야한다.
	- 예를 들어, 웹 개발에서 design은 디자인 설계서인지 또는 구현인지에 따라서 code 구현인지 디자인 지침서인지가 달라진다. 이 예시가 시사하는 점은 다음과 같다. 
		  1. 루트 디렉토리의 도메인의 폴더명은 프로젝트 맥락과 연결되어 있다. 
		  2. 루트 디렉토리를 구분하는 축은 추상적인 레벨에서 다뤄져야한다.
3. 한 프로젝트 내에서 폴더 구분 기준은 통일되어야하고, 예외가 없어야한다. 
4. 폴더 구조는 에이전트의 업무 효율을 저해하지 않아야한다. 
	- 예를 들어, 작업을 진행할 때, 에이전트가 변경해야하는 파일이 분리되어 있거나 흩어져 있으면, 에이전트는 수정을 놓칠 수 있다. 


### 범용 루트 구조

```
project-root/
  AGENTS.md              ← 에이전트 지침 (필수)
  GOVERNMENT/            ← 에이전트/사용자 운영 지식 (universal)
    User-Constitution/   ← 원칙·정의 (인간 직접 저작, 에이전트 수정 금지)
    Court-Precedents/    ← 결정 기록·판례 (vault-staging 파이프라인)
    Agents-Bylaws/       ← 운영 절차·가이드 (에이전트 유지)
    Working-Pad/         ← 진행 중 작업 상태 (공유)
  [name]-space/          ← 프로젝트별 콘텐츠 폴더 (-space suffix 컨벤션)
```

**-space 컨벤션**: 콘텐츠 폴더는 프로젝트 성격에 따라 자유롭게 명명한다. `-space` suffix를 권장한다.
예: `code-space/` `design-space/` `research-space/` `doc-space/` `work-space/` `data-space/` `theory-space/` `learn-space/`

**2-layer case** (보호할 확정 결정 없는 경량 프로젝트):
```
GOVERNMENT/
  Agents-Guidelines/
  Working-Pad/
```


### 콘텐츠 폴더 (-space)

**"범용 도메인 목록은 없다"는 절반만 맞다.** 폴더 *이름*은 프로젝트마다 다르지만, **명명 규칙(`-space` suffix)과 구분 축(workflow)은 범용**이다. 즉 범용 *목록*은 없어도 범용 *규칙*은 있다.

**독립 콘텐츠 폴더를 만드는 기준 — workflow 구분:**
- 툴·컨벤션·세션 성격이 다른가
- 분리가 사용자·에이전트의 탐색과 업무 효율을 높이는가
- 분리가 불필요한 작업량을 만들지 않는가

**판단 질문**: "이 영역의 작업 세션은 다른 영역 세션과 구분해서 들어가는가?" — 그렇다면 독립 폴더.

**이름 원칙**: 이름이 모호하면 경계도 모호하다. workflow를 명확히 반영해야 한다. 예를 들어 시각 작업은 `design-space/`, UI 구현은 `code-space/`로 — 같은 "design"이라도 작업 성격으로 가른다.

**주의**: 목적(experiment, infrastructure)이나 주제(theory)는 폴더 이름으로 부적합할 수 있다. 수치 실험·서버 구현 모두 코딩 세션이므로 `code-space/`에 속한다. 폴더는 **무엇을 위한가**가 아니라 **어떻게 작업하는가**로 구분한다.

### 예시 (작업 축별 대표 -space)

| 작업 축 | 대표 폴더 | 세션 성격 |
|---|---|---|
| 코드 구현 | `code-space/` | IDE·테스트·빌드 |
| 문서·API 레퍼런스 | `doc-space/` | 작성·렌더링 |
| 연구·이론 정리 | `research-space/` | 노트·수식·논문 합성 |
| 시각 설계 | `design-space/` | Figma·시안 |
| 사업·전략 | `biz-space/` | 기획·분석 |
| 데이터·실험 결과 | `data-space/` | 데이터셋·플롯 |

| 프로젝트 | 콘텐츠 폴더 | 비고 |
|---|---|---|
| quantum-cad | `product-space/` `code-space/` `research-space/` `biz-space/` | 제품 스펙·코드·연구·사업 4축 (작업 A 확정 2026-05-31) |
| AAD | `code-space/` `product-space/` | 코드 + 제품 정의 분리 (definition·roadmap) |
| LSWT | `code-space/` `doc-space/` `research-space/` | 라이브러리 + 문서 + 이론 |
| Study/ | `Mathematics/` `Physics/` … (분야=topic) | 단일 note workflow — 분야는 -space 아님 (작업 B 확정 2026-05-31) |

### GOVERNMENT/ 내부 구조

`GOVERNMENT/` 내부는 4-layer로 구성하되, 필요에 따라 subset으로 쓴다.

**Prefix 체계 — 기준은 "변경 권한"(누가 수정하는가)이지 저작 출처가 아니다:**
- `User-` = 에이전트 수정 **금지** (사용자 전용 편집)
- `Court-` = vault-staging 경유 (에이전트 제안 → 사용자 승인), 확정 후 안정
- `Agents-` = 에이전트 편집 (사용자 승인 권고)
- prefix 없음(`Working-Pad`) = 공유 편집

> **판별 사례 (저작 ≠ 변경권한)**: `user-instructions/`는 출처가 사용자(`source: user-instruction`)지만 에이전트가 관찰·갱신하므로 `Agents-Bylaws/`에 둔다. 저작 출처는 frontmatter `source:`가 기록하고, 폴더 prefix는 변경 권한을 나타낸다. — "절차처럼 보여도 누가 수정하는가"로 가른다.

| 레이어 | CoALA 유형 | 용도 | 쓰기 권한 |
|---|---|---|---|
| `User-Constitution/` | Semantic(운영) | 원칙·정의·확정 결정 | 인간만 |
| `Court-Precedents/` | Episodic | 결정 기록·판례 | vault-staging 경유 |
| `Agents-Bylaws/` | Procedural | 운영 절차·가이드 | 에이전트 |
| `Working-Pad/` | Working | 진행 중 작업 상태 | 공유 |

**Court-Precedents/ 독립 근거**: 헌법(원칙)과 판례(축적 결정)는 의미론적으로 다른 종류 — CoALA의 Semantic vs Episodic 구분과 일치.

### SOTA 정당화

| 프레임워크 | GOVERNMENT/ 매핑 |
|---|---|
| CoALA | Procedural + Episodic + Operational Semantic |
| Karpathy LLM Wiki | schema/ + wiki/ + source/ |
| Karpathy LLM OS | Disk (operating layer) |

우리 구조 = LLM Wiki + CoALA Episodic(Court-Precedents/) + 권한 계층(User-/Agents-/Court-)

> CORPUS/LIBRARY/ 탐색 기록 → `idea-proposals/2026-05-31-doctrine-corpus-library-exploration.md`

---

## 3. 프로젝트별 폴더 구조

공통 원칙: **에이전트 운영 지식(GOVERNMENT/)은 프로젝트 콘텐츠와 루트 레벨에서 분리**한다. 콘텐츠 폴더는 `-space` suffix 컨벤션으로 프로젝트 성격에 따라 자유롭게 명명한다.

---

### 1. quantum-cad

스타트업 운영 프로젝트. 코드·연구·사업이 독립 공존하는 복합 구조. **(작업 A 확정 2026-05-31)**

현재: `agent-knowledge/`(운영: wiki + working-pad) + `qcad-workspace/`(operations/ programs/ research/) + `qcad-codespace/`(코드).

**확정 구조 — AAD와 동형(同型). `operations/`의 혼재(제품 스펙 + 사업)를 분해한다:**
```
quantum_cad/
  AGENTS.md
  GOVERNMENT/          ← agent-knowledge/ 재편
    User-Constitution/ ← (신규 — 운영 원칙)
    Court-Precedents/  ← wiki/decisions/
    Agents-Bylaws/     ← wiki/ 나머지 (agents-playbook·user-instructions·llm-wiki·log) + repo-scripts/ → tools/
    Working-Pad/       ← working-pad/ 통째 (TASK-QUEUE·handoff·inbox·issue-notes·proposals)
  product-space/       ← operations/cad/ (architecture·core·validation) + project-identity.md — 제품 기술 정의
  code-space/          ← qcad-codespace/
  research-space/      ← qcad-workspace/research/ (papers·concepts·notes)
  biz-space/           ← operations/strategy/ + programs/ (army-competition·tex-corps)
```

**마이그레이션 매핑:**

| 현재 | 성격 | 이동 대상 |
|---|---|---|
| `operations/cad/` (architecture·core·validation) | 제품 기술 스펙 | `product-space/` |
| `operations/project-identity.md` | 제품 정체성 | `product-space/` |
| `operations/strategy/` (business-model·market·mvp·positioning·roadmap) | 사업 | `biz-space/strategy/` |
| `programs/` (army-competition·tex-corps) | 사업 이니셔티브 | `biz-space/programs/` |
| `qcad-workspace/research/` (papers·concepts·notes) | 연구 | `research-space/` |
| `qcad-codespace/` | 코드 | `code-space/` (개명) |
| `agent-knowledge/wiki/decisions/` | Operating-Episodic | `GOVERNMENT/Court-Precedents/` |
| `agent-knowledge/wiki/` (나머지) | Operating-Procedural | `GOVERNMENT/Agents-Bylaws/` |
| `agent-knowledge/working-pad/` | Operating-Working | `GOVERNMENT/Working-Pad/` |
| `repo-scripts/` (check-frontmatter·generate-agents·sync) | Operating-tools | `GOVERNMENT/Agents-Bylaws/tools/` |

**boundary 해소 ("결론난 지식 vs 진행 중 업무"):** 이 축은 분류축이 아니다. 활성 program(army-competition 등)이어도 그 **산출물**(BMC·발표·시장조사)은 콘텐츠 → `biz-space/`, **진행 추적**(다음 액션·마감)만 → `Working-Pad/`. active/concluded가 아니라 **artifact vs work-state**로 가른다. program 진행 추적은 루트 `Working-Pad/` 단일 TASK-QUEUE로 통합(분산 시 누락 위험). tex-corps 자체 AGENTS.md는 `biz-space/programs/tex-corps/` 컨텍스트 노트로 흡수(중첩 거버넌스 미설치).

---

### ai-automation-dashboard

소프트웨어 제품 개발 + bootstrapping 케이스. 현재 `knowledge/`가 **Operating 지식과 Content 지식을 혼재** — 핵심은 `project-definition/`(제품 정의 = Content)이 보호 레이어(user-vault)에 들어가 있는 것. [미논의 O] 해소 방향으로 분리한다.

**확정 구조 (2026-05-31 설계 확정, 물리 이동·코드는 후속):**
```
ai-automation-dashboard/
  AGENTS.md
  GOVERNMENT/            ← Operating: 어떻게 일하는가
    User-Constitution/   ← Semantic: 원칙·정의
      principles/        ← 순수 원칙 (project-operating-principles)
      methodology/       ← philosophy·layer-aware·cross-repo (draft 졸업 시)
    Court-Precedents/    ← Episodic: decisions/ 통째로 (원본 보존; 증류는 distill-decision)
    Agents-Bylaws/       ← Procedural
      coding-agent/ policies/ templates/ skills/   ← agents-playbook/
      user-instructions/                           ← project-wiki/user-instructions/
    Working-Pad/         ← handoff/ idea-proposals/ inbox/ issue-notes/ vault-staging/
  product-space/         ← Content: 무엇을 만드는가
    definition/          ← project-definition/ (제품 정의·정체성·기능·UI)
    roadmap/             ← project-roadmap/ (제품 계획)
  code-space/            ← 제품 코드 (구 product-workspace/)
  spike/                 ← 프로토타입·실험 코드 (위치 재검토 필요)
```

**마이그레이션 매핑:**

| 현재 | 성격 | 이동 대상 |
|---|---|---|
| `user-vault/project-definition/` | Content | `product-space/definition/` |
| `project-wiki/project-roadmap/` | Content | `product-space/roadmap/` |
| `user-vault/decisions/` | Operating | `GOVERNMENT/Court-Precedents/` |
| `user-vault/principles/` (순수 원칙) | Operating-Semantic | `User-Constitution/principles/` |
| `user-vault/principles/` (절차성: writing-style·surface-pattern·md-policy) | Operating-Procedural | `Agents-Bylaws/policies/` |
| `project-wiki/agents-playbook/` | Operating | `GOVERNMENT/Agents-Bylaws/` |
| `project-wiki/user-instructions/` | Operating | `GOVERNMENT/Agents-Bylaws/` |
| `working-pad/*` | Operating | `GOVERNMENT/Working-Pad/` |
| 방법론 문서(philosophy·layer-aware·cross-repo) | Operating(일차 역할) | `GOVERNMENT/User-Constitution/` |
| `product-workspace/` | 코드 | `code-space/` (개명) |

---

### Linear-Spin-Wave-Theory
- Python 라이브러리 개발 (현재 휴면). 순수 코드 프로젝트 — 별도 도메인 분리 없음.
- 추후 변경안 (파일명 미정)
	- lswt 이론에 대한 정리를 위한 폴더
		- lswt code 구현을 위한 지식
		- spin wave 관련 논문, 도서 공부 내용
	- lswt 코드에 대한 구현

```
Linear-Spin-Wave-Theory/
  GOVERNMENT/         ← 미설치 (추후 활성화 시 추가)
  code-space/         ← lswt/ + tests/ (패키지 + 테스트)
  doc-space/          ← docs/ + examples/ (API 문서 + 사용 예시)
  research-space/     ← spin wave 이론 정리, LSWT 알고리즘 이해
  legacy/
```

---

### learning 시리즈 (Tensor-Network-Study · ml-study · nqs-learning) — 작업 B 확정 2026-05-31

**핵심: "통일 구조"는 단일 폴더 템플릿이 아니라 판단 규칙이다.** 세 레포의 실제 형태가 전부 달라 같은 템플릿을 강제하면 틀린 구조가 된다. Section 4 레이어 조건을 적용해 각자 다른 구조를 갖는다.

| 레포 | 실제 형태 | 구조 |
|---|---|---|
| **Tensor-Network-Study** | 성숙 build-and-learn (src/ 라이브러리 + Projects/ 실험 + Docs/Benchmarks/Tutorials) | 2-layer GOVERNMENT/ + `code-space/` + `research-space/` (+ `learn-space/` 선택) |
| **ml-study** | vendored 레퍼런스 모음 (llm.c·makemore·micrograd·nanoGPT 클론, 각자 LICENSE) | 1-layer (AGENTS.md "스터디용 클론, upstream 수정 금지") + 레포 그대로 |
| **nqs-learning** | 배아 (노트북 1개 + netket venv) | 1-layer, 성장 시 승급 |

**판단 규칙**: 코드+이론 둘 다 능동 생성 → -space 분리 / vendored 외부 코드 → 참조 저장소 / 노트북 한두 개 배아 → AGENTS.md만.

Tensor-Network-Study 구조 (성숙 케이스):
```
Tensor-Network-Study/
  GOVERNMENT/         ← 2-layer (Agents-Guidelines/ + Working-Pad/) — 학습 프로젝트, 보호 레이어 약함
  code-space/         ← src/ + Projects/ (라이브러리 + 개별 실험)
  research-space/     ← Docs/ (이론 노트) + Benchmarks/ (검증 결과)
  learn-space/        ← Tutorials/ (research와 세션 성격 같으면 research-space로 흡수)
```

**Study/ ↔ learning-repo 중첩** (예: Study/CS/Tensor Networks vs TNS/research-space): Study/ = 프로젝트 초월 **항구적** 이론 노트, repo `research-space/` = **그 코드 개발에 종속된** 노트. 이 경계로 가른다.

---

### Study/ (iCloud Obsidian)

지식 축적 시스템. 학문 분야별 폴더 구조.

```
Study/
  AGENTS.md            ← "정리·연결·요약 역할, 새 내용 임의 작성 금지"
  GOVERNMENT/          ← _Vault-Operation/ 대체 (명명 통일). 지식축적형 → subset
    User-Constitution/ ← 01 Constitution/ (Vault Principles)
    Agents-Bylaws/     ← 02 Law/ (Conventions·Workflows·frontmatter-policy·Logs·Migrations)
      tools/           ← 04 Tools/ (check_frontmatter.py 등 — 정책과 co-locate)
    Working-Pad/       ← 03 Working Pad/ (Handoff·Inbox·Issue Notes·Task Queue·Structure Reviews)
  Mathematics/         ← 분야 = topic, -space 아님 (수학/물리 노트 = 동일 note workflow)
  Physics/
  Computer Science/
  Electrical Engineering/
  Lectures/  Materials/  Ideas/  Workspace/   ← cross-cutting
  _Assets/
  maps/
```

> **명명 통일** (2026-05-31): `_Vault-Operation/` → `GOVERNMENT/`. **분야 폴더 -space 미적용 확정**: 분야는 *topic*이지 *workflow*가 아니므로(전부 Obsidian 노트 세션) -space 부적합 — Study/ 전체가 단일 note space, 분야는 그 안의 토픽 분류. **`04 Tools/` → `Agents-Bylaws/tools/`** (frontmatter-policy.yaml 정책과 check_frontmatter.py 집행을 같은 레이어에). `Court-Precedents/`는 생략(개인 학습엔 선례 축적 약함 — Section 4). Workspace/Study Plans는 진행 계획 성격이면 Working-Pad로 이동 검토.

---
### arxiv-ingest

arXiv 논문 수집 도구. 소형 Python 앱. `/Users/david/Tools/` 에 위치 — GitHub 프로젝트가 아닌 로컬 유틸리티.

```
arxiv-ingest/
  core.py
  web_app.py
  menubar_app.py
  arxiv-ingest.command
  docs/
  templates/
  requirements.txt
  SPEC.md
```

> AAD 독립 프로젝트로 등록 예정. 현재 AAD 불안정으로 보류 중. 검색 엔진 기능은 다른 프로젝트에서도 재사용 예정.
> 검색 엔진은 다른 프로젝트에서 사용할 수 있으니 코드 구현 구조가 단순하고 모듈화되어야함.

---


## 4. GOVERNMENT/ 레이어 필요성 판단

Type 분류를 폐기했으므로, "어떤 유형의 프로젝트인가"가 아니라 **"각 레이어의 필요 조건이 충족되는가"**로 GOVERNMENT/ subset을 정한다.

### 레이어별 필요 조건

| 레이어 | 필요 조건 | 불충족 시 |
|---|---|---|
| `Working-Pad/` | 항상 (진행 중 작업이 없는 프로젝트는 없다) | — |
| `Agents-Bylaws/` | 에이전트가 따를 절차·규칙·가이드가 있을 때 | 생략 → 2-layer에서는 `Agents-Guidelines/`로 대체 |
| `Court-Precedents/` | 결정 이력이 **선례로 축적·재참조**될 때 | 생략, 결정은 `User-Constitution/`에 흡수 |
| `User-Constitution/` | 아래 3조건 **모두** 충족 시 | 생략 → 2-layer |

### User-Constitution/ 필요 3조건

보호 레이어의 존재 이유 = **에이전트가 건드리면 안 되는 인간 승인 사실을 보호**. 다음이 모두 참일 때 가치 있다:
1. 에이전트가 콘텐츠를 정기적으로 작성·수정한다
2. 보호해야 할 "확정된 사실·결정"이 존재한다
3. 그 사실이 바뀌면 프로젝트 방향이 달라진다

### 학습·연구 프로젝트에서 보호 레이어가 약한 이유

- 학습 목표·연구 방향은 발견에 따라 자연스럽게 바뀐다
- 외부 이해관계자가 없어 "보호 기준선"이 약하다
- "승인된 사실" 개념이 개인 학습에선 불분명하다

→ 이 경우 보호 레이어를 생략하고 **2-layer**(`Agents-Guidelines/` + `Working-Pad/`)를 쓴다. 연구 범위를 명시적으로 고정하고 싶으면 `Agents-Guidelines/scope.md`로 처리한다.

---

## 5. GOVERNMENT/ subset 선택 패턴

Section 4 조건을 적용한 결과 패턴. Type이 아니라 **레이어 조건 충족 여부**로 결정된다.

### Full 4-layer — 운영 프로젝트 (보호 결정 + 선례 축적 + 절차 모두)
```
GOVERNMENT/
  User-Constitution/    ← 원칙·정의
  Court-Precedents/     ← 결정 기록·판례
  Agents-Bylaws/        ← 절차·가이드
  Working-Pad/          ← TASK-QUEUE, handoff, issue-notes, vault-staging
[name]-space/ ...
AGENTS.md
```

### 3-layer — 보호 결정 있으나 선례 축적 불필요 (Court-Precedents 생략)
```
GOVERNMENT/
  User-Constitution/    ← 결정도 여기 흡수
  Agents-Bylaws/
  Working-Pad/
AGENTS.md
```

### 2-layer — 보호 결정 없는 학습·연구·경량
```
GOVERNMENT/
  Agents-Guidelines/    ← Constitution 없으니 종속 없는 최상위 지침 (범위 고정은 scope.md)
  Working-Pad/
AGENTS.md
```

### 1-layer — 참조 전용 휴면 프로젝트
```
AGENTS.md               ← "참조 전용, 수정 제안 자제" — 이것으로 충분
```

### 지식 축적형 — 에이전트 역할이 write가 아닌 organize (예: Study/)
```
GOVERNMENT/ (subset)
분야 폴더 ...            ← topic 분류 (Mathematics/ Physics/ …) — 단일 note workflow이므로 -space 아님
maps/                   ← 내비게이션 맵 (map-*.md)
AGENTS.md               ← "정리·연결·요약 역할, 새 내용 임의 작성 금지"
```

### 복합 도메인 내부 구조 메모
- `User-Constitution/`의 `decisions/`·`definitions/`: 도메인 구분은 파일 prefix 또는 하위 폴더
- `Agents-Bylaws/{domain}/`: 도메인별 절차 분리 가능
- `Working-Pad/`: 단일 TASK-QUEUE로 도메인 통합 (분산 시 누락 위험)

---

## 6. AAD 대시보드 연결

> 상세 UI 설계는 **작업 E (AAD Knowledge 탭 분리)**로 위임. 여기서는 원칙만 둔다.

**핵심 원칙**: AAD는 프로젝트를 `GOVERNMENT/`(운영 지식)과 `-space`(콘텐츠)로 **분리 인식**한다.

- **인식 메커니즘**: 폴더 이름이 아니라 `project_roots.kind` 등록으로 (각 프로젝트가 콘텐츠 폴더를 자유 명명해도 AAD가 인식)
- `GOVERNMENT/` → 에이전트 패널·지침 뷰어·작업 관리
- `[name]-space/` → 성격별 에디터 모드 (`code-space` → 코드 에디터, `research-space`·`doc-space` → 지식 에디터)
- **장기 비전**: 한 창에서 코드 편집(VSCode류) + 지식 작업(Obsidian류) + 에이전트 대화 통합

**표시 강도는 GOVERNMENT/ subset과 -space 구성에서 자연히 도출된다** (Type 표 불필요):

| 프로젝트 구성 | Knowledge | Work | Management | 주목적 |
|---|---|---|---|---|
| Full 4-layer + 다수 -space | ✅ 핵심 | ✅ 핵심 | ✅ | 전체 운영 |
| 2-layer + 단일 -space | ✅ 경량 | ✅ | △ | 연구·진행 추적 |
| 지식 축적형 (organize 역할) | △ 탐색용 | ❌ | ❌ | 지식 접근 |
| 1-layer (AGENTS.md만) | ❌ | △ 가시성 | ❌ | 포트폴리오 카드 |

---

---

# [미논의 사항 — 추가 논의 필요]

아래 항목은 이 문서 초안 작성 시점에 아직 논의되지 않은 내용이다.
확정 전 반드시 검토가 필요하다.

> ⚠️ **Type 분류 폐기 반영 필요**: 아래 항목 중 `Type 1~4`를 전제로 한 질문(A·B·C·E·G·H 등)은 **Type 분류 폐기로 무효이거나 GOVERNMENT/ subset 기준으로 재구성**해야 한다. 본문 Section 4·5는 이미 조건 기반으로 재작성됨 — 미논의 항목의 개별 재구성은 후속 설계 토론(작업 A·B)에서 다룬다.
> - **[미논의 O]**(bootstrapping)·**[미논의 P]**(레이어 명명)는 본 세션에서 논의/확정됨.

---

## A. Type 1 복합 도메인 — 에이전트 지식 내부 구조 상세

**미결 질문:**
- `project-wiki/` 내부를 도메인별로 나눌 때 어느 수준까지 분리하는가?
  - `project-wiki/code/`, `project-wiki/theory/`, `project-wiki/strategy/`?
  - 아니면 주제(기능·모듈)별로?
- `user-vault/decisions/`에서 도메인 맥락을 어떻게 구분하는가?
- 복합 도메인 프로젝트의 AGENTS.md는 도메인별로 별도 지침을 포함해야 하는가?

## B. Type 2 — user-vault 기준선 불필요 확정 여부

**미결 질문:**
- "연구 범위 고정"이 필요한 경우 `project-wiki/scope.md`로 대체한다고 했는데, 이것이 충분한가?
- Type 2에서 user-vault를 완전히 제거하는 것이 원칙인가, 아니면 선택적으로 추가 가능한가?

## C. Type 3 — 에이전트의 역할 경계

**미결 질문:**
- Type 3 지식 베이스에서 에이전트는 무엇을 할 수 있고 무엇을 할 수 없는가?
  - 허용: 정리, 링크 추가, 맵 업데이트, 요약?
  - 금지: 새 노트 생성, 기존 내용 수정?
- `inbox/` 처리 절차는 어떻게 정의하는가?
- Type 3가 다른 Type 1 프로젝트의 이론적 기반이 될 때, 그 연결을 어떻게 관리하는가?
  - 예: Study/ → quantum_cad의 theory 도메인

## D. 프로젝트 생애주기 (Lifecycle)

**미결 질문:**
- 프로젝트가 Type 간에 이동할 수 있는가?
  - Type 2(학습) → Type 1(공개 라이브러리)? (Tensor-Network-Study → 공개 패키지)
  - Type 1(활성) → Type 4(아카이브)?
- 생애주기 상태(active / dormant / complete / archived)를 AAD가 어떻게 표시하는가?
- Linear-Spin-Wave-Theory처럼 군 휴학으로 일시 중단된 Type 1 프로젝트는 어떻게 다루는가?

## E. Type 4 — AAD 연결의 가치 정의

**미결 질문:**
- Type 4를 AAD에 연결하는 구체적 가치는 무엇인가?
  - "포트폴리오 가시성"이 실제로 유용한가?
  - Work 탭 없이 Project Board에만 등록하는 것으로 충분한가?
- 연결하지 않는 것이 더 나은 케이스가 있는가?

## F. paper-finder 분류

**미결 질문:**
- `/Users/david/GitHub/paper-finder` 의 실제 성격 미확인.
- Type 2(능동 연구 도구 개발)인지, Type 4(참조 도구)인지 확인 필요.

## G. AAD 대시보드 UI — 유형별 표시 방식

**미결 질문:**
- 각 Type에 따라 AAD 대시보드가 다르게 표시되어야 하는가?
  - Type 1: Knowledge + Work + Management 모두 표시
  - Type 2: 간소화된 뷰?
  - Type 3: 탐색 중심 뷰?
  - Type 4: 카드만?
- Project Board에서 Type을 명시적으로 표시해야 하는가?

## H. Study/ (Type 3)와 Type 1 프로젝트의 관계

**미결 질문:**
- Study/의 특정 도메인(물리, CS 등)이 quantum_cad나 LSWT의 이론 도메인과 겹친다.
- 이 겹침을 AAD에서 어떻게 표현·관리하는가?
- Study/를 AAD에 연결하는 것이 적절한가, 아니면 별도 접근이 필요한가?

## I. "프로젝트"의 경계 정의

**미결 질문:**
- AAD에서 "프로젝트"로 등록 가능한 최소 조건은 무엇인가?
- 아직 시작하지 않은 아이디어(pre-project)를 어떻게 다루는가?
- 개인 프로젝트와 협업 프로젝트를 구분해야 하는가?

## J. 복합 도메인 내부 속성 — 추가 도메인 우선순위

**미결 질문:**
- Code / Theory / Strategy / Data / Experiment / Documentation / Design / Infrastructure 중에서
  현재 포트폴리오에서 실제로 관련 있는 도메인은 어디까지인가?
- 각 도메인이 에이전트 지식 구조에 어떤 영향을 미치는지 케이스별 상세 분석 필요.

---

## K. 두 유형의 경계 불명확 케이스

**비판**: "프로젝트 배경 이론 노트"(예: 물리 공식 정리, 도메인 지식 요약)는 Project Content Knowledge인가, Agent Operating Knowledge인가? 에이전트가 작업에 참조하면 Operating이고, 프로젝트 산출물이면 Content다. 같은 파일이 두 성격을 동시에 가질 수 있다.

**미결 질문:**
- 하나의 파일이 두 유형 모두에 해당할 때 어떻게 처리하는가?
- 경계 케이스의 판단 기준을 명시해야 하는가?

## L. 에이전트가 콘텐츠를 생성할 때의 레이어 전환

**비판**: 에이전트는 결국 Project Content를 직접 생성한다(코드 작성, 문서 초안 등). 이 순간 레이어 경계를 넘는다. Section 1의 분리 원칙은 "어디에 존재하는가"를 정의하지만, "언제 어떻게 콘텐츠 레이어로 이동하는가"(전환 절차)를 다루지 않는다. vault-staging이 일부 해결하지만 원칙 레벨에서 명시가 없다.

**미결 질문:**
- 에이전트 생성 결과물이 Project Content가 되는 시점과 절차를 원칙에 포함시킬 것인가?

## M. "사전 허가 구역"이 만드는 간접 권한 긴장

**비판**: 에이전트는 working-pad에 자유롭게 쓸 수 있고, vault-staging → user-vault 파이프라인이 존재한다. 이는 에이전트가 working-pad 경유로 user-vault에 간접 영향을 줄 수 있음을 의미한다. "자율적 권한 없음"이라는 원칙과 실제 구조 사이의 긴장이 있다.

**미결 질문:**
- vault-staging 파이프라인에서 에이전트 역할의 한계를 명시해야 하는가?
- 이 긴장을 원칙 레벨에서 인정하고 다룰 것인가?

## N. Agent Operating Knowledge 내부 이질성

**비판**: 결정 기록(decisions), 절차(procedures), 세션 상태(handoff), 진행 작업(task-queue)을 모두 "Agent Operating Knowledge"로 묶는다. 그러나 이들의 수명, 변경 주체, 접근 패턴이 완전히 다르다. 예: 아키텍처 결정(수년간 유효)과 오늘의 task-queue(수 시간 유효)는 같은 유형으로 취급하기 어렵다.

**미결 질문:**
- Agent Operating Knowledge 내부를 하위 유형으로 분류해야 하는가?
- 현재 3-layer 구조(user-vault / project-wiki / working-pad)가 이 이질성을 충분히 해결하는가, 아니면 추가 분류가 필요한가?

## P. 에이전트 운영 지식 내부 레이어 명명 규칙 ✅ **확정 (2026-05-31)**

**결정 사항**: 두 명명 체계를 통합, 인스티튜션 메타포(User/Court/Agents) prefix + 4-layer 구조 확정.

| 레이어 | 확정 이름 | 구 이름 | CoALA 유형 |
|---|---|---|---|
| Layer 1 | `User-Constitution/` | `user-vault/` | Semantic(운영) |
| Layer 2 | `Court-Precedents/` | (없음, 신규) | Episodic |
| Layer 3 | `Agents-Bylaws/` | `project-wiki/` | Procedural |
| Layer 4 | `Working-Pad/` | `working-pad/` | Working |

**Prefix 체계:**
- `User-` = 인간 직접 저작, 에이전트 수정 금지
- `Court-` = 기관(프로젝트) 축적, vault-staging 파이프라인 (agent proposes → user approves)
- `Agents-` = 에이전트 유지, 자유롭게 읽기·쓰기
- prefix 없음(`Working-Pad`) = 공유

**Court-Precedents/ 독립 근거**: 헌법(원칙)과 판례(축적 결정)는 의미론적으로 다른 종류 — CoALA의 Semantic vs Episodic 구분과 일치.

**통일 근거**: `Constitution/Law/Working Pad` 명명에서 출발, 인스티튜션 메타포를 조합해 권한 주체를 prefix로 명시. 프로젝트 전체 통일 적용.

> 이 결정은 2026-05-31 세션에서 확정됨. GOVERNMENT/ 최상위 폴더 명칭 확정과 동시에 이루어짐.

## O. Knowledge가 제품인 프로젝트 — meta/object 레이어 미구분 문제

**배경**: AAD는 "프로젝트 지식 관리 시스템"을 설계하면서 동시에 그 시스템 안에서 AAD 개발을 운영한다. 이는 **bootstrapping 문제**다 — 컴파일러가 자기 자신을 컴파일하는 것처럼, 시스템이 자신을 설계하는 동안 그 시스템 안에서 동작하고 있다.

**현재 상태**: AAD의 `knowledge/` 폴더는 두 종류의 지식을 혼재하고 있다.
- **Object-layer (제품 지식)**: AAD가 정의·설계하는 것. 예: Project Knowledge Philosophy, 3-layer 템플릿, 프로젝트 분류 체계 → 이것이 AAD의 산출물.
- **Meta-layer (개발 지식)**: AAD를 만들기 위한 에이전트 운영 지식. 예: arch 결정, task-queue, handoff → 이것이 AAD 개발의 Operating Knowledge.

이 혼재는 실수가 아니라 이 상황의 필연적 속성이었다.

**정정**: 두 종류의 경계 자체가 흐릿해지는 것은 아니다. 개념적으로는 항상 구분 가능하다. 다만 Philosophy 같은 문서가 **draft 상태**일 때, 확정되지 않은 제품 산출물이면서 동시에 현재 작업의 기준으로도 쓰인다 — 이는 파일의 **상태(status) 문제**이며, vault-staging이 다루는 영역이다.

**해소 (2026-05-31):** 구분한다. 핵심 통찰 — 진짜 혼재는 `project-definition/`(제품 정의 = Content)이 보호 레이어(user-vault)에 섞인 것이었다.

- **Content**(제품이 무엇인가: 정의·정체성·기능·UI·로드맵) → `product-space/`
- **Operating**(어떻게 일하는가: 결정·원칙·절차·진행 작업) → `GOVERNMENT/`
- **방법론 문서**(philosophy·layer-aware·cross-repo-template)는 이중 성격(제품이 구현하는 개념 + 모든 프로젝트 운영 기준)이나 **일차 역할이 운영 기준**이므로 `GOVERNMENT/User-Constitution/`. AAD 제품이 이를 구현하는 건 `code-space/` 코드가 담당 — 지식을 별도 복제하지 않는다.
- **draft → canonical**: `Working-Pad/vault-staging/` → 사용자 승인 → `User-Constitution/` 또는 `Court-Precedents/`. (status 문제는 Working-Pad가 흡수)

> 상세 마이그레이션 매핑은 Section 3 (ai-automation-dashboard) 참조. 물리 이동·코드 마이그레이션은 후속 작업.





---


### 프로젝트 콘텐츠 (Project Content Knowledge)
프로젝트가 **만들어내는 것**. 프로젝트의 목적이자 산출물. 예시

- 소프트웨어 프로젝트: 소스코드, 테스트, 빌드 결과물
- 연구 프로젝트: 논문, 실험 결과, 수식 유도
- 운영 프로젝트: 전략 문서, 발표자료, 사업 계획
- 학습 프로젝트: 구현 코드, 노트북, 검증 결과

### 에이전트 운영 지식 (Agent Operating Knowledge)
에이전트가 그 프로젝트에서 **어떻게 일하는가**에 관한 메타 레이어.

- 결정 기록: 왜 이 방향으로 결정했는가
- 절차: 에이전트가 따라야 할 규칙과 프로세스
- 컨텍스트: 세션 간 연속성을 위한 상태 정보
- 진행 중 작업: handoff, issue-notes, task-queue



---
