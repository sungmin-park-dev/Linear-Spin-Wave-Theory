# 코멘트 · 흐름 점 단일 Q 나선을 회전틀에서 원시 셀 LSWT로 푼다 (D34)
<!-- rw-source: spiral-frame-ana -->

> 연구 작업대 앱의 기록. 질문에 답할 때는 그 질문 아래에 `### 답 · <이름> · YYYY-MM-DD HH:MM`과 답을 적고, 끝에 `- 상태: 답함`을 덧붙인다. 다른 기록은 고치지 않는다.

## c-20261021-1626 · 질문 · 전체
<!-- rw: {"src":"user"} -->
- 상태: 대기

해당 줄은 무엇을 위해서 남겨둔 줄이야?

### 답 · claude · 2026-10-21 16:26

질문에 가리킨 줄이 없습니다. 범위가 "전체"로만 들어와서 어느 줄인지 확정하지 못했습니다. 이 흐름 점 파일(`workbench/flow/spiral-frame-ana.yaml`)에서 줄 목적이 애매할 만한 곳을 아래에 추려 적습니다.

- **`result: solved`와 `progress: done`(3–4행)**: 이 점이 닫혔다는 표시입니다. 근거는 17행의 `result:`에 있는 일치 결과입니다. 삼각 120° 나선, DM 나선, cone 나선이 supercell LSWT와 에너지, 띠, $S^{ab}(\mathbf q,\omega)$까지 반올림 오차 수준으로 맞습니다. 비공약 J1-J2 사슬은 해석적 pitch와 $\omega(\mathbf k)$에 $10^{-14}$까지 맞습니다.
- **`setup:`의 "allowed only when every term commutes with the axis rotation"(16행)**: 회전틀 방법이 성립하는 조건을 남긴 줄입니다. 모든 항이 축 회전과 교환해야 하므로 axial XXZ, axial DM, 축 방향 장만 허용됩니다. 이 조건이 깨지면 `SpiralSymmetryError`가 납니다(17행 끝). 즉 17행 끝은 방법의 적용 한계를 코드가 강제한다는 기록입니다.
- **`evidence:`의 `checks: []`(22, 27, 31행)**: 근거 세 건(Toth–Lake `tothLinear2015`, D34 표, `lswt-classical-order` 노트의 "Single-Q Spirals in a Rotating Frame" 절)에 아직 붙은 검증 항목이 없다는 뜻입니다. 이 파일만으로는 왜 비어 있는지 알 수 없습니다. 나중에 사용자의 물리·수학 검토 결과를 채울 자리일 가능성이 높지만, 이는 제 추론입니다. 비어 있는 것은 AGENTS.md의 원칙대로 코드 수치 일치가 사용자 검토를 대신하지 않는다는 점과 맞습니다.
- **`sources:`의 `commit 06f2985`와 2026-10-01 제안서(33–35행)**: 구현 시점과 원 제안서를 추적하기 위한 출처 줄입니다.

어느 줄인지 알려 주시면 그 줄만 따로 근거를 찾아 답하겠습니다. 줄 번호나 문구를 알려 주시면 됩니다.

- 상태: 답함
