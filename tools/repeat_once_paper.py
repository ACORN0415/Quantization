"""Static final figure and Korean manuscript; no inference or new experiments."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/repeat_once_final'


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=json.loads((OUT/'summary.json').read_text())
    e=json.loads((ROOT/'src/experiments/mechanism_fr_results/evidence.json').read_text())
    equal=sum(c['status']=='valid_equal' for c in s['comparisons'])
    different=sum(c['status']=='valid_different' for c in s['comparisons'])
    gates=sum(c['status']=='valid_equal' for c in s['gates'])
    fig,(ax,side)=plt.subplots(1,2,figsize=(10.8,4.8),gridspec_kw={'width_ratios':[1.75,1]})
    y=list(range(12,0,-1))
    ax.axvline(0,color='#a0a0a0',lw=1)
    ax.scatter(e['prompt_differences'],y,c=['#ab4141' if v<0 else '#276b95' for v in e['prompt_differences']],s=42)
    lo,hi=e['ci95'];mean=e['mean']
    ax.errorbar(mean,0,xerr=[[mean-lo],[hi-mean]],fmt='D',color='#173f58',capsize=5,markersize=7)
    ax.set_yticks(y+[0], [f'P{i:02d}' for i in range(12)]+['Mean / 95% CI'])
    ax.set_xlabel('Late-frame MUSIQ: fixed-chunk repeat minus rolling')
    ax.set_title('A. Existing controlled quality result',loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.18)
    ax.set_ylim(-1,13)
    side.axis('off')
    side.set_title('B. Final implementation check',loc='left',fontweight='bold')
    side.text(0,.90,f'Repeated-path reproductions: {gates}/2 exact\n\n'
              f'Once vs archived repeat: {equal}/24 exact\n'
              f'Valid bit mismatches: {different}\n'
              f'Other / unfinished: {24-equal-different}\n\n'
              f'New worker attempts: {len(s["budget"]["attempts"])}/26\n\n'
              'Direct codec traces: new runs only.\n'
              'Historical repeat call traces unavailable.\n\n'
              'Exactness is implementation evidence,\n'
              'not a new quality experiment.',va='top',fontsize=11,linespacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT/'main_figure.pdf',bbox_inches='tight')
    fig.savefig(OUT/'main_figure.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    if s['conclusion']=='all_24_exact':
        finding=('마지막 구현 검사에서 두 반복 경로 재현이 기존 출력과 비트 일치했고, '
                 '단회 경로 24개도 각각 대응하는 기존 반복 경로의 최종 잠재 텐서와 비트 일치했다. '
                 '이는 시험한 구현·환경·입력에서 반복 연산을 생략해도 최종 잠재 텐서가 같았다는 검증이다.')
    elif s['conclusion']=='exact_hypothesis_failed':
        finding=(f'마지막 검사에서 {different}개의 유효 대응 비교가 비트 불일치했다. '
                 '따라서 단회와 반복 경로가 모든 시험 사례에서 정확히 같다는 가설은 지지되지 않았다. '
                 '비트 차이 자체를 지각적 품질 차이라고 해석하지 않는다.')
    else:
        finding=('마지막 구현 검사는 게이트 실패 또는 미완료를 포함하므로 24쌍 전체 일치를 주장하지 않는다. '
                 '이 제한은 기존 그룹 규칙 품질 개입의 결과를 소급해서 무효화하지 않는다.')
    text=f'''# 스트리밍 영상 KV 캐시에서 반복 횟수와 그룹 규칙을 구분한 실증 분석

최종 범위의 한국어 논문 초안. 기존 초안과 동결된 실험 기록은 보존한다.

## 초록

스트리밍 영상 생성의 저비트 KV 캐시 품질은 비트폭뿐 아니라 저장 규칙에 영향을 받는다.
본 연구는 Wan2.1 기반 Self-Forcing의 INT4 캐시에서 물리 버퍼 기준 그룹을 사용하는
rolling과 청크별 그룹을 유지하며 반복 양자화하는 경로를 비교했다. 12개 프롬프트와
각 두 시드에서 후반 전체 원본 프레임 MUSIQ의 대응 차이는 평균 {mean:.4f}점,
95% 프롬프트 부트스트랩 구간 [{lo:.4f}, {hi:.4f}]였다. 반복 연산을 유지한 개입에서도
품질이 개선됐으므로 정책 차이를 반복 횟수만으로 설명할 수 없다. {finding}
본 연구는 새 압축 알고리즘, 모든 입력에 대한 오차 우위, 시간 품질 또는 같은 메모리
예산의 우위를 주장하지 않는다.

## 연구 질문과 기존 결과

초기 P1/P3 비교에서는 INT4/INT2에서 P3의 후반 표본 프레임 MUSIQ가 각각7.69/7.05점
높았다. 그러나 그룹 경계·현재 청크 저장·반복 횟수 등 여러 조건이 달랐다. 따라서 주된
후속 질문을 반복을 남겨둔 채 청크별 그룹 규칙을 적용해도 품질이 달라지는지로 좁혔다.
고정 그룹의 이상적 RTN 멱등성은 해석의 배경이며 신규 정리가 아니다. 실제 BF16 계산과
FP16 scale 저장에는 별도 검증이 필요하다. 그룹 재편 후 원본 대비 오차가 감소하는 반례도
있으므로 재편이 모든 입력에서 오차를 늘린다는 명제를 사용하지 않는다.

## 방법과 평가

주 품질 실험은 새12프롬프트×2시드×BF16/rolling/청크별 반복의72개 궤적이다.
주지표는 후반 원본 프레임249–500 전체의 MUSIQ이고, 시드를 프롬프트 안에서 평균한
12개 대응 차이를 분석했다. 캐시는 INT4, token group16, 문맥21잠재프레임이며,
42청크·126잠재프레임·501출력프레임을 생성했다. 코드는 byte-container와 BF16 attention을
사용하므로 실제 packed INT4 커널의 효율 실험으로 해석하지 않는다.

## 주결과

12개 프롬프트 중11개에서 청크별 반복 경로가 높고1개에서 낮았다. 대응 평균은
{mean:.4f}점, 양측 정확 부호 반전 검정 p={e['exact_two_sided_sign_flip_p']:.8f}였다.
검정에는 귀무가설 아래 대응 차이의 대칭성/교환 가능성이 필요하고, 작은 편의 표본의
부트스트랩 구간은 근사적이다. 이는 청크별 그룹 규칙 개입의 결과이며 쓰기 경계의 scale
변화와 퇴거 위상 이동 각각의 순수 효과를 분리한 결과는 아니다.

![주 품질 결과와 마지막 구현 검증](main_figure.png)

그림1. 왼쪽은 기존 주 실험의 프롬프트별 대응 차이와 평균95% 구간이다. 오른쪽은
별도의 사후 구현 검증이다. 두 실험의 표본이나 결과를 합쳐 새 품질 검정으로 계산하지 않았다.

## 마지막 구현 검증

{finding}

새 반복2회는 p00/303과 p11/404로 고정했다. 두 재현이 출력 비트·호출·조건 검증을
통과한 경우에만 같은12프롬프트×2시드의 단회24개를 생성했다. 실제 양자화기 진입/반환에서
K/V·층·최초/재인코딩 횟수를 수집했다. 반복 입력은 실제 저장 상태의 복원 반환 객체임을
확인했고, 계측용 복원/양자화 호출은 추가하지 않았다. CPU 계측 ON/OFF 검사는 출력뿐
아니라 실제 codec 호출 수 불변도 확인했다. 과거24개 반복 실행에는 호출 로그가 없으므로
그 실행들의 직접 계측을 수행했다고 주장하지 않는다. 텐서 비교는 형상·dtype과 정규화한
원소 바이트의 직접 비교이며 파일 해시는 산출물 무결성에만 사용했다.

최종 판정은 `{s['conclusion']}`다. 유효 일치 {equal}, 유효 불일치 {different},
그 외 {24-equal-different}개다. 새 worker 시도는 {len(s['budget']['attempts'])}/26회이며,
유효 완료는 {s['completed_valid']}회다. 자세한 게이트·무효·미완료와 시간은 최종 보고서에 있다.

## 선행연구와 기여 범위

QVG는 이미 청크별 단회 압축으로 재압축 드리프트를 피하는 구현을 기술한다.
따라서 단회 저장이나 그 실패 모드의 최초 발견을 주장하지 않는다.
우리는 시험한 구현에서 반복 횟수와 그룹 규칙을 구분해 비교하고 전체 생성 결과와 연결한다.
특정 논문에 특정 비교가 없다는 사실을 연구 전체의 신규성으로 간주하지 않는다.
관련 원문: https://arxiv.org/html/2602.02958v5 .

## 한계와 종료 판단

MUSIQ는 프레임 품질이며 시간 일관성·객체 동일성·프롬프트 충실도 전체를 입증하지 않는다.
이 결과는 한 모델·한 구현·시험한 입력 범위에 한정한다. 동일 비트폭은 동일 메모리
예산이 아니며, 과거 fused 커널의 속도와 이 실험의 품질을 합쳐 제시하지 않는다.
후속 TT와 여러 압축 후보는 별도 개발 결과이며 본문의 주 품질 효과를 확대하는 근거로
사용하지 않는다. 마지막 검증의 성공 여부에 따라 표현을 위와 같이 제한하고 추가 후보
탐색을 종료한다. 투고처 수준이나 신규성의 충분성을 이 검증만으로 보장하지 않는다.

## 자료

- [기존 주 실험](../mechanism_fr_results/REPORT_KO.md)
- [주 실험의 수치·등록·산출물](../mechanism_fr_results/evidence.json)
- [마지막 검사 계획](../REPEAT_ONCE_FINAL.md)
- [최종 결과·시간·실행 범위](REPORT_KO.md)
- [24쌍 비교표](comparisons.csv)
'''
    (OUT/'PAPER_DRAFT_KO.md').write_text(text)


if __name__=='__main__':render()
