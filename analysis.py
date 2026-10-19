"""
옷 검색량 x 서울 기온 분석
Codyssey 'AI 데이터 분석: 데이터 기반 트렌드 분석' 미션

실행 방법:  python analysis.py
만들어지는 것:
    images/   그래프 PNG 9개
    outputs/  표(CSV)와 summary.txt (리포트에 옮겨 적을 숫자가 모여 있음)

분석 질문 (REPORT.md 와 같은 번호)
    Q1. 반팔·겉옷·패딩은 각각 언제 가장 많이 검색될까?
    Q2. 그때 기온은 몇 도일까? 옷마다 '사는 기온'이 있고 해마다 비슷할까?
    Q3. 더위·추위가 일찍/늦게 온 해에는 검색 시기도 따라 움직였을까? (날씨 vs 달력)
    Q4. 사람들은 날씨가 바뀐 뒤에 살까, 바뀌기 전에 미리 살까?
    Q5. 해마다 겉옷 관심이 시작·종료되는 날과 기온으로 본 계절 길이는 어떻게 변했을까?
    Q6. (활용) 패딩만 날씨와 연동된다면, 올해 패딩은 언제 가장 붐빌까?

사용한 시계열 기법: 이동평균, 월별 집계, 구간별 통계(정점일 기온),
                    변화율(연평균 전년 대비), 시차·상관 분석
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter, MaxNLocator

warnings.filterwarnings("ignore", message="Workbook contains no default style")

# ----------------------------------------------------------------------------
# 0. 설정값 (바꿔서 다시 돌려볼 수 있는 숫자들)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "outputs"
WEATHER_FILE = DATA_DIR / "seoul_asos_daily_2017_2026.csv"
SEARCH_FILE = DATA_DIR / "datalab_clothes_2016_2026.xlsx"

START = "2017-01-01"        # 기온 파일이 2017년부터라서 맞춤 (2016년의 튀는 값도 함께 제외됨)
SMOOTH_DAYS = 7             # 검색량 이동평균 일수
TEMP_SMOOTH_DAYS = 7        # Q2~Q4에서 쓰는 기온 이동평균 일수
SEASON_TEMP_SMOOTH_DAYS = 9  # Q5 기온 계절 구분에 쓰는 이동평균 일수
OUTLIER_WINDOW = 15         # 튀는 값 판단에 쓰는 주변 일수
OUTLIER_FACTOR = 3.0        # 주변 중앙값의 몇 배를 넘으면 튀는 값으로 볼지

# 검색 정점을 찾는 구간과 기온이 변하는 방향
#   warm = 기온이 오르는 시기 (기온이 기준 이상이 되는 첫날을 찾음)
#   cold = 기온이 내리는 시기 (기온이 기준 이하가 되는 첫날을 찾음)
EVENTS = [
    dict(name="반팔", col="반팔", win=((3, 1), (8, 31)), direction="warm", trigger_from=(2, 1)),
    dict(name="겉옷(봄)", col="겉옷", win=((2, 1), (4, 30)), direction="warm", trigger_from=(2, 1)),
    dict(name="겉옷(가을)", col="겉옷", win=((9, 1), (12, 31)), direction="cold", trigger_from=(9, 1)),
    dict(name="패딩", col="패딩", win=((9, 1), (12, 31)), direction="cold", trigger_from=(9, 1)),
]
SWEEP_HALF_WIDTH = 6        # 기준 기온을 정점일 기온 중앙값 기준 위아래 몇 도까지 바꿔볼지
MIN_VALID_YEARS = 6         # 기준 기온 하나를 평가할 때 필요한 최소 연도 수
PERM_B = 3000               # 순열 검정 반복 횟수
PERM_SEED = 0               # 순열 검정 난수 시드(재현용)
SMOOTH_CHECK = (3, 7, 14)   # 정점일 찾을 때 이동평균 길이 민감도 점검

# 그래프 색 (dataviz 스킬의 검증된 팔레트 앞 3칸: 세 색 모두 서로 구분되는 것을 확인함)
COLORS = {"반팔": "#eb6834", "겉옷": "#1baf7a", "패딩": "#2a78d6"}
INK = "#0b0b0b"         # 글자
INK_2 = "#52514e"       # 보조 글자·기온 선
SURFACE = "#fcfcfb"     # 그래프 바탕
GRID = "#e4e3df"

LOG_LINES = []


def say(text=""):
    """화면에 출력하고 summary.txt 에도 남긴다."""
    print(text)
    LOG_LINES.append(text)


# ----------------------------------------------------------------------------
# 1. 불러오기와 기본 정보 확인
# ----------------------------------------------------------------------------
def load_weather():
    """기상청 일 자료 CSV (서울). 한글 인코딩이 cp949 인 경우가 많아 차례로 시도."""
    last_error = None
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            raw = pd.read_csv(WEATHER_FILE, encoding=enc)
            break
        except UnicodeDecodeError as err:
            last_error = err
    else:
        raise last_error
    date_col = [c for c in raw.columns if "일시" in c][0]
    mean_col = [c for c in raw.columns if "평균기온" in c][0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    temp = raw.set_index(date_col)[mean_col].astype(float).sort_index()
    temp.name = "평균기온"
    say("[기온 파일] " + WEATHER_FILE.name)
    say(f"  기간 {temp.index.min().date()} ~ {temp.index.max().date()}, {len(temp)}행")
    say(f"  컬럼 {list(raw.columns)}")
    say("  결측치(컬럼별): " + ", ".join(f"{c}={int(raw[c].isna().sum())}" for c in raw.columns))
    say("  (이 분석은 평균기온만 사용하며, 최저·최고기온의 결측은 분석에 영향 없음)")
    return temp


def load_search():
    """네이버 데이터랩 엑셀. 위쪽 설명 줄을 건너뛰고 '날짜' 헤더 줄부터 읽는다."""
    sheet = openpyxl.load_workbook(SEARCH_FILE, data_only=True).worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "날짜")
    meta = {r[0]: r[1] for r in rows[:header_idx] if r and r[0]}
    header = rows[header_idx]
    names = [header[i] for i in range(1, len(header), 2) if header[i]]
    frames = {}
    for k, name in enumerate(names):
        date_i, val_i = 2 * k, 2 * k + 1
        pairs = [(r[date_i], float(r[val_i])) for r in rows[header_idx + 1:] if r[date_i] and r[val_i] is not None]
        s = pd.Series({pd.Timestamp(d): v for d, v in pairs}).sort_index()
        frames[name] = s
    raw = pd.DataFrame(frames)
    say("[검색량 파일] " + SEARCH_FILE.name)
    say(f"  조회 조건: 기간={meta.get('기간')}, 범위={meta.get('범위')}, 성별={meta.get('성별')}, 연령대={meta.get('연령대')}")
    say(f"  기간 {raw.index.min().date()} ~ {raw.index.max().date()}, {len(raw)}행, 컬럼 {list(raw.columns)}")
    say("  결측치(컬럼별): " + ", ".join(f"{c}={int(raw[c].isna().sum())}" for c in raw.columns))
    say("  세 묶음을 한 번에 조회해서 같은 지수 기준(가장 큰 값=100)을 공유함")
    return raw


# ----------------------------------------------------------------------------
# 2. 정제: 기간 맞추기, 빠진 날, 튀는 값
# ----------------------------------------------------------------------------
def clean_search(raw, last_day):
    s = raw.loc[START:last_day].copy()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    missing = full.difference(s.index)
    say(f"[정제] 분석 기간 {full.min().date()} ~ {full.max().date()} ({len(full)}일)")
    say(f"  빠진 날짜 {len(missing)}개" + (" -> 앞뒤 값으로 선형 보간" if len(missing) else ""))
    s = s.reindex(full).interpolate(limit_direction="both")

    cleaned = s.copy()
    records = []
    say(f"  튀는 값 기준: 하루 값이 주변 {OUTLIER_WINDOW}일 중앙값의 {OUTLIER_FACTOR:g}배를 넘으면 중앙값으로 대체")
    for col in s.columns:
        med = s[col].rolling(OUTLIER_WINDOW, center=True, min_periods=OUTLIER_WINDOW // 2 + 1).median()
        is_out = s[col] > OUTLIER_FACTOR * med
        for day in s.index[is_out]:
            records.append(dict(날짜=day.date(), 묶음=col, 원래값=round(s.at[day, col], 4), 대체값=round(med.at[day], 4)))
        cleaned.loc[is_out, col] = med[is_out]
        say(f"  {col}: 튀는 값 {int(is_out.sum())}개")
    outliers = pd.DataFrame(records, columns=["날짜", "묶음", "원래값", "대체값"])
    for _, r in outliers.iterrows():
        say(f"    - {r['날짜']} {r['묶음']}: {r['원래값']} -> {r['대체값']}")
    return s, cleaned, outliers


def smooth(df, days):
    return df.rolling(days, center=True, min_periods=days // 2 + 1).mean()


def doy(ts):
    """윤년이어도 같은 날짜가 같은 번호가 되도록 맞춘 '그해 몇 번째 날'."""
    d = ts.dayofyear
    return d - 1 if (ts.is_leap_year and ts.month > 2) else d


def md_label(day_number):
    """번호 -> '월-일' 글자 (평년 기준)"""
    return (pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(round(day_number)) - 1)).strftime("%m-%d")


MONTH_START_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


# ----------------------------------------------------------------------------
# 3. Q1·Q2: 정점일과 정점일 기온
# ----------------------------------------------------------------------------
def monthly_profile(cleaned):
    full_years = cleaned.loc[:f"{cleaned.index.max().year - 1}-12-31"]  # 끝난 해만 사용
    monthly = full_years.groupby(full_years.index.month).mean()
    normalized = monthly / monthly.max()
    return monthly, normalized, full_years.index.year.min(), full_years.index.year.max()


def find_peaks(sm, temp7, last_day):
    rows = []
    for ev in EVENTS:
        for year in range(2017, last_day.year + 1):
            start = pd.Timestamp(year, *ev["win"][0])
            end = pd.Timestamp(year, *ev["win"][1])
            if end > last_day:
                continue  # 아직 끝나지 않은 구간은 제외
            seg = sm.loc[start:end, ev["col"]]
            day = seg.idxmax()
            rows.append(dict(event=ev["name"], year=year, peak_date=day, peak_doy=doy(day),
                             peak_value=seg.max(), peak_temp=temp7.at[day]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Q3·Q4: 기온이 기준선에 닿은 날과 정점일 비교
# ----------------------------------------------------------------------------
def trigger_date(temp7, year, threshold, direction, from_md):
    seg = temp7.loc[pd.Timestamp(year, *from_md):pd.Timestamp(year, 12, 31)].dropna()
    hit = seg[seg >= threshold] if direction == "warm" else seg[seg <= threshold]
    return hit.index[0] if len(hit) else pd.NaT


def loo_mae(peak, trig):
    """한 해씩 빼고 나머지로 예측해서 맞혀보는 평균 오차(일).
    달력 모델: 나머지 해의 평균 정점 날짜를 그대로 예측
    날씨 모델: (기온이 기준에 닿은 날) + (나머지 해의 평균 시차) 로 예측"""
    n = len(peak)
    err_cal, err_wx = [], []
    for i in range(n):
        keep = np.arange(n) != i
        err_cal.append(abs(peak[i] - peak[keep].mean()))
        err_wx.append(abs(peak[i] - (trig[i] + (peak[keep] - trig[keep]).mean())))
    return float(np.mean(err_cal)), float(np.mean(err_wx))


def evaluate_threshold(peaks_ev, temp7, ev, threshold):
    """기준 기온 하나에 대해: 해마다 기온 도달일 -> 시차, 상관, 예측 오차"""
    rows = []
    for _, p in peaks_ev.iterrows():
        trig = trigger_date(temp7, int(p["year"]), threshold, ev["direction"], ev["trigger_from"])
        if pd.isna(trig):
            continue
        rows.append(dict(year=int(p["year"]), peak_doy=int(p["peak_doy"]), trigger_doy=doy(trig), trigger_date=trig))
    d = pd.DataFrame(rows)
    out = dict(threshold=threshold, n=len(d), mean_lag=np.nan, sd_lag=np.nan, r=np.nan,
               mae_calendar=np.nan, mae_weather=np.nan, ratio=np.nan)
    if len(d) >= MIN_VALID_YEARS:
        peak = d["peak_doy"].to_numpy(float)
        trig = d["trigger_doy"].to_numpy(float)
        lag = peak - trig
        out["mean_lag"], out["sd_lag"] = lag.mean(), lag.std(ddof=1)
        if peak.std() > 0 and trig.std() > 0:
            out["r"] = float(np.corrcoef(peak, trig)[0, 1])
        out["mae_calendar"], out["mae_weather"] = loo_mae(peak, trig)
        out["ratio"] = out["mae_weather"] / out["mae_calendar"] if out["mae_calendar"] > 0 else np.nan
    return out, d


def trigger_matrix(peaks_ev, temp7, ev, thresholds):
    """해(행) x 기준 기온(열) 표: 기온이 기준에 처음 닿은 날(연중 일수). 모든 해에 값이 있는 기준만 남긴다."""
    pe = peaks_ev.sort_values("year").reset_index(drop=True)
    cols = {}
    for t in thresholds:
        _, d = evaluate_threshold(pe, temp7, ev, t)
        cols[t] = d.set_index("year")["trigger_doy"]
    mat = pd.DataFrame(cols).reindex(pe["year"]).dropna(axis=1)
    return pe["peak_doy"].to_numpy(float), mat


def loo_errors(peak, trig):
    """한 해씩 빼고 맞힌 평균 오차 (달력, 날씨) - loo_mae 와 같은 계산을 한 번에 처리"""
    n = len(peak)
    lag = peak - trig
    mean_lag = (lag.sum() - lag) / (n - 1)
    err_wx = np.abs(peak - (trig + mean_lag))
    err_cal = np.abs(peak - (peak.sum() - peak) / (n - 1))
    return float(err_cal.mean()), float(err_wx.mean())


def nested_validation(peak, mat):
    """공정한 비교: 기준 기온도 '빼 놓은 해를 제외한 해'에서만 골라서 그 해를 맞혀 본다.
    (기준을 전체 해에서 고르면 정답을 보고 고른 셈이라 날씨 모델이 유리해진다)"""
    n, k = len(peak), mat.shape[1]
    trig = mat.to_numpy(float)
    err_cal, err_wx, chosen = [], [], []
    for i in range(n):
        tr = np.arange(n) != i
        train_err = []
        for j in range(k):
            lag = peak[tr] - trig[tr, j]
            train_err.append(np.abs(peak[tr] - (trig[tr, j] + lag.mean())).mean())
        j = int(np.argmin(train_err))
        lag = peak[tr] - trig[tr, j]
        err_wx.append(abs(peak[i] - (trig[i, j] + lag.mean())))
        err_cal.append(abs(peak[i] - peak[tr].mean()))
        chosen.append(int(mat.columns[j]))
    return float(np.mean(err_cal)), float(np.mean(err_wx)), chosen


def permutation_test(peak, mat, n_iter=PERM_B, seed=PERM_SEED):
    """우연일 확률: 해마다 검색 정점일을 무작위로 섞어도(날씨와 무관해도),
    모든 기준 중 가장 좋은 기준에서 지금만큼 좋은(오차비가 낮은) 결과가 나올 확률"""
    trig = mat.to_numpy(float)

    def best_ratio(pv):
        return min(loo_errors(pv, trig[:, j])[1] / loo_errors(pv, trig[:, j])[0] for j in range(trig.shape[1]))

    observed = best_ratio(peak)
    rng = np.random.default_rng(seed)
    hits = sum(best_ratio(rng.permutation(peak)) <= observed for _ in range(n_iter))
    return observed, (hits + 1) / (n_iter + 1)


# ----------------------------------------------------------------------------
# 5. Q5: 겉옷 관심의 시작·종료와 기온으로 본 계절 길이
# ----------------------------------------------------------------------------
def season_table(sm, temp, last_day):
    """해마다
       - 겉옷 봄: 봄 정점 이후 관심이 '바닥과 정점의 중간'까지 내려온 첫날 (관심이 꺼지는 날)
       - 겉옷 가을: 가을 정점 이전에 관심이 '바닥과 정점의 중간' 위로 올라선 첫날 (관심이 시작되는 날)
       - 기온 계절 (단순화한 기후 기준, 9일 이동평균 기온):
           봄 시작   = 1~6월 중 5도 미만이던 마지막 날 + 1
           여름 시작 = 1~7월 중 20도 미만이던 마지막 날 + 1
           가을 시작 = 7월 이후 20도 이상이던 마지막 날 + 1
           겨울 시작 = 9월 이후 5도 이상이던 마지막 날 + 1"""
    tsm = temp.rolling(SEASON_TEMP_SMOOTH_DAYS, center=True, min_periods=SEASON_TEMP_SMOOTH_DAYS // 2 + 1).mean()
    x = sm["겉옷"]
    rows = []
    for year in range(2017, last_day.year + 1):
        r = dict(year=year)
        day = lambda m, d: pd.Timestamp(year, m, d)
        # 겉옷 봄
        if day(6, 30) <= last_day:
            seg = x.loc[day(2, 1):day(4, 30)]
            p, top = seg.idxmax(), seg.max()
            after = x.loc[p:day(6, 30)]
            half = after.min() + 0.5 * (top - after.min())
            r["겉옷_봄_관심꺼짐"] = after[after <= half].index[0]
        # 겉옷 가을
        if day(12, 31) <= last_day:
            seg = x.loc[day(9, 1):day(12, 31)]
            p, top = seg.idxmax(), seg.max()
            before = x.loc[day(8, 1):p]
            half = before.min() + 0.5 * (top - before.min())
            r["겉옷_가을_관심시작"] = before[before <= half].index[-1] + pd.Timedelta(days=1)
        # 기온 계절
        t = tsm.loc[day(1, 1):day(12, 31)]
        if day(7, 31) <= last_day:
            r["기온_봄시작"] = t.loc[day(1, 1):day(6, 30)][lambda z: z < 5].index[-1] + pd.Timedelta(days=1)
            r["기온_여름시작"] = t.loc[day(1, 1):day(7, 31)][lambda z: z < 20].index[-1] + pd.Timedelta(days=1)
        if day(12, 31) <= last_day:
            r["기온_가을시작"] = t.loc[day(7, 1):][lambda z: z >= 20].index[-1] + pd.Timedelta(days=1)
            r["기온_겨울시작"] = t.loc[day(9, 1):][lambda z: z >= 5].index[-1] + pd.Timedelta(days=1)
        rows.append(r)
    tbl = pd.DataFrame(rows).set_index("year")
    for c in tbl.columns:
        tbl[c] = pd.to_datetime(tbl[c])
    tbl["기온_봄길이"] = (tbl["기온_여름시작"] - tbl["기온_봄시작"]).dt.days
    tbl["기온_여름길이"] = (tbl["기온_가을시작"] - tbl["기온_여름시작"]).dt.days
    tbl["기온_가을길이"] = (tbl["기온_겨울시작"] - tbl["기온_가을시작"]).dt.days
    tbl["겉옷_안찾는기간"] = (tbl["겉옷_가을_관심시작"] - tbl["겉옷_봄_관심꺼짐"]).dt.days
    return tbl


def trend_per_year(series):
    s = series.dropna()
    if len(s) < 4:
        return np.nan
    return float(np.polyfit(s.index.to_numpy(float), s.to_numpy(float), 1)[0])


# ----------------------------------------------------------------------------
# 6. 그래프
# ----------------------------------------------------------------------------
def set_korean_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    else:
        try:
            import koreanize_matplotlib  # noqa: F401  (설치되어 있으면 한글 폰트를 자동 설정)
        except ImportError:
            print("경고: 한글 글꼴을 찾지 못했습니다. 그래프 글자가 네모로 보이면 'pip install koreanize-matplotlib' 후 다시 실행하세요.")
    plt.rcParams["axes.unicode_minus"] = False


def setup_style():
    set_korean_font()
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def date_axis(ax, which="y"):
    fmt = FuncFormatter(lambda v, _: md_label(v))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(fmt)
    (ax.yaxis if which == "y" else ax.xaxis).set_major_locator(MaxNLocator(5, integer=True))


def save(fig, name):
    path = IMG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    say(f"  저장: images/{name}")


def headline(fig, title, sub=None):
    """그래프 맨 위: 굵은 글씨 = 결론(제목), 아래 보통 글씨 = 읽는 법"""
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 0.985 - 0.055, sub, ha="left", va="top", fontsize=10, color=INK_2, linespacing=1.5)


# ----------------------------------------------------------------------------
# Q6. 올해 패딩 타이밍: 기온이 기준에 닿은 날을 0으로 맞춰 본 검색 흐름 + 올해 시나리오
# ----------------------------------------------------------------------------
def padding_event_study(sm, d, half=40):
    """해마다 '기온이 기준에 닿은 날'을 0으로 맞추고, 그 해 정점 대비 검색 비율(0~1)을 나란히 놓는다."""
    cols = {}
    for _, r in d.iterrows():
        c = pd.Timestamp(r["trigger_date"])
        year = int(r["year"])
        top = sm["패딩"].loc[pd.Timestamp(year, 9, 1):pd.Timestamp(year, 12, 31)].max()
        seg = sm["패딩"].loc[c - pd.Timedelta(days=half):c + pd.Timedelta(days=half)] / top
        seg.index = (seg.index - c).days
        cols[year] = seg
    return pd.DataFrame(cols)


def fig_event_study(es, marks):
    """Q6 근거 그래프: 기온이 10도 아래로 내려간 날을 0으로 맞추면 패딩 검색이 어떻게 오르는가"""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    col = COLORS["패딩"]
    for y in es.columns:
        ax.plot(es.index, es[y], color=INK_2, lw=1, alpha=0.35)
    mean = es.mean(axis=1)
    ax.plot(mean.index, mean.values, color=col, lw=3, label="9년 평균", zorder=4)
    ax.axvline(0, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(-1.0, 1.15, "기온이 10도 아래로 처음 내려간 날 (0일)", fontsize=9.5, color=INK, va="top", ha="right")
    ax.axhline(0.5, color=GRID, lw=1)
    for key, label, dy in [("half", "정점의 절반", 0.0), ("p80", "정점의 80%", 0.1)]:
        x = marks[key]
        y = mean.loc[x]
        ax.scatter([x], [y], s=90, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.annotate(f"{label}: {x:+d}일", (x, y), xytext=(x + (-1 if key == "half" else 1) * 3, y + dy),
                    ha="right" if key == "half" else "left", fontsize=10, color=INK,
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=1.5))
    ax.set_xlabel("기온이 10도 아래로 내려간 날로부터 (일)")
    ax.set_ylabel("그해 정점 대비 검색량")
    ax.set_ylim(0, 1.18)
    ax.set_xlim(es.index.min(), es.index.max())
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    headline(fig, f"Q6. 패딩 검색은 기온이 10도 아래로 내려가기 {abs(marks['half'])}일 전에 이미 정점의 절반, 내려간 뒤 {marks['p80']}일이면 80%예요",
             "회색 선 = 해마다(9개) 한 줄, 파란 굵은 선 = 평균. 모든 해를 '기온이 10도 아래로 처음 내려간 날'에 맞춰 겹쳤어요.\n"
             "세로축은 그해 정점을 1로 본 검색량이에요. 선이 0일 전에 이미 올라 있으면 '10도에 닿기 전부터 슬슬 찾기 시작한다'는 뜻이에요.\n"
             "해마다 정점일은 닿은 날 + 평균 9일(편차 ±8일)이라 평균 선은 완만한 언덕처럼 보여요. 붐비는 시기는 '닿은 날 전후 며칠~2주'로 읽는 게 안전해요.")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "08_padding_event_study.png")


def fig_forecast(d, scen, fc):
    """Q6 결과 그래프: 과거 9년의 '닿은 날 -> 정점일'과 올해 시나리오"""
    fig, ax = plt.subplots(figsize=(11, 7.8))
    col = COLORS["패딩"]
    d = d.sort_values("year").reset_index(drop=True)
    rows = [(f"'{str(int(r['year']))[2:]}", r["trigger_doy"], r["peak_doy"], False, 0) for _, r in d.iterrows()]
    rows += [(label, t, p, True, band) for label, t, p, band in scen]
    for i, (label, t, p, fut, band) in enumerate(rows):
        if fut:
            ax.plot([t, p], [i, i], color=col, lw=2, alpha=0.5, zorder=2)
            if band:
                ax.barh(i, 2 * band, left=p - band, height=0.5, color=col, alpha=0.15, zorder=1)
        else:
            ax.plot([t, p], [i, i], color=INK_2, lw=1.8, alpha=0.5, zorder=2)
        ax.scatter([t], [i], s=75, facecolor=SURFACE, edgecolor=col if fut else INK_2, linewidth=1.8, zorder=3)
        ax.scatter([p], [i], s=75, color=col, alpha=1 if not fut else 0.6, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.4, -0.8)
    ax.axhline(len(d) - 0.5, color=GRID, lw=1)
    today = doy(fc["today"])
    ax.axvline(today, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(today + 0.8, len(rows) - 0.6, f"데이터 마지막 날 ({fc['today']:%m-%d})", fontsize=9.5, va="bottom")
    ax.set_xlim(285, 350)
    date_axis(ax, "x")
    ax.scatter([], [], s=75, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.8, label="기온이 10도 아래로 처음 내려간 날")
    ax.scatter([], [], s=75, color=col, label="검색이 가장 많았던 날(올해는 예상)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9.5, frameon=False)
    if fc["cond"]:
        title = (f"Q6. 올해는 요즘 기온이 과거 9년보다 높아서, 패딩이 가장 붐비는 때는 평년({fc['mid_peak']})보다 늦은 {fc['cond_peak']} 전후로 예상돼요")
        tip = (f"위 9줄 = 과거 기록, 아래 4줄 = 올해 예상이에요. 마지막 줄은 '요즘 기온이 높은 해일수록 10도에 늦게 닿았다'는 관계(r={fc['cond']['r']:.2f})로 계산했어요.\n"
               f"앞의 3줄은 오늘 기온을 안 쓰고 과거 9년 중 가장 이른 날·가운데·가장 늦은 날로 가정한 경우예요. 정점 예상 = 닿은 날 + 평균 {fc['mean_lag']:.0f}일, 연한 띠 = 예상이 빗나갈 수 있는 폭이에요.\n"
               f"올해 최근 7일 평균기온은 {fc['cond']['xnow']:.1f}도로 과거 9년 같은 날(평균 {fc['cond']['xmean']:.1f}도)보다 높고, 과거 최고({fc['cond']['xmax']:.1f}도)와 비슷하거나 넘어서 신뢰는 낮은 편이에요.")
    else:
        title = "Q6. 올해는 기온이 10도에 닿은 날 + 평균 9일이 패딩이 가장 붐비는 때예요"
        tip = "위 9줄 = 과거 기록, 아래 줄 = 올해예요. 연한 띠 = 예상이 빗나갈 수 있는 폭이에요."
    headline(fig, title, tip)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save(fig, "09_padding_2026_forecast.png")



def fig_overview(cleaned, sm):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True)
    for ax, col in zip(axes, ["반팔", "겉옷", "패딩"]):
        ax.plot(cleaned.index, cleaned[col], color=COLORS[col], lw=0.8, alpha=0.35)
        ax.plot(sm.index, sm[col], color=COLORS[col], lw=2)
        ax.set_title(col)
        ax.set_ylabel("검색 지수")
        ax.set_ylim(0, cleaned[col].max() * 1.1)
    headline(fig, "세 옷 모두 해마다 같은 계절에 검색이 솟고, 2024년 이후 봉우리는 낮아졌어요",
             f"연한 선은 하루 값, 진한 선은 {SMOOTH_DAYS}일 이동평균이에요. 패딩 수치가 훨씬 커서 세로축 범위는 패널마다 달라요.")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "01_overview.png")


def fig_monthly(normalized, y0, y1):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    months = np.arange(1, 13)
    for col in ["반팔", "겉옷", "패딩"]:
        ax.plot(months, normalized[col].to_numpy(), color=COLORS[col], marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=col)
        top = int(normalized[col].idxmax())
        ax.annotate(f"{col} {top}월", (top, 1.0), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=10)
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}월" for m in months])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("그 옷의 연중 최고 월을 1로 맞춘 비율")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    headline(fig, "Q1. 반팔은 5월, 겉옷은 10월(3월에도 한 번 더), 패딩은 11월에 가장 많이 검색돼요",
             f"{y0}~{y1}년 월평균이에요. 옷마다 가장 높은 달을 1로 맞춰서 시기만 비교할 수 있게 했어요.")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "02_monthly_pattern.png")


def fig_peaks(peaks):
    names = [ev["name"] for ev in EVENTS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for i, name in enumerate(names):
        d = peaks[peaks["event"] == name]
        col = COLORS[[e["col"] for e in EVENTS if e["name"] == name][0]]
        for ax, key in ((a1, "peak_doy"), (a2, "peak_temp")):
            ax.scatter(d[key], np.full(len(d), i), s=70, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            med = d[key].median()
            ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
            ax.text(med, i - 0.4, md_label(med) if key == "peak_doy" else f"{med:.1f}℃",
                    ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels(names)
    a1.set_ylim(len(names) - 0.4, -0.9)
    a1.set_xticks([d for d in MONTH_START_DOY if a1.get_xlim()[0] - 15 < d < a1.get_xlim()[1] + 15])
    a1.set_xticklabels([f"{MONTH_START_DOY.index(d) + 1}월" for d in a1.get_xticks()])
    a1.set_title("검색이 가장 많았던 날")
    a2.set_title("그날의 서울 기온 (7일 평균)")
    a2.set_xlabel("℃")
    headline(fig, "Q1·Q2. 옷마다 '많이 찾는 때'와 '그때의 기온'이 대체로 정해져 있어요",
             "점 1개가 1년이에요. 검은 막대와 숫자는 중앙값이에요. 기온은 반팔 약 20도, 겉옷 봄 약 8도·가을 약 13도, 패딩 약 4도예요.\n"
             "점이 넓게 퍼질수록 해마다 들쭉날쭉하다는 뜻이에요(패딩 기온이 가장 넓어요).")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "03_peak_dates_and_temps.png")


def fig_weather_vs_calendar(detail, best_detail, bests, robust):
    """Q3 핵심 그래프: 정점일을 맞히는 오차 (달력 vs 날씨). 기준 기온을 훈련 해에서 고르는 공정한 값을 강조"""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    w = 0.26
    x = np.arange(len(EVENTS))
    ymax = 0
    for i, ev in enumerate(EVENTS):
        col = COLORS[ev["col"]]
        rb = robust[ev["name"]]
        binfo = best_detail[ev["name"]][0]
        vals = [rb["nested_cal"], binfo["mae_weather"], rb["nested_wx"]]
        styles = [dict(color="#b9b8b3"), dict(color=col, alpha=0.35), dict(color=col)]
        for k, (v, st) in enumerate(zip(vals, styles)):
            ax.bar(i + (k - 1) * w, v, width=w * 0.92, zorder=3, **st)
            ax.text(i + (k - 1) * w, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=INK,
                    fontweight="bold" if k == 2 else "normal")
        ymax = max(ymax, max(vals))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ev['name']}\n우연일 확률 p={robust[ev['name']]['p']:.3f}\n(전체에서 고른 최적 기준 {bests[ev['name']]}도)"
                        for ev in EVENTS], fontsize=10.5)
    ax.set_ylabel("정점일 예측 오차 (일, 낮을수록 정확)")
    ax.set_ylim(0, ymax * 1.32)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color="#b9b8b3", label="달력으로 예측 (해마다 평균 날짜)"),
                       Patch(color=INK_2, alpha=0.35, label="날씨로 예측: 전체 해에서 고른 최적 기준 (정답을 보고 고른 값이라 낙관적)"),
                       Patch(color=INK_2, label="날씨로 예측: 기준도 훈련 해에서 고름 (공정한 비교)")],
              loc="upper right", fontsize=9.5)
    headline(fig, "Q3. 공정하게 비교하면 날씨가 달력을 뚜렷하게 이기는 건 패딩뿐이에요",
             "한 해를 빼놓고 나머지 해로 그 해의 정점일을 맞혀 본 평균 오차예요. 가장 믿을 만한 비교는 회색과 진한 색 막대예요(연한 막대는 참고용).\n"
             "p = 날씨와 무관해도 기준 13개 중 이만큼 좋은 기준이 나올 확률(순열 검정 3,000회)이에요. 낮을수록 우연일 가능성이 적어요.\n"
             "겉옷(봄)은 최적 기준에선 날씨가 이겼지만 기준을 해마다 새로 고르면 달력보다 나빠져요. 즉 그 결과는 불안정해요.")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save(fig, "04_weather_vs_calendar.png")


def fig_year_by_year(best_detail, bests):
    """Q3·Q4 근거 그래프: 해마다 '기온이 닿은 날'과 '검색 정점일' 사이의 거리"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for ax, ev in zip(axes, EVENTS):
        info, d = best_detail[ev["name"]]
        col = COLORS[ev["col"]]
        d = d.sort_values("year").reset_index(drop=True)
        for i, r in d.iterrows():
            ax.plot([r["trigger_doy"], r["peak_doy"]], [i, i], color=INK_2, lw=1.8, alpha=0.55, zorder=2)
        ax.scatter(d["trigger_doy"], range(len(d)), s=80, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.8, zorder=3,
                   label="기온이 기준에 처음 닿은 날")
        ax.scatter(d["peak_doy"], range(len(d)), s=80, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=4,
                   label="검색이 가장 많았던 날")
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"'{str(int(y))[2:]}" for y in d["year"]])
        ax.set_ylim(len(d) - 0.4, -0.8)
        lo = min(d["trigger_doy"].min(), d["peak_doy"].min()) - 5
        hi = max(d["trigger_doy"].max(), d["peak_doy"].max()) + 5
        ax.set_xlim(lo, hi)
        date_axis(ax, "x")
        lag = info["mean_lag"]
        word = "뒤" if lag >= 0 else "앞"
        ax.set_title(f"{ev['name']}  (기준 {bests[ev['name']]}도)\n정점이 평균 {abs(lag):.1f}일 {word}, 상관 r={info['r']:.2f}", fontsize=11)
        if ev is EVENTS[0]:
            ax.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, -0.2), ncol=1)
    headline(fig, "Q4. 날씨 연동이 가장 뚜렷한 패딩은 기온이 10도 아래로 내려간 뒤 며칠 안에 정점을 찍어요",
             "해마다 한 줄이에요. 빈 동그라미 = 기온이 기준에 처음 닿은 날, 색 동그라미 = 검색이 가장 많았던 날. 색 동그라미가 오른쪽이면 날씨가 바뀐 '뒤'에 반응한 거예요.\n"
             "줄 길이가 해마다 비슷하고 두 동그라미가 같이 움직이면(상관 r이 클수록) 날씨를 따라간 거예요. 기준 기온은 전체 해에서 고른 최적값이라 낙관적이고,\n"
             "공정한 검증(그래프 4)을 통과한 건 패딩뿐이에요. 패딩의 평균 시차도 이동평균 길이에 따라 +1~+9일로 움직이고, 반팔·겉옷의 시차는 참고용이에요.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.84])
    save(fig, "05_year_by_year.png")


def fig_sweep(sweeps, tstar, bests):
    """Q3·Q4 점검: 기준 기온을 바꾸면 결론이 어떻게 달라지는가"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), sharey=True)
    ymax = max(s["ratio"].max() for s in sweeps.values()) * 1.2
    for ax, ev in zip(axes, EVENTS):
        s = sweeps[ev["name"]].dropna(subset=["ratio"])
        col = COLORS[ev["col"]]
        ax.axhspan(0, 1, color=col, alpha=0.10, zorder=0)
        ax.axhline(1, color=INK_2, lw=1)
        ax.plot(s["threshold"], s["ratio"], color=col, marker="o", markersize=6, markeredgecolor=SURFACE, zorder=3)
        ax.axvline(tstar[ev["name"]], color=INK_2, lw=1, ls=":")
        b = s[s["threshold"] == bests[ev["name"]]].iloc[0]
        ax.scatter([b["threshold"]], [b["ratio"]], s=140, color=col, edgecolor=INK, linewidth=1.8, zorder=4)
        ax.annotate(f"최적 {b['threshold']:.0f}도", (b["threshold"], b["ratio"]), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=10, color=INK, fontweight="bold")
        wins = int((s["ratio"] < 1).sum())
        ax.set_title(f"{ev['name']}\n날씨가 유리한 기준 {wins}/{len(s)}개", fontsize=11)
        ax.set_xlabel("기준 기온 (도)")
        ax.set_ylim(0.3, ymax)
        box = dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1.5)
        ax.text(0.03, 0.06, "날씨가 유리", transform=ax.transAxes, fontsize=9, color=INK_2, bbox=box, zorder=5)
        ax.text(0.03, 0.95, "달력이 유리", transform=ax.transAxes, fontsize=9, color=INK_2, bbox=box, zorder=5)
        ax.margins(x=0.08)
    axes[0].set_ylabel("날씨 오차 / 달력 오차")
    headline(fig, "점검. 날씨가 이기는지는 기준 기온을 몇 도로 잡느냐에 따라 달라져요",
             "세로축이 1보다 아래(색 칠한 부분)면 날씨가 더 잘 맞힌 거예요. 점선 = 그날의 기온 중앙값(그래프 4의 연한 막대 기준), 큰 점 = 가장 잘 맞은 기준.\n"
             "패딩은 9~10도에서만, 겉옷은 특정 기온에서만 날씨가 앞서요. 13개 중 가장 좋은 걸 고른 것이라 우연일 수 있어서 '참고용'으로 봐 주세요.")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save(fig, "06_threshold_sweep.png")


def fig_seasons(tbl):
    """Q5: 겉옷 검색에서 읽은 날짜와 기온에서 읽은 날짜를 나란히"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    years = tbl.index.to_numpy()

    def d2(col):
        return tbl[col].map(lambda t: doy(t) if pd.notna(t) else np.nan)

    def line(ax, values, color, label, ls="-"):
        ax.plot(years, values, color=color, ls=ls, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)

    def corr(a, b):
        both = pd.concat([a, b], axis=1).dropna()
        return float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]) if len(both) >= 4 else np.nan

    a, b, c = axes
    fall_a, fall_b = d2("겉옷_가을_관심시작"), d2("기온_가을시작")
    line(a, fall_a, COLORS["겉옷"], "겉옷 검색에서 읽은 날 (관심이 오르기 시작)")
    line(a, fall_b, INK_2, "기온에서 읽은 날 (가을 시작)", "--")
    gap = (fall_a - fall_b).dropna()
    a.set_title(f"가을: 두 선이 같이 움직여요\n상관 r={corr(fall_a, fall_b):.2f}, 평균 {gap.mean():+.0f}일 차이", fontsize=11)
    date_axis(a)

    spr_a, spr_b = d2("겉옷_봄_관심꺼짐"), d2("기온_여름시작")
    line(b, spr_a, COLORS["겉옷"], "겉옷 검색에서 읽은 날 (관심이 절반으로 줄어듦)")
    line(b, spr_b, INK_2, "기온에서 읽은 날 (여름 시작)", "--")
    b.set_title(f"봄: 따로 움직여요\n상관 r={corr(spr_a, spr_b):.2f}", fontsize=11)
    date_axis(b)

    line(c, tbl["겉옷_안찾는기간"], COLORS["겉옷"], "겉옷을 안 찾는 기간")
    line(c, tbl["기온_여름길이"], INK_2, "기온으로 본 여름 길이", "--")
    c.set_title(f"여름 길이: 겉옷 안 찾는 기간은 해마다 {trend_per_year(tbl['겉옷_안찾는기간']):+.1f}일씩 줄지만\n기온 여름은 {trend_per_year(tbl['기온_여름길이']):+.1f}일씩 변해요", fontsize=11)
    c.set_ylabel("일")
    for ax in axes:
        ax.set_xticks(years)
        ax.set_xticklabels([f"'{str(y)[2:]}" for y in years])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9, ncol=1)
    headline(fig, "Q5. 가을 겉옷 관심은 기온을 따라가지만, 겉옷을 안 찾는 기간이 줄어드는 건 기온만으로 설명이 안 돼요",
             "초록 = 겉옷 검색에서 읽은 날짜, 회색 점선 = 같은 해 기온에서 읽은 날짜예요. 두 선이 같이 움직이면 날씨를 따라간 거예요.\n"
             "기온 계절은 9일 평균기온이 5도·20도를 넘나드는 시점으로 단순화한 기준이고, 2026년은 아직 끝나지 않아 일부 항목이 빠져 있어요(해마다 9~10개 점뿐이라 참고용).")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save(fig, "07_seasons.png")


# ----------------------------------------------------------------------------
# 7. 실행
# ----------------------------------------------------------------------------
def main():
    IMG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    setup_style()

    say("=" * 70)
    say("1. 데이터 기본 정보")
    say("=" * 70)
    temp = load_weather()
    raw = load_search()
    last_day = min(temp.index.max(), raw.index.max())

    say("")
    say("=" * 70)
    say("2. 정제 (결측치·이상치)")
    say("=" * 70)
    raw_in_range, cleaned, outliers = clean_search(raw, last_day)
    outliers.to_csv(OUT_DIR / "outliers.csv", index=False, encoding="utf-8-sig")
    temp = temp.loc[START:last_day]
    if temp.isna().any():
        say(f"  평균기온 결측 {int(temp.isna().sum())}개 -> 선형 보간")
        temp = temp.interpolate()
    sm = smooth(cleaned, SMOOTH_DAYS)
    temp7 = temp.rolling(TEMP_SMOOTH_DAYS, center=True, min_periods=TEMP_SMOOTH_DAYS // 2 + 1).mean()

    # ---- Q1
    say("")
    say("=" * 70)
    say("3. Q1 반팔·겉옷·패딩은 언제 가장 많이 검색될까?")
    say("=" * 70)
    monthly, normalized, y0, y1 = monthly_profile(cleaned)
    say(f"[월별 평균 검색 지수, {y0}~{y1}년]")
    say(monthly.round(3).to_string())
    for col in monthly.columns:
        say(f"  {col}: 가장 높은 달 {int(monthly[col].idxmax())}월, 가장 낮은 달 {int(monthly[col].idxmin())}월")
    monthly.to_csv(OUT_DIR / "q1_monthly_mean.csv", encoding="utf-8-sig")

    peaks = find_peaks(sm, temp7, last_day)
    peaks_out = peaks.copy()
    peaks_out["peak_date"] = peaks_out["peak_date"].dt.date
    peaks_out.round(3).to_csv(OUT_DIR / "q1_q2_peaks.csv", index=False, encoding="utf-8-sig")
    say("")
    say("[해마다 검색이 가장 많았던 날 (이동평균 기준)]")
    for ev in EVENTS:
        d = peaks[peaks["event"] == ev["name"]]
        dates = ", ".join(f"{int(r.year) % 100:02d}년 {r.peak_date:%m-%d}" for r in d.itertuples())
        say(f"  {ev['name']}: {dates}")
        say(f"     중앙값 {md_label(d['peak_doy'].median())}, 가장 이른 {md_label(d['peak_doy'].min())}, 가장 늦은 {md_label(d['peak_doy'].max())}")

    # ---- Q2
    say("")
    say("=" * 70)
    say("4. Q2 그날의 기온은 몇 도일까?")
    say("=" * 70)
    tstar = {}
    for ev in EVENTS:
        d = peaks[peaks["event"] == ev["name"]]["peak_temp"]
        tstar[ev["name"]] = int(round(d.median()))
        say(f"  {ev['name']}: 정점일 기온 중앙값 {d.median():.1f}℃ (평균 {d.mean():.1f}, 표준편차 {d.std():.1f}, 범위 {d.min():.1f}~{d.max():.1f}, n={len(d)})")

    # ---- Q3, Q4
    say("")
    say("=" * 70)
    say("5. Q3·Q4 날씨 vs 달력, 반응 vs 선행")
    say("=" * 70)
    say("  기준 기온 = 위 Q2의 정점일 기온 중앙값(정수로 반올림). 해마다 기온(7일 평균)이 처음 그 기준에 닿은 날을 찾는다.")
    say("  시차 = 검색 정점일 - 기온이 기준에 닿은 날 (양수: 정점이 나중, 음수: 정점이 먼저)")
    say("  예측 오차는 한 해씩 빼고 나머지 해로 맞혀본 평균 오차(일). 달력 모델은 '평균 날짜', 날씨 모델은 '기온 도달일 + 평균 시차'.")
    detail, sweeps, sweep_rows = {}, {}, []
    for ev in EVENTS:
        pe = peaks[peaks["event"] == ev["name"]].sort_values("year")
        info, d = evaluate_threshold(pe, temp7, ev, tstar[ev["name"]])
        detail[ev["name"]] = (info, d)
        say(f"  {ev['name']}: 기준 {info['threshold']}℃ (n={info['n']}) 평균 시차 {info['mean_lag']:+.1f}일(±{info['sd_lag']:.1f}), "
            f"상관 r={info['r']:.2f}, 예측 오차 달력 {info['mae_calendar']:.1f}일 vs 날씨 {info['mae_weather']:.1f}일")
        rows = []
        for t in range(tstar[ev["name"]] - SWEEP_HALF_WIDTH, tstar[ev["name"]] + SWEEP_HALF_WIDTH + 1):
            o, _ = evaluate_threshold(pe, temp7, ev, t)
            o["event"] = ev["name"]
            rows.append(o)
        sweeps[ev["name"]] = pd.DataFrame(rows)
        sweep_rows += rows
    pd.DataFrame(sweep_rows).round(3).to_csv(OUT_DIR / "q3_q4_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    say("")
    say("[기준 기온을 바꿔 본 결과 요약 (n이 부족한 기준은 제외)]")
    for ev in EVENTS:
        s = sweeps[ev["name"]].dropna(subset=["ratio"])
        wins = int((s["ratio"] < 1).sum())
        say(f"  {ev['name']}: 시험한 기준 {len(s)}개 중 날씨 모델이 달력 모델보다 오차가 작은 기준 {wins}개, "
            f"평균 시차는 {s['mean_lag'].min():+.0f}일 ~ {s['mean_lag'].max():+.0f}일 사이에서 움직임")
        best = s.loc[s["ratio"].idxmin()]
        say(f"     가장 잘 맞은 기준 {best['threshold']:.0f}℃: 상관 r={best['r']:.2f}, "
            f"오차 달력 {best['mae_calendar']:.1f}일 vs 날씨 {best['mae_weather']:.1f}일, 평균 시차 {best['mean_lag']:+.1f}일")
    say("  (주의: 기준을 여러 개 시험해서 가장 좋은 것을 고르면 우연히 좋아 보일 수 있음. 해마다 표본도 9~10개뿐)")
    say("  (자세한 표: outputs/q3_q4_threshold_sweep.csv)")

    bests, best_detail = {}, {}
    for ev in EVENTS:
        sv = sweeps[ev["name"]].dropna(subset=["ratio"])
        bests[ev["name"]] = int(sv.loc[sv["ratio"].idxmin(), "threshold"])
        pe = peaks[peaks["event"] == ev["name"]].sort_values("year")
        best_detail[ev["name"]] = evaluate_threshold(pe, temp7, ev, bests[ev["name"]])

    # ---- Q3 검증 보강: 공정한 비교(중첩 검증), 우연 확률(순열 검정), 이동평균 길이 민감도
    say("")
    say("[검증 보강 1] 기준 기온까지 훈련 해에서 고르는 공정한 비교 + 우연일 확률(순열 검정)")
    say("  - 공정한 비교: 빼 놓은 해를 제외한 해에서만 기준 기온을 골라 그 해의 정점일을 맞힘")
    say(f"  - 우연일 확률 p: 정점일을 무작위로 섞어도 모든 기준 중 최고가 지금만큼 좋을 확률 ({PERM_B}회 반복)")
    robust, rob_rows = {}, []
    for ev in EVENTS:
        pe = peaks[peaks["event"] == ev["name"]].sort_values("year")
        thr = range(tstar[ev["name"]] - SWEEP_HALF_WIDTH, tstar[ev["name"]] + SWEEP_HALF_WIDTH + 1)
        pk, mat = trigger_matrix(pe, temp7, ev, thr)
        n_cal, n_wx, chosen = nested_validation(pk, mat)
        obs, pval = permutation_test(pk, mat)
        picks = pd.Series(chosen).value_counts()
        robust[ev["name"]] = dict(nested_cal=n_cal, nested_wx=n_wx, ratio=n_wx / n_cal, p=pval, observed=obs,
                                  n_thresholds=mat.shape[1], chosen=chosen)
        say(f"  {ev['name']}: 오차 달력 {n_cal:.1f}일 vs 날씨(공정) {n_wx:.1f}일 (비 {n_wx / n_cal:.2f}), "
            f"우연일 확률 p={pval:.3f}, 해마다 고른 기준 {({int(k): int(v) for k, v in picks.sort_index().items()})}")
        rob_rows.append(dict(event=ev["name"], n_years=len(pk), n_thresholds=mat.shape[1],
                             mae_calendar=round(n_cal, 2), mae_weather_nested=round(n_wx, 2),
                             ratio_nested=round(n_wx / n_cal, 3), best_ratio_in_sample=round(obs, 3),
                             p_permutation=round(pval, 4), chosen_thresholds=" ".join(map(str, chosen))))
    pd.DataFrame(rob_rows).to_csv(OUT_DIR / "q3_robustness.csv", index=False, encoding="utf-8-sig")
    say("  (비 < 1 이면 날씨가 유리, > 1 이면 달력이 유리. 기준 후보의 중심을 전체 해의 정점일 기온 중앙값으로 잡은 점은 약한 정보 누수예요)")

    say("")
    say("[검증 보강 2] 정점일을 찾는 이동평균 길이를 바꿔도 결론이 유지될까? (가장 잘 맞은 기준 기준)")
    sm_rows = []
    for w in SMOOTH_CHECK:
        pkw = find_peaks(smooth(cleaned, w), temp7, last_day)
        for ev in EVENTS:
            pe = pkw[pkw["event"] == ev["name"]].sort_values("year")
            ts = int(round(pe["peak_temp"].median()))
            best = None
            for t in range(ts - SWEEP_HALF_WIDTH, ts + SWEEP_HALF_WIDTH + 1):
                o, _ = evaluate_threshold(pe, temp7, ev, t)
                if np.isfinite(o["ratio"]) and (best is None or o["ratio"] < best["ratio"]):
                    best = o
            sm_rows.append(dict(smooth_days=w, event=ev["name"], median_peak=md_label(pe["peak_doy"].median()),
                                median_peak_temp=round(pe["peak_temp"].median(), 1), best_threshold=best["threshold"],
                                r=round(best["r"], 2), mean_lag=round(best["mean_lag"], 1), ratio=round(best["ratio"], 2)))
    smdf = pd.DataFrame(sm_rows)
    smdf.to_csv(OUT_DIR / "robustness_smoothing.csv", index=False, encoding="utf-8-sig")
    say(smdf.to_string(index=False))

    # ---- Q5
    say("")
    say("=" * 70)
    say("6. Q5 계절의 시작과 길이")
    say("=" * 70)
    say("  겉옷 관심 날짜: 봄은 정점 이후 관심이 (바닥~정점)의 절반까지 내려온 첫날, 가을은 정점 이전에 절반 위로 올라선 첫날")
    say("  기온 계절은 단순화한 기준(9일 이동평균 기온이 5도·20도를 넘나드는 시점)이며 기상청 공식 정의와 다를 수 있음")
    seasons = season_table(sm, temp, last_day)
    seasons_out = seasons.copy()
    for c in seasons_out.columns:
        if str(seasons_out[c].dtype).startswith("datetime"):
            seasons_out[c] = seasons_out[c].dt.date
    seasons_out.to_csv(OUT_DIR / "q5_seasons.csv", encoding="utf-8-sig")
    show = seasons.copy()
    for c in show.columns:
        if str(show[c].dtype).startswith("datetime"):
            show[c] = show[c].dt.strftime("%m-%d")
    say(show.to_string())

    def corr_days(a, b):
        both = pd.concat([a, b], axis=1).dropna()
        if len(both) < 4:
            return np.nan, len(both)
        da = both.iloc[:, 0].map(doy).to_numpy(float)
        db = both.iloc[:, 1].map(doy).to_numpy(float)
        return float(np.corrcoef(da, db)[0, 1]), len(both)

    r1, n1 = corr_days(seasons["겉옷_봄_관심꺼짐"], seasons["기온_여름시작"])
    r2, n2 = corr_days(seasons["겉옷_가을_관심시작"], seasons["기온_가을시작"])
    say("")
    say(f"  해마다의 움직임이 같이 가는지(상관): 겉옷 봄 관심꺼짐 vs 기온 여름시작 r={r1:.2f} (n={n1}), "
        f"겉옷 가을 관심시작 vs 기온 가을시작 r={r2:.2f} (n={n2})")
    gap = (seasons["겉옷_가을_관심시작"].map(lambda t: doy(t) if pd.notna(t) else np.nan)
           - seasons["기온_가을시작"].map(lambda t: doy(t) if pd.notna(t) else np.nan)).dropna()
    say(f"  가을: 겉옷 관심 시작일 - 기온 가을시작일 = 평균 {gap.mean():+.1f}일, 해마다 {gap.min():+.0f} ~ {gap.max():+.0f}일")
    say("  연도별 추세 (직선 기울기, 일/년; 연도 수가 적어 참고용):")
    for col in ["기온_봄길이", "기온_여름길이", "기온_가을길이", "겉옷_안찾는기간"]:
        s = seasons[col]
        say(f"    {col}: {trend_per_year(s):+.1f}일/년 (n={int(s.notna().sum())})")

    # ---- Q6
    say("")
    say("=" * 70)
    say("Q6. 올해 패딩은 언제 몰릴까? (패딩만 날씨와 연동된다는 결과를 올해에 적용)")
    say("=" * 70)
    _, dpad = best_detail["패딩"]
    info_pad = best_detail["패딩"][0]
    thr_pad = int(bests["패딩"])
    lag_pad = float(info_pad["mean_lag"])
    mae_pad = float(robust["패딩"]["nested_wx"])
    sm7p = smooth(cleaned, SMOOTH_DAYS)
    es = padding_event_study(sm7p, dpad)
    mean_curve = es.mean(axis=1)
    half_x = int(mean_curve[mean_curve >= 0.5].index[0])
    peak_x = int(mean_curve.idxmax())
    marks = dict(half=half_x, peak=int(round(lag_pad)))
    p80_x = int(mean_curve[mean_curve >= 0.8].index[0])
    marks_data = dict(half=half_x, p80=p80_x)
    say(f"  기준 기온 {thr_pad}도, 평균 시차 +{lag_pad:.1f}일, 공정한 검증의 평균 오차 ±{mae_pad:.1f}일")
    say(f"  기온이 기준에 닿은 날을 0으로 맞춘 9년 평균 검색 곡선: 정점의 절반을 넘은 날 {half_x:+d}일, 평균 곡선이 가장 높은 날 {peak_x:+d}일")
    for lv in (0.3, 0.5, 0.8):
        say(f"    평균이 정점의 {int(lv * 100)}%를 처음 넘은 날: {int(mean_curve[mean_curve >= lv].index[0]):+d}일")
    pd.DataFrame({"days_from_trigger": mean_curve.index, "mean_share_of_peak": mean_curve.round(3).values}).to_csv(
        OUT_DIR / "q6_event_study.csv", index=False, encoding="utf-8-sig")

    cross = dpad.sort_values("year")["trigger_doy"].astype(float)
    cross_dates = pd.to_datetime(dpad["trigger_date"])
    now_cross = trigger_date(temp7, last_day.year, thr_pad, "cold", (9, 1))
    temp_now = float(temp7.dropna().iloc[-1])
    if pd.isna(now_cross):
        say(f"  올해는 아직 {thr_pad}도 아래로 내려가지 않았음 -> 시나리오로 예상")
        scen_def = [("올해 이른 추위 시나리오", cross.min(), 6), ("올해 평년 시나리오", cross.median(), 6), ("올해 늦은 추위 시나리오", cross.max(), 6)]
    else:
        say(f"  올해는 {now_cross:%m-%d}에 닿음 -> 실제 도달일로 예상")
        scen_def = [("올해 실제 도달", float(doy(now_cross)), 6)]
    cond = None
    if pd.isna(now_cross):
        trail = temp.rolling(7).mean()
        yrs = [int(y) for y in dpad.sort_values("year")["year"]]
        xs = np.array([trail.loc[pd.Timestamp(y, last_day.month, last_day.day)] for y in yrs], float)
        cs = dpad.sort_values("year")["trigger_doy"].to_numpy(float)
        xnow = float(trail.loc[last_day])
        n6 = len(xs)
        e_reg, e_mean = [], []
        for i in range(n6):
            m = np.arange(n6) != i
            b_, a_ = np.polyfit(xs[m], cs[m], 1)
            e_reg.append(abs(cs[i] - (a_ + b_ * xs[i])))
            e_mean.append(abs(cs[i] - cs[m].mean()))
        b_all, a_all = np.polyfit(xs, cs, 1)
        r_all = float(np.corrcoef(xs, cs)[0, 1])
        rng6 = np.random.default_rng(PERM_SEED)
        hits6 = sum(abs(np.corrcoef(xs, rng6.permutation(cs))[0, 1]) >= abs(r_all) for _ in range(5000))
        p6 = (hits6 + 1) / 5001
        cond_cross = a_all + b_all * xnow
        cond = dict(cross=cond_cross, e_reg=float(np.mean(e_reg)), e_mean=float(np.mean(e_mean)), r=r_all, p=p6,
                    slope=float(b_all), xnow=xnow, xmin=float(xs.min()), xmax=float(xs.max()), xmean=float(xs.mean()))
        say("")
        say(f"  [오늘 기온을 반영한 예상] {last_day:%m-%d}까지 최근 7일 평균기온이 높은 해일수록 기온이 {thr_pad}도에 늦게 닿았는지 확인")
        say(f"    과거 9년 같은 날 최근 7일 평균기온: 평균 {cond['xmean']:.1f}도, 범위 {cond['xmin']:.1f}~{cond['xmax']:.1f}도 / 올해 {xnow:.1f}도")
        say(f"    상관 r={r_all:.2f} (우연일 확률 p={p6:.3f}), 1도 높을수록 {b_all:.1f}일 늦게 닿음")
        say(f"    한 해씩 빼고 맞힌 평균 오차: 이 방법 {cond['e_reg']:.1f}일 vs 그냥 평균 날짜 {cond['e_mean']:.1f}일")
        say(f"    올해 예상 도달일 {md_label(cond_cross)} (평년 중앙값 {md_label(cross.median())}보다 {cond_cross - cross.median():+.0f}일)"
            + (" - 올해 기온이 과거 범위를 벗어나 있어 신뢰가 낮음" if xnow > xs.max() or xnow < xs.min() else ""))
        band_c = float(np.sqrt(cond["e_reg"] ** 2 + mae_pad ** 2))
        scen_def.append(("올해 예상\n(오늘 기온 반영)", cond_cross, band_c))
    scen, rows6 = [], []
    for label, t, _ in scen_def:
        p = t + lag_pad
        bnd = _ if _ != 6 else mae_pad
        scen.append((label, t, p, bnd))
        rows6.append(dict(scenario=label.replace("\n", " "), cross_date=md_label(t), expected_peak=md_label(p),
                          expected_range=f"{md_label(p - bnd)}~{md_label(p + bnd)}"))
        say(f"    {label.replace(chr(10), ' ')}: 기온 도달 {md_label(t)} -> 정점 예상 {md_label(p)} (보통 {md_label(p - bnd)}~{md_label(p + bnd)} 안팎)")
    pd.DataFrame(rows6).to_csv(OUT_DIR / "q6_padding_forecast.csv", index=False, encoding="utf-8-sig")
    say(f"  (참고: 과거 9년 정점일 중앙값은 {md_label(peaks[peaks.event == '패딩']['peak_doy'].median())}. 예측의 평균 오차 ±{mae_pad:.1f}일은 공정한 검증 기준)")
    say("  (이 예측은 검색 관심이 몰리는 시기이며, 가격·재고는 보지 않았음)")
    fc = dict(today=last_day, temp_now=temp_now, mean_lag=lag_pad, mae=mae_pad,
              mid_cross=md_label(cross.median()), mid_peak=md_label(cross.median() + lag_pad), cond=cond,
              cond_peak=md_label(cond["cross"] + lag_pad) if cond else None)

    # ---- 변화율
    say("")
    say("=" * 70)
    say("7. 변화율: 연평균 검색 지수와 전년 대비 증감")
    say("=" * 70)
    yearly = cleaned.groupby(cleaned.index.year).mean()
    change = yearly.pct_change() * 100
    level = pd.concat({"연평균": yearly.round(4), "전년대비(%)": change.round(1)}, axis=1)
    level.to_csv(OUT_DIR / "yearly_level.csv", encoding="utf-8-sig")
    say(level.to_string())
    say(f"  ({last_day.year}년은 {last_day:%m-%d}까지만 있어서 연평균이 계절에 따라 왜곡됨 -> 비교에서 제외)")

    # ---- 그래프
    say("")
    say("=" * 70)
    say("8. 그래프 저장")
    say("=" * 70)
    fig_overview(cleaned, sm)
    fig_monthly(normalized, y0, y1)
    fig_peaks(peaks)
    fig_weather_vs_calendar(detail, best_detail, bests, robust)
    fig_year_by_year(best_detail, bests)
    fig_sweep(sweeps, tstar, bests)
    fig_seasons(seasons)
    fig_event_study(es, marks_data)
    fig_forecast(dpad, scen, fc)

    (OUT_DIR / "summary.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print("\n끝났어요. images/ 폴더의 그래프와 outputs/summary.txt 를 확인하세요.")


if __name__ == "__main__":
    main()
