# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 1. 가설 - K-Beauty는 성장하고 있을까? 
# ## 가설세우기, 데이터 로드, 전처리
# %% [markdown]
# ## e : 추정치, p : 잠정치, - : 자료없음, ... : 미상자료, x : 비밀보호, ▽ : 시계열 불연속

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
import os
if os.name =='nt':
    sns.set(font="Malgun Gothic")
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# %%
df_raw = pd.read_csv('C:/Users/wkddn/Downloads/국가_대륙_별_상품군별_온라인쇼핑_해외직접판매액_20210921150126.csv', encoding = 'cp949')


# %%
df_raw.shape


# %%
# 국가(대륙)별 데이터 빈도수 세기
df_raw['국가(대륙)별'].value_counts()

# %% [markdown]
# ## 분석과 시각화를 위한 tidy data 만들기

# %%
df = df_raw.melt(id_vars=['국가(대륙)별', '상품군별', '판매유형별'], var_name='기간', value_name='백만원')


# %%
df.shape


# %%
df.info()


# %%
int("2021 2/4 p".split()[0])


# %%
df['연도'] = df['기간'].map(lambda x : int(x.split()[0]))


# %%
df['분기'] = df['기간'].map(lambda x: int(x.split()[1].split('/')[0]))


# %%
df.head()


# %%
df['백만원'] = df['백만원'].replace('-', pd.np.nan).astype(float)


# %%
df.head()


# %%
df = df[(df['국가(대륙)별'] != '합계') & (df['상품군별'] != '합계')].copy()


# %%
df.info()


# %%
df.isnull().sum()

# %% [markdown]
# ## 시각화

# %%
df_total = df[df['판매유형별'] == "계"].copy()


# %%
sns.lineplot(data=df_total, x='연도', y='백만원')


# %%
df_total.info()


# %%
sns.lineplot(data=df_total, x='연도', y='백만원', hue='상품군별')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# %%
# 자세히 보기 위해 서브플롯으로 표현
sns.relplot(data=df_total, x='연도', y='백만원', 
            hue='상품군별', kind='line', col='상품군별',
            col_wrap=4)


# %%
# 화장품이 압도적으로 높아 다른 값들을 보기가 힘드므로 화장품, 의류 및 패션 관련상품을 제외하고 확인
df_sub = df_total[~df_total['상품군별'].isin(['화장품', '의류 및 패션 관련상품'])].copy()


# %%
sns.relplot(data=df_sub, x='연도', y='백만원', hue='상품군별',
            col='상품군별', col_wrap=4, kind='line')


# %%
df_cosmetic = df_total[df_total['상품군별'] == '화장품'].copy()


# %%
df_cosmetic['상품군별'].unique()


# %%
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_cosmetic, x='연도', y='백만원', hue='분기')


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_cosmetic, x='기간', y='백만원')


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_cosmetic[df_cosmetic['국가(대륙)별'] != '중국'], x='기간', y='백만원', hue='국가(대륙)별')


# %%
df_sub = df[(df['판매유형별'] != '계') & (df['판매유형별'] != '면세점')].copy()


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_sub, x='기간', y='백만원', hue='판매유형별', ci=None)


# %%
df_fashion = df[df["상품군별"] == "의류 및 패션 관련상품"].copy()
df_fashion.head()


# %%
df_fashion = df[(df["상품군별"] == "의류 및 패션 관련상품") & (df["판매유형별"] == "계")].copy()
df_fashion.head()


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_fashion, x="기간", y="백만원", hue="국가(대륙)별")


# %%
df_fashion2 = df[(df["상품군별"] == "의류 및 패션 관련상품") & (df["판매유형별"] != "계")].copy()

plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_fashion2, x="기간", y="백만원", hue="판매유형별", ci=None)


# %%
df_fashion.pivot_table(index="국가(대륙)별", columns="연도", values="백만원")


# %%
df_fashion.pivot_table(index="국가(대륙)별", columns="연도", values="백만원", aggfunc="sum")


# %%
df_fashion["판매유형별"].value_counts()


# %%
result = df_fashion.pivot_table(index="국가(대륙)별", columns="연도", values="백만원", aggfunc="sum")
result


# %%
sns.heatmap(result)


# %%
sns.heatmap(result, annot=True)


# %%
# fmt == 소숫점 없이 float형의 숫자를 나타낼 수 있음
plt.figure(figsize=(10, 6))
sns.heatmap(result, annot=True, fmt='.0f')


# %%
df_total


# %%
sns.barplot(data=df_total, x='연도', y='백만원')


# %%
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_total, x="연도", y="백만원", hue="국가(대륙)별")


# %%
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_total, x="연도", y="백만원", hue="상품군별")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0,)

# %% [markdown]
# # 온라인 쇼핑액이 증가하고 있고, 판매를 하는 사람들이 많아지고 있다는 것을 알 수 있음

