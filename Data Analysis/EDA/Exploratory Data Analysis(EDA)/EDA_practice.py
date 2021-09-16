# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
import pandas as pd
import seaborn as sns


df = pd.read_csv('young_survey.csv')
df.head()

basic_info = df.iloc[:, 140:]
basic_info

basic_info.describe()

basic_info['Gender'].value_counts()

basic_info['Handedness'].value_counts()

basic_info['Education'].value_counts()

sns.violinplot(data=basic_info, y='Age')

sns.violinplot(data=basic_info, x='Gender', y='Age')

sns.violinplot(data=basic_info, x='Gender', y='Age', hue='Handedness')

sns.jointplot(data=basic_info, x='Height', y='Weight')

oc_df = pd.read_csv('occupations.csv')


oc_df.head()

# +

women = oc_df[oc_df['gender'] == 'F']
women['occupation']
# -

women['occupation'].value_counts().sort_values(ascending=False)

men = oc_df[oc_df['gender'] == 'M']
men.head()

men['occupation'].value_counts().sort_values(ascending=False)

# ## 상관 관계 분석 (Coorrelation Analysis)

music = df.iloc[:, :19]

sns.heatmap(music.corr())

df.corr()['Age'].sort_values(ascending=False).tail(20) # 나이가 증가에 따른 음악 취향

df.corr()['Getting up'][1:19].sort_values(ascending=True) # 아침에 일찍 일어나는 사람들이 좋아하는 음악 장르

# ## 클러스터 분석(Cluster Analysis)

df.head()

interests = df.loc[:,'History':'Pets']

interests.head()

corr = interests.corr()

corr

corr['History'].sort_values(ascending=False)

sns.clustermap(corr)

# ## 실습 : 영화 카페 운영하기
#

mv_df = pd.read_csv('survey.csv')
mv_df.head()

movie = mv_df.loc[:, 'Horror':'Action']

mc = movie.corr()

sns.clustermap(mc)

# ## 타이타닉 EDA

titanic = pd.read_csv('titanic.csv')
titanic.info()

# 문제: 다음 중 맞는 것을 모두 고르시오.
#
# 1.타이타닉의 승객은 30대와 40대가 가장 많다.
#
# 2.가장 높은 요금을 낸 사람은 30대이다.
#
# 3.생존자가 사망자보다 더 많다.
#
# 4.1등실, 2등실, 3등실 중 가장 많은 사람이 탑승한 곳은 3등실이다.
#
# 5.가장 생존율이 높은 객실 등급은 1등실이다.
#
# 6.나이가 어릴수록 생존율이 높다.
#
# 7.나이보다 성별이 생존율에 더 많은 영향을 미친다.
#
#
#

# 1. 타이타닉의 승객은 30대와 40대가 가장 많다 
# -> 20대가 가장 많기때문에 False
titanic.plot(kind='hist', y='Age', bins=30)


# 2. 가장 높은 요금을 낸 사람은 30대이다 == True
titanic.plot(kind='scatter', x='Age', y='Fare')

# 3. 생존자가 사망자보다 더 많다.
# 0 = 사망, 1 = 생존  
# 아쉽게도 사망자가 더 많기때문에 False
titanic['Survived'].value_counts()

# 4. 1등실, 2등실, 3등실 중 가장 많은 사람이 탑승한 곳은 3등실이다. == True
titanic['Pclass'].value_counts()

# 5. 가장 생존율이 높은 객실 등급은 1등실이다.
titanic.plot(kind='scatter', x='Pclass', y='Survived')

# +
# KDE Plot을 사용하여 겹쳐진 정도 확인
sns.kdeplot(titanic['Pclass'], titanic['Survived'])

# 위쪽에서 1등실이 제일 밀집되어 있으므로 생존율이 더 높다 == True

# +
# 6. 나이가 어릴수록 생존율이 높다 == False
sns.violinplot(data=titanic, x='Survived', y='Age')

# 생존한 사람들의 나이 분포와 사망한 사람들의 나이 분포 사이에는 큰 차이가 보이지 않아,
# 나이가 어릴수록 생존율이 높다고 하긴 어려울 것 같다
# -

# 7. 나이보다 성별이 생존율에 더 많은 영향을 미친다. == True
sns.stripplot(data=titanic, x='Survived', y='Age', hue='Sex')

# 따라서 보기 중 맞는 것은 '2,4,5,7'입니다.






