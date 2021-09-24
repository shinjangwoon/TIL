# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 1. ���� - K-Beauty�� �����ϰ� ������? 
# ## ���������, ������ �ε�, ��ó��
# %% [markdown]
# ## e : ����ġ, p : ����ġ, - : �ڷ����, ... : �̻��ڷ�, x : ��к�ȣ, �� : �ð迭 �ҿ���

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
df_raw = pd.read_csv('C:/Users/wkddn/Downloads/����_���_��_��ǰ����_�¶��μ���_�ؿ������Ǹž�_20210921150126.csv', encoding = 'cp949')


# %%
df_raw.shape


# %%
# ����(���)�� ������ �󵵼� ����
df_raw['����(���)��'].value_counts()

# %% [markdown]
# ## �м��� �ð�ȭ�� ���� tidy data �����

# %%
df = df_raw.melt(id_vars=['����(���)��', '��ǰ����', '�Ǹ�������'], var_name='�Ⱓ', value_name='�鸸��')


# %%
df.shape


# %%
df.info()


# %%
int("2021 2/4 p".split()[0])


# %%
df['����'] = df['�Ⱓ'].map(lambda x : int(x.split()[0]))


# %%
df['�б�'] = df['�Ⱓ'].map(lambda x: int(x.split()[1].split('/')[0]))


# %%
df.head()


# %%
df['�鸸��'] = df['�鸸��'].replace('-', pd.np.nan).astype(float)


# %%
df.head()


# %%
df = df[(df['����(���)��'] != '�հ�') & (df['��ǰ����'] != '�հ�')].copy()


# %%
df.info()


# %%
df.isnull().sum()

# %% [markdown]
# ## �ð�ȭ

# %%
df_total = df[df['�Ǹ�������'] == "��"].copy()


# %%
sns.lineplot(data=df_total, x='����', y='�鸸��')


# %%
df_total.info()


# %%
sns.lineplot(data=df_total, x='����', y='�鸸��', hue='��ǰ����')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# %%
# �ڼ��� ���� ���� �����÷����� ǥ��
sns.relplot(data=df_total, x='����', y='�鸸��', 
            hue='��ǰ����', kind='line', col='��ǰ����',
            col_wrap=4)


# %%
# ȭ��ǰ�� �е������� ���� �ٸ� ������ ���Ⱑ ����Ƿ� ȭ��ǰ, �Ƿ� �� �м� ���û�ǰ�� �����ϰ� Ȯ��
df_sub = df_total[~df_total['��ǰ����'].isin(['ȭ��ǰ', '�Ƿ� �� �м� ���û�ǰ'])].copy()


# %%
sns.relplot(data=df_sub, x='����', y='�鸸��', hue='��ǰ����',
            col='��ǰ����', col_wrap=4, kind='line')


# %%
df_cosmetic = df_total[df_total['��ǰ����'] == 'ȭ��ǰ'].copy()


# %%
df_cosmetic['��ǰ����'].unique()


# %%
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_cosmetic, x='����', y='�鸸��', hue='�б�')


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_cosmetic, x='�Ⱓ', y='�鸸��')


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_cosmetic[df_cosmetic['����(���)��'] != '�߱�'], x='�Ⱓ', y='�鸸��', hue='����(���)��')


# %%
df_sub = df[(df['�Ǹ�������'] != '��') & (df['�Ǹ�������'] != '�鼼��')].copy()


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_sub, x='�Ⱓ', y='�鸸��', hue='�Ǹ�������', ci=None)


# %%
df_fashion = df[df["��ǰ����"] == "�Ƿ� �� �м� ���û�ǰ"].copy()
df_fashion.head()


# %%
df_fashion = df[(df["��ǰ����"] == "�Ƿ� �� �м� ���û�ǰ") & (df["�Ǹ�������"] == "��")].copy()
df_fashion.head()


# %%
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_fashion, x="�Ⱓ", y="�鸸��", hue="����(���)��")


# %%
df_fashion2 = df[(df["��ǰ����"] == "�Ƿ� �� �м� ���û�ǰ") & (df["�Ǹ�������"] != "��")].copy()

plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_fashion2, x="�Ⱓ", y="�鸸��", hue="�Ǹ�������", ci=None)


# %%
df_fashion.pivot_table(index="����(���)��", columns="����", values="�鸸��")


# %%
df_fashion.pivot_table(index="����(���)��", columns="����", values="�鸸��", aggfunc="sum")


# %%
df_fashion["�Ǹ�������"].value_counts()


# %%
result = df_fashion.pivot_table(index="����(���)��", columns="����", values="�鸸��", aggfunc="sum")
result


# %%
sns.heatmap(result)


# %%
sns.heatmap(result, annot=True)


# %%
# fmt == �Ҽ��� ���� float���� ���ڸ� ��Ÿ�� �� ����
plt.figure(figsize=(10, 6))
sns.heatmap(result, annot=True, fmt='.0f')


# %%
df_total


# %%
sns.barplot(data=df_total, x='����', y='�鸸��')


# %%
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_total, x="����", y="�鸸��", hue="����(���)��")


# %%
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_total, x="����", y="�鸸��", hue="��ǰ����")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0,)

# %% [markdown]
# # �¶��� ���ξ��� �����ϰ� �ְ�, �ǸŸ� �ϴ� ������� �������� �ִٴ� ���� �� �� ����

