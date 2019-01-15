
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movie_data = pd.read_csv('tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[3]:


row = movie_data.shape[0]
col = movie_data.shape[1]
print('movie_data数据集共有{}行，{}列'.format(row, col))


# In[4]:


movie_data.head()


# In[5]:


movie_data.tail()


# In[6]:


movie_data.sample(5)


# In[7]:


movie_data.dtypes


# In[8]:


movie_data.isnull().any()


# In[61]:


num_na_col = movie_data.isnull().sum()[movie_data.isnull().any()]
print('该数据集有缺失值，各列缺失量如下：\n{}'.format(num_na_col))


# In[10]:


movie_data.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# ### 方法三

# 根据一审老师的建议增补如下：

# 先查看缺失值的分布情况：

# In[3]:


movie_data.info()


# In[4]:


msno.missingno.matrix(movie_data);


# 从图表中可看出，缺失值多分布于homepage、tagline、keywords和production_companies这几列，少量分布于imdb_id、cast、director、genres。这些都是字符型，除director和genres都未参与后续分析。可以将它们填充为“Unknown”。

# In[5]:


movie_data.fillna('Unknown', inplace = True)
msno.missingno.matrix(movie_data);


# 如果缺失值的分布同时涉及字符型和数值型，需分开处理。则以上几列需要单独选取出来：

# In[6]:


nan_col = movie_data.columns[movie_data.isnull().any()]
obj_nan_col = nan_col[movie_data[nan_col].dtypes == 'object']
num_nan_col = nan_col[movie_data[nan_col].dtypes.isin(['int64', 'float64'])]
movie_data[obj_nan_col] = movie_data[obj_nan_col].fillna('Unknown')
movie_data[num_nan_col] = movie_data[num_nan_col].fillna(movie_data[num_nan_col].median())


# ### 方法一

# In[11]:


movie_data = pd.read_csv('tmdb-movies.csv') # 因为各种处理方法是平行步骤，为避免上一节的影响，需重新读取
movie_data.dropna(axis = 0, inplace = True)
movie_data.isnull().any().sum()


# 对缺失值采用删除法，删掉包含缺失值的样本（*行*）。

# In[12]:


row_new = movie_data.shape[0]
col = movie_data.shape[1]
diff = row - row_new
print('删除缺失值后的movie_data数据集共有{}行，{}列，较原来减少{}个样本。'.format(row_new, col, diff))


# 可见删除缺失样本的方法，会造成大量的数据缺失，是否值得？考虑重新读取数据，对缺失值采用插补法。

# ### 方法二

# In[13]:


movie_data = pd.read_csv('tmdb-movies.csv')
movie_data.fillna(method = 'ffill', axis = 0, inplace = True)


# In[15]:


movie_data.isnull().any()


# In[16]:


row_new = movie_data.shape[0]
col = movie_data.shape[1]
diff = row - row_new
print('处理缺失值后的movie_data数据集共有{}行，{}列，较原来减少{}个样本。'.format(row_new, col, diff))


# 查看处理后的分布情况：

# In[17]:


movie_data.describe()


# 我想这是比较合适的办法。或者用`.interpolate(method = 'linear', axis)`，命令书写类似 。

# In[18]:


movie_data = pd.read_csv('tmdb-movies.csv')
movie_data.interpolate(method = 'linear', axis = 0, inplace = True)
movie_data.describe()


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[19]:


movie_data[['id', 'popularity','budget','runtime','vote_average']]


# In[5]:


movie_data[:20].append(movie_data[47:49])


# 根据一审老师的意见增补：

# In[27]:


movie_data.iloc[list(range(20)) + [47, 48]]


# In[21]:


movie_data[['popularity']][49:60]


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[22]:


movie_data[movie_data.popularity > 5]


# In[23]:


movie_data[(movie_data.popularity > 5) & (movie_data.release_year > 1996)]


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[24]:


movie_data.groupby(['release_year'])['revenue'].agg(['mean'])


# In[28]:


movie_data.groupby(['director'])['popularity'].agg(['mean']).sort_values('mean', ascending = False)


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[29]:


movie_ppl = movie_data[['original_title', 'popularity']].sort_values(by = 'popularity',ascending = False)[:20]


# In[30]:


plt.figure(figsize = [12, 7])
sb.barplot(data = movie_ppl, x = 'popularity', y = 'original_title', color = 'pink');
plt.ylabel('Movies')


# 参考资料：
# 1. https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot

# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# 根据一审意见修改：

# In[193]:


movie_data['net'] = movie_data.revenue - movie_data.budget
movie_net = pd.DataFrame(movie_data.groupby(['release_year'])['net'].mean())
movie_net['sem'] = list(movie_data.groupby(['release_year'])['net'].sem())
movie_net['sum'] = list(movie_data.groupby(['release_year'])['net'].sum())
movie_net['count'] = list(movie_data.groupby(['release_year'])['net'].count())
movie_net['release_year'] = movie_net.index
movie_net.head()


# In[208]:


plt.figure(figsize = [14, 10])
plt.subplot(3, 1, 1)
plt.errorbar(data = movie_net, x = 'release_year', y = 'net', yerr = 'sem');
plt.xticks([], []);
plt.title('Mean of net profit per year');

plt.subplot(3, 1, 2)
plt.errorbar(data = movie_net, x = 'release_year', y = 'count');
plt.xticks([], []);
plt.title('Movie counts per year')

plt.subplot(3, 1, 3)
plt.errorbar(data = movie_net, x = 'release_year', y = 'sum');
plt.xticks(movie_net['release_year'], rotation = 60);
plt.title('Sum of net profit per year')


# 每年电影平均净利润从1960年至2015年间，总体呈上升趋势，前期较平缓，在1971~1973年间有明显的抬高，此后平稳上升。每年均值波动范围较相当，但1977年波动较大，可能说明当年有特别出众或特别糟糕的电影。
# 
# 每年总净利润总体呈上升趋势。1960至2000年之间上升速率较缓，2000年之后上涨幅度明显增大。每年电影发行的数量与总利润的增涨趋势接近。

# 参考资料：
# https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot

# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[34]:


tmp = movie_data['director'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('director') 
movie_data_split = movie_data[['original_title', 'revenue']].join(tmp)
movie_data_split = movie_data_split[movie_data_split.director != 'Unknown']
top_directors = pd.DataFrame(movie_data_split.groupby(['director'])['original_title'].count()).sort_values(by = 'original_title', ascending = False)[:10].index
movie_dir = movie_data_split[movie_data_split.director.isin(top_directors)]
movie_dir.head()


# 参考资料：
# 1. https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/

# In[29]:


movie_dir = movie_dir.sort_values(['director', 'revenue'], ascending = False)


# In[30]:


def top_movie(x):
    df = pd.DataFrame()
    for i in range(len(x)):
        df_1 = movie_dir[movie_dir.director == x[i]][:3]
        df = df.append(df_1)
    return df


# In[31]:


movie_dir_top = top_movie(top_directors)


# In[32]:


g = sb.FacetGrid(data = movie_dir_top, col = 'director',col_wrap = 5, sharey = False, height = 4,aspect = 1.5)
g.map(sb.barplot, 'revenue', 'original_title')
g.set_titles('{col_name}')
g.fig.subplots_adjust(hspace=0.1,wspace = 2);


# In[33]:


rev_dir = movie_dir_top.groupby(['director'])['revenue'].sum()
print('最多产的前10名导演是{}，总值在{}至{}之间。其中以{}总票房最高。'.format(list(top_directors),
                                                    rev_dir.min(), rev_dir.max(),
                                                    list(rev_dir[rev_dir == rev_dir.max()].index)))


# 参考资料：
# 1. https://seaborn.pydata.org/tutorial/axis_grids.html

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[8]:


sel_year = movie_data.release_year.isin(list(range(1968, 2016)))
month = list(map(lambda x: x.split('/')[1], movie_data['release_date']))
sel_June = pd.Series(list((map(lambda x: x == '6', month))))
movie_june = movie_data[sel_year&sel_June]


# In[16]:


movie_june['release_year'].value_counts().sort_index().plot(kind='bar', figsize=(15, 5), color = 'pink');
plt.xlabel('Release Year');
plt.ylabel('Movie Count');


# 1968-2015年间，六月发行的电影数量整体增长。1995年之前增幅平缓，1996年六月出现一个小高峰，回落之后增幅较之前明显提升；2010年明显回落，之后继续以原来的速度增长。2013年产量最高。

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[10]:


tmp = movie_june['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genres') 
movie_june_split = movie_june[['original_title', 'release_year']][sel_year].join(tmp)
movie_genres = movie_june_split[movie_june_split.genres.isin(['Comedy', 'Drama'])]
movie_genres.head()


# In[14]:


plt.figure(figsize = [15, 6])
sb.countplot(data = movie_genres, x = 'release_year', hue = 'genres', palette="Set2");
plt.xticks(rotation = 90)
plt.ylabel('Movie Count');
plt.xlabel('Release Year');


# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
