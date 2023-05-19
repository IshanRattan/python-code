
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BusinessDataAnalyzer():

    _CLASS_VARS = ['business_csv_path', 'newbusiness_csv_path', 'countries_csv_path', 'categories_csv_path']

    def __init__(self, **kwargs):
        self._mapping = {key : value for key, value in kwargs.items() if key in self._CLASS_VARS}
        try:
            assert len(self._mapping) == len(self._CLASS_VARS)
        except AssertionError as e:
            raise(AssertionError(f'One or more input variables are missing! {e}'))

    def _load_csv(self, path : str) -> pd.read_csv:
        '''Reads input csv from the path specified
           Returns pandas df.
           :param path: [str] csv path
           :return: pd.DataFrame
           '''
        return pd.read_csv(path)

    def _group_by(self, dataframe : pd.DataFrame,
                  columns : list,
                  aggfunc : dict = None) -> pd.DataFrame.groupby:
        '''
        Takes pandas df, column name(s) & aggregate function(optional) as inputs
           Returns pandas.DataFrame.groupby object.
        :param dataframe: pandas df
        :param columns: [str] column(s) criteria
        :param aggfunc: [dict] calculation metric(mean, median etc) [optional]
        :return: pandas.DataFrame.groupby object
        '''
        try:
            if aggfunc:
                return dataframe.groupby(by=columns).agg(aggfunc)
            else:
                return dataframe.groupby(by=columns)
        except Exception as e:
            print(e)
            return None

    def _merge(self, dataframe : pd.DataFrame,
               dataframe_to_merge : pd.DataFrame,
               on : list,
               how : str = None,
               indicator : bool = False) -> pd.DataFrame:
        '''
        Merges 2 pandas dfs
        Returns pandas df.
        :param dataframe: first dataframe
        :param dataframe_to_merge: second dataframe
        :param on: [str] column name to merge on
        :param how: [str] merge type('outer', 'left', 'right' etc) [optional]
        :param indicator: [bool] adds info on source of each row [optional]
        :return:
        '''
        if how or indicator:
            if how:
                if indicator: return dataframe.merge(dataframe_to_merge, on=on, how=how, indicator=True)
                else: return dataframe.merge(dataframe_to_merge, on=on, how=how)
            else:
                return dataframe.merge(dataframe_to_merge, on=on, indicator=indicator)
        else:
            return dataframe.merge(dataframe_to_merge, on=on)

    def _sort_data(self, dataframe : pd.DataFrame,
                   sort_by : list = None,
                   ascending : bool = False) -> pd.DataFrame:
        '''
        Sort pandas df by one or more columns.
        :param dataframe: pandas df
        :param sort_by: [list(str)] list of column or columns to sort by
        :param ascending: [bool] sorted data in ascending order [optional]
        :return: pandas df
        '''
        return dataframe.sort_values(by=sort_by, ascending=ascending)

    def _plot_data(self, dataframe : pd.DataFrame,
                   x : str = None,
                   y : str = None,
                   kind : str ='count',
                   col : str = None,
                   col_wrap : int = None,
                   hue=None) -> sns.catplot:
        '''
        Creates plot based on user input.
        If no y label is provided, countplot is generated.
        :param dataframe: pandas df
        :param x: [str] x-axis variable [optional]
        :param y: [str] y-axis variable [optional]
        :param kind: [str] plot type [optional]
        :param col: [str] split visualisation into multiple plots based on column value [optional]
        :param col_wrap: number of plots per column [optional]
        :return: sns.catplot
        '''
        vars = locals()
        del vars['self']
        del vars['dataframe']

        func_args = ', '.join([f"{key}='{value}'" if value and isinstance(value, str) else f"{key}={value}" for key, value in vars.items()])
        return eval(f'sns.catplot(data=dataframe, {func_args})')


    def analyze(self):
        '''
        :return:
        '''
        businesses = self._load_csv(self._mapping['business_csv_path'])
        countries = self._load_csv(self._mapping['countries_csv_path'])
        new_businesses = self._load_csv(self._mapping['newbusiness_csv_path'])
        categories = self._load_csv(self._mapping['categories_csv_path'])

        businesses_categories = self._merge(businesses, categories, on=['category_code'], how='outer')
        count_business_cats = self._group_by(businesses_categories, columns=['category'], aggfunc={'year_founded' : 'count'})
        count_business_cats.columns = ['count_business_cats']
        print(count_business_cats)

        businesses_categories['category_code'].unique()
        old_restaurants = businesses_categories.query('category_code == "CAT4" & year_founded < 1800')

        old_restaurants = self._sort_data(old_restaurants, sort_by=['year_founded'])
        print(old_restaurants)

        businesses_categories_temp = self._merge(businesses, categories, on=['category_code'], how='inner')
        businesses_categories_countries = self._merge(businesses_categories_temp,
                                                      countries,
                                                      on=['country_code'],
                                                      how='inner')

        businesses_categories_countries = self._sort_data(businesses_categories_countries,
                                                          sort_by=['year_founded'],
                                                          ascending=True)

        plot_businesses = self._plot_data(dataframe=businesses_categories_countries,
                                   y='category',
                                   col='continent',
                                   col_wrap=2)
        plot_businesses.fig.suptitle('No. of businesses per category')
        plt.show()

        oldest_by_continent_category = self._group_by(businesses_categories_countries,
                                                      columns=['continent', 'category']).agg({'year_founded' : 'min'})


        print(oldest_by_continent_category.head())


csvs = {'business_csv_path' : 'data/businesses.csv',
        'newbusiness_csv_path' : 'data/new_businesses.csv',
        'countries_csv_path' : 'data/countries.csv',
        'categories_csv_path' : 'data/categories.csv'}

obj = BusinessDataAnalyzer(**csvs)
obj.analyze()
