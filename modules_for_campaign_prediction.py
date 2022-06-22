# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:21:12 2022

@author: ANAS
"""

import seaborn as sns
import matplotlib.pyplot as plt

class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df,con_column,cat_column):
        '''
        

        Parameters
        ----------
        df : DATAFRAME
            Overall dataframe
        con_column : LIST
            Continous column list
        cat_column : LIST
            Categorical column list

        Returns
        -------
        None.

        '''
        
        # continous
        for con in con_column:
            plt.figure()
            sns.distplot(df[con])
            plt.title(con.capitalize())
            plt.savefig(os.path.join(PNG_PATH,f'distplot-{con}.png'))
            plt.show()
            
        # categorical
        for cate in cat_column:
            plt.figure()
            sns.countplot(df[cate])
            plt.title(cate.capitalize())
            plt.savefig(os.path.join(PNG_PATH,f'countplot-{cate}.png'))
            plt.show()
