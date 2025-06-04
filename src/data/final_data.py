import pandas as pd

def join_top_data(data_a, data_b, year: list):
    '''
    Description:
        Join the Top 5 data by year.
    Parameters:
        data_a, data_b: dataframes
            Dataframes to join.
    year:
        Years range to join.
    Return:
        Joined Dataframe in a vector
    '''
    data_a = data_a.query('ANO_LICITACAO == @year')
    data_b = data_b.query('ANO_LICITACAO == @year')
    data = data_a.merge(data_b, on='NR_LICITACAO')
    data.drop(['CD_ORGAO_x', 'CD_ORGAO_y', 	'ANO_LICITACAO_x',
                'NOME_ORGAO', 'CD_ORGAO_y', 'NM_ORGAO', 
                'ANO_LICITACAO_y', 'NR_LICITACAO'], axis=1,
                 inplace=True)
    data.dropna(inplace=True)
    return data
    
# Works as a DAG from join_top_data
def build_pred_data(data_a, data_b, years: list):
    '''
    Description:
        Join the Top 5 data by year.
    Parameters:
        data_a, data_b: dataframes
            Dataframes to join.
    year:
        Years range to join.
    Return:
        Final Joined Dataframe in a vector
    '''
    data_vec = []

    for year in years:
        data_vec.append(join_top_data(data_a, data_b, year))

    data = data_vec[0].append(data_vec[1])

    for i in range(len(data_vec)-2):
        data = data.append(data_vec[i+2])
    data.reset_index(drop=True, inplace=True)

    return data