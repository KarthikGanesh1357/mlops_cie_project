import os
import pandas as pd
from IPython.display import HTML, display

def load_data(data_path: str, years: list):
    '''
    Description:
        Function that loads the processed data for bids and items.
        The data is on parquet format.
    Parameters:
        data_path: string
            A string with de path of the processed data.
        years: list
            The list of the covered years, with int values.
    Return:
        bid:
            The processed public bids.
        item:
            The processed items.
    '''
    bid = pd.read_parquet(data_path+os.sep+'processed'+
                            os.sep+str(min(years))+'_'+str(max(years))+
                            '_'+'licitacao.parquet')
    item = pd.read_parquet(data_path+os.sep+'processed'+
                            os.sep+str(min(years))+'_'+str(max(years))+
                            '_'+'item.parquet')
    # Display data without any duplicated samples                        
    display(HTML('<b>Public Bids Dataset</b>'), bid.head(3))
    bid.drop_duplicates(inplace=True)
    print(f'Number of samples: {bid.shape[0]}')
    print(' ')
    display(HTML('<b>Items Dataset</b>'), item.head(3))
    bid.drop_duplicates(inplace=True)
    print(f'Number of samples: {item.shape[0]}')
    print(' ')
    # Check for Public Organ Information
    organ_check = bid['CD_ORGAO'].nunique() == bid['NM_ORGAO'].nunique()

    if organ_check == True:
        print('There are no problems with Code and Organ names.')
    else:
        print('There are problems with Code and Organ names.')

    print(' ')
    print(f'There are {bid["CD_ORGAO"].nunique()} Public Organs in the Dataset.')   

    return bid, item