import pandas as pd
import os

def build_features(dataset: str, data_path: str, cols: list, values: list):
    '''
    Description:
        Function to build the required fields to public bids analysis.
        It's possible to get data on multiple folders by years name.
        All bids are filtered by Buying status (TP_OBJETO == 'COM')
    Parameters:
        dataset: string 
            A string with the dataset name.
        data_path: string
            The data path on the development environment.
        cols: list
            The required columns (features) to the new dataset.
    Return:
        A list with the built datasets.
    '''
    print(f'Building Features for {dataset}...')
    datasets = []
    # Iterate to get data on multiple folders
    for idx in values:
        data = pd.read_csv(data_path+os.sep+'external'+
                            os.sep+str(idx)+os.sep+dataset+'.csv',
                            usecols=cols
                                    )
        data.drop_duplicates(inplace=True)
        # For processing "licitacao.csv"
        # Filter by Buy (COM)
        if dataset == 'licitacao':
            data.query('TP_OBJETO == "COM"', inplace=True)
            data.drop(['TP_OBJETO'], axis=1, inplace=True)
            datasets.append(data)
        # For processing "item.csv"
        elif dataset == 'item':
            # Gets the bid year
            data['ANO_LICITACAO'] = idx
            datasets.append(data)
        else:
            pass
    # Merge Datasets by conditional case.
    if len(datasets) == 1:
        data_proc = datasets[0]
    elif len(datasets) == 2:
        data_proc = datasets[0].append(datasets[1])
    else:    
        data_proc = datasets[0].append(datasets[1])
        for i in range(len(datasets)-2):
            data_proc = data_proc.append(datasets[i+2])
        data_proc.reset_index(drop=True, inplace=True)
    # Save the features on a parquet file    
    print('Saving extracted features to parquet file...')
    data_proc.to_parquet(data_path+os.sep+'processed'+os.sep+
                    str(min(values))+'_'+str(max(values))+
                    '_'+dataset+'.parquet')
    print('Building features process finished!')