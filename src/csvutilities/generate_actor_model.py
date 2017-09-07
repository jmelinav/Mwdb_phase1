import pandas as pd
import numpy as np

DATASET_ROOT_PATH = '/Users/faraday/Documents/Projects/mwdb/Mwdb_phase1/resources/phase1_dataset/'


def readcsv(name):
    return pd.read_csv(name)

def generate_actor_model(actorid):
    print("Generating actor model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_actor = readcsv(DATASET_ROOT_PATH + 'movie-actor.csv')

    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')

    #min_tag = (df_tags['timestamp'].groupby(df_tags['movieid']).min()).to_frame()
    #print(min_tag.columns)
    min_tag = df_tags['timestamp'].groupby(df_tags['movieid']).min()
    min_tag = min_tag.to_frame()
    min_tag = min_tag.reset_index(level=['movieid'])
    max_tag = df_tags['timestamp'].groupby(df_tags['movieid']).max()
    max_tag = max_tag.to_frame()
    max_tag = max_tag.reset_index(level=['movieid'])
    min_tag.columns = ['movieid', 'timestamp_min']
    max_tag.columns = ['movieid', 'timestamp_max']
    min_max_tag = pd.merge(min_tag,max_tag,on='movieid')
    df_tags_with_min_max = pd.merge(df_tags,min_max_tag,on='movieid')
    df_tags_with_min_max['between 1-2'] = df_tags_with_min_max.apply(
        lambda row: ((row['timestamp'].timestamp() - row['timestamp_min'].timestamp())
                     / (row['timestamp_max'].timestamp() - row['timestamp_min'].timestamp())
                     if row['timestamp_max'].timestamp() != row['timestamp_min'].timestamp() else 1)
                    * (2 - 1) + 1, axis=1)

    merged = pd.merge(df_actor,df_tags_with_min_max, on='movieid')
    merged['COUNTER'] = 1
    group_data = pd.DataFrame(merged.groupby(['actorid', 'tagid'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['actorid'], 'tagid')
    merged.to_csv(
        DATASET_ROOT_PATH + 'actor_tag.csv',
        index=False)
    term_vector.to_csv(
        DATASET_ROOT_PATH + 'actor_tag_agg.csv')
    print(group_data.ix[:5])

generate_actor_model(5)