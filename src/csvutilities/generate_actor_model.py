import pandas as pd
import numpy as np

RANGE_MIN = 1

RANGE_MAX = 2

DATASET_ROOT_PATH = '/Users/faraday/Documents/Projects/mwdb/Mwdb_phase1/resources/phase1_dataset/'


def readcsv(name):
    return pd.read_csv(name)

def generate_actor_model(actorid):
    print("Generating actor model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
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
    df_tags_with_min_max['ts_rank'] = df_tags_with_min_max.apply(
        lambda row: ((row['timestamp'].timestamp() - row['timestamp_min'].timestamp())
                     / (row['timestamp_max'].timestamp() - row['timestamp_min'].timestamp())
                     if row['timestamp_max'].timestamp() != row['timestamp_min'].timestamp() else 1)
                    * (RANGE_MAX - RANGE_MIN) + RANGE_MIN, axis=1)

    df_actor = readcsv(DATASET_ROOT_PATH + 'movie-actor.csv')
    min_tag = df_actor['actor_movie_rank'].groupby(df_actor['movieid']).min()
    min_tag = min_tag.to_frame()
    min_tag = min_tag.reset_index(level=['movieid'])
    max_tag = df_actor['actor_movie_rank'].groupby(df_actor['movieid']).max()
    max_tag = max_tag.to_frame()
    max_tag = max_tag.reset_index(level=['movieid'])
    min_tag.columns = ['movieid', 'rank_min']
    max_tag.columns = ['movieid', 'rank_max']
    min_max_tag = pd.merge(min_tag, max_tag, on='movieid')
    df_actor_with_min_max = pd.merge(df_actor, min_max_tag, on='movieid')
    df_actor_with_min_max['rank'] = df_actor_with_min_max.apply(
        lambda row: ((row['actor_movie_rank'] - row['rank_min'])
                     / (row['rank_max'] - row['rank_min'])
                     if row['rank_max'] != row['rank_min'] else 1)
                    * (RANGE_MAX - RANGE_MIN) + RANGE_MIN, axis=1)

    merged = pd.merge(df_actor_with_min_max.iloc[:,[0,1,2,5]],df_tags_with_min_max.iloc[:,[1,2,6]], on='movieid')
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