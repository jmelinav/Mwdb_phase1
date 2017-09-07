import pandas as pd
import numpy as np

def readcsv(name):
    return pd.read_csv(name)

def generate_actor_model(actorid):
    print("Generating actor model")
    df_tags = readcsv('/Users/mj/Documents/Courses/fall-2017/cse-515-mwdb/phase-1/Mwdb_phase1/resources/phase1_dataset/mltags.csv')
    df_actor = readcsv('/Users/mj/Documents/Courses/fall-2017/cse-515-mwdb/phase-1/Mwdb_phase1/resources/phase1_dataset/movie-actor.csv')

    merged = pd.merge(df_actor,df_tags, on='movieid')
    print(merged.iloc[:5,[1,4]])
    #data = merged.pivot_table(index=['tagid','actorid'] , values = ['tagid'], aggfunc=lambda tagid: len(tagid.unique()))
    merged['COUNTER'] = 1
    group_data = pd.DataFrame(merged.groupby(['actorid', 'tagid'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['actorid'], 'tagid')
    merged['timestamp'] =  pd.to_datetime(merged['timestamp'], format='%Y/%m/%d %H:%M:%S')
    print('min date : ',min(merged['timestamp']),"max date : " ,max(merged['timestamp']))
    min_ts = min(merged['timestamp']).timestamp()
    max_ts = max(merged['timestamp']).timestamp()
    #merged['between 0-1'] = ((merged['timestamp'].timestamp() - min_ts) / (max_ts - min_ts)) * (1 - 0) + 0;
    merged['between 0-1'] = merged.apply(lambda row: ((row['timestamp'].timestamp() - min_ts) / (max_ts - min_ts)) * (1 - 0) + 0, axis=1)
    merged.to_csv(
        '/Users/mj/Documents/Courses/fall-2017/cse-515-mwdb/phase-1/Mwdb_phase1/resources/phase1_dataset/actor_tag.csv',
        index=False)
    print('ts min : ', min_ts, 'ts max : ', max_ts)
    term_vector.to_csv(
        '/Users/mj/Documents/Courses/fall-2017/cse-515-mwdb/phase-1/Mwdb_phase1/resources/phase1_dataset/actor_tag_agg.csv')
    print(group_data.ix[:5])

generate_actor_model(5)