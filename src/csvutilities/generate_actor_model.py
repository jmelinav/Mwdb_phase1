import pandas as pd
import numpy as np
import os
RANGE_MIN = 1
RANGE_MAX = 2
DATASET_ROOT_PATH = os.path.join(os.getcwd(),'../..','resources/phase1_dataset')
OUT_PUT =[]
ACTOR_TF ='actor_tf.csv'
ACTOR_IDF ='actor_idf.csv'
GENRE_TF ='genre_tf.csv'
GENRE_IDF ='genre_idf.csv'
USER_TF ='user_tf.csv'
USER_IDF ='user_idf.csv'


def readcsv(name):
    return pd.read_csv(name)

def generate_actor_model(actorid):
    print("Generating actor model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_genome_tags = readcsv(DATASET_ROOT_PATH + 'genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags,'timestamp','movieid',
                                  'timestamp_min','timestamp_max','ts_rank')
    df_actor = readcsv(DATASET_ROOT_PATH + 'movie-actor.csv')
    df_actor_with_min_max = scale_down_to_between_two_one(df_actor,'actor_movie_rank'
                                                          ,'movieid','rank_min','rank_max','actor_rank')

    merged = pd.merge(df_actor_with_min_max.iloc[:,[0,1,2,5]],df_tags_with_min_max.iloc[:,[1,2,6]], on='movieid')
    merged['COUNTER'] = 1
    merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])
    merged['ts_rank'] = pd.to_numeric(merged['ts_rank'])
    merged['actor_rank'] = pd.to_numeric(merged['actor_rank'])
    merged['COUNTER'] = merged.COUNTER * merged.ts_rank * merged.actor_rank
    merged=pd.merge(merged,df_genome_tags, left_on='tagid', right_on='tagId')
    group_data = pd.DataFrame(merged.groupby(['actorid', 'tag'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['actorid'], 'tag')

    count_df = generate_idf(term_vector)

    term_vector = term_vector.fillna(0)
    term_vector = generate_TF(term_vector)
    term_vector.to_csv(
        DATASET_ROOT_PATH + ACTOR_TF)
    tf_idf = term_vector.copy(deep=True)
    tf_idf = tf_idf.mul(count_df.ix[2], axis='columns')
    tf_idf.to_csv(DATASET_ROOT_PATH+ACTOR_IDF)

def print_term_vector(subject,model,id):
    filename = get_filename(subject,model)
    tv = readcsv(filename)
    data = get_model_for_id(id, tv)
    print('TF', data.sort_values(by='term_vector', ascending=0))
    # data = get_model_for_id(actor_id, tf_idf)
    # print("tf-idf", data.sort_values(by='term_vector', ascending=0))

def get_filename(subject,model):
    return subject+'_'+model


def generate_genre_vector(genre):
    print("Generating genre model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_genome_tags = readcsv(DATASET_ROOT_PATH + 'genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')
    df_movies = readcsv(DATASET_ROOT_PATH + 'mlmovies.csv')
    df_movies = pd.concat([pd.Series(row['movieid'], row['genres'].split('|'))
               for _, row in df_movies.iterrows()]).reset_index()
    df_movies.columns = ['genres','movieid']
    merged = pd.merge(df_movies, df_tags_with_min_max.iloc[:, [1, 2, 6]],
                      on='movieid')
    merged['COUNTER'] = 1
    merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])
    merged['ts_rank'] = pd.to_numeric(merged['ts_rank'])
    merged['COUNTER'] = merged.COUNTER * merged.ts_rank
    merged = pd.merge(merged, df_genome_tags, left_on='tagid', right_on='tagId')
    group_data = pd.DataFrame(merged.groupby(['genres', 'tag'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['genres'], 'tag')
    count_df = generate_idf(term_vector)

    term_vector = term_vector.fillna(0)
    term_vector = generate_TF(term_vector)
    term_vector.to_csv(
        DATASET_ROOT_PATH + 'genre_tf.csv')
    tf_idf = term_vector.copy(deep=True)
    tf_idf = tf_idf.mul(count_df.ix[2], axis='columns')
    tf_idf.to_csv(DATASET_ROOT_PATH + 'genre_tf_idf.csv')
    genre = 'Thriller'
    data = get_model_for_id(genre, term_vector)
    print('TF', data.sort_values(by='term_vector', ascending=0))
    data = get_model_for_id(genre, tf_idf)
    print("tf-idf", data.sort_values(by='term_vector', ascending=0))

def generate_user_vector(user):
    print("Generating User model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_genome_tags = readcsv(DATASET_ROOT_PATH + 'genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')

    merged = pd.merge(df_tags_with_min_max, df_genome_tags, left_on='tagid', right_on='tagId')
    merged['COUNTER'] = 1
    merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])
    merged['ts_rank'] = pd.to_numeric(merged['ts_rank'])
    merged['COUNTER'] = merged.COUNTER * merged.ts_rank
    group_data = pd.DataFrame(merged.groupby(['userid', 'tag'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['userid'], 'tag')
    count_df = generate_idf(term_vector)
    term_vector = term_vector.fillna(0)
    term_vector = generate_TF(term_vector)
    term_vector.to_csv(
        DATASET_ROOT_PATH + 'user_tf.csv')
    tf_idf = term_vector.copy(deep=True)
    tf_idf = tf_idf.mul(count_df.ix[2], axis='columns')
    tf_idf.to_csv(DATASET_ROOT_PATH + 'user_tf_idf.csv')
    user = 146
    data = get_model_for_id(user, term_vector)
    print('TF', data.sort_values(by='term_vector', ascending=0))
    data = get_model_for_id(user, tf_idf)
    print("tf-idf", data.sort_values(by='term_vector', ascending=0))


def tf_idf_diff(genre1, genre2):
    print("Generating td-idf diff model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_genome_tags = readcsv(DATASET_ROOT_PATH + 'genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')
    df_movies = readcsv(DATASET_ROOT_PATH + 'mlmovies.csv')
    df_movies = pd.concat([pd.Series(row['movieid'], row['genres'].split('|'))
                           for _, row in df_movies.iterrows()]).reset_index()
    df_movies.columns = ['genres', 'movieid']
    df_movies_genre1 = df_movies[df_movies['genres'] == genre1]
    df_movies_genre2 = df_movies[df_movies['genres'] == genre2]

    genre1_tv= get_term_vector(df_genome_tags, df_movies_genre1, df_tags_with_min_max)
    genre2_tv = get_term_vector(df_genome_tags, df_movies_genre2, df_tags_with_min_max)
    genre1_genre_2 = pd.concat([genre1_tv,genre2_tv])
    genre1_genre_2 = genre1_genre_2.drop_duplicates(keep = 'first')
    count_df = generate_idf(genre1_genre_2)
    genre1_tv = genre1_tv.fillna(0)
    genre2_tv = genre2_tv.fillna(0)
    genre1_tv = generate_TF(genre1_tv)
    genre2_tv = generate_TF(genre2_tv)
    genre1_tv = genre1_tv.sum(axis='index').to_frame()
    genre2_tv = genre2_tv.sum(axis='index').to_frame()
    count_df = count_df.T.iloc[:, [2]]
    genre1_tv = pd.merge(genre1_tv, count_df, left_index=True, right_index=True)
    genre2_tv = pd.merge(genre2_tv, count_df, left_index=True, right_index=True)
    genre1_tv.columns = ['tf','idf']
    genre2_tv.columns = ['tf','idf']
    genre1_tv['tf_idf'] = genre1_tv.tf * genre1_tv.idf
    genre2_tv['tf_idf'] = genre2_tv.tf * genre2_tv.idf
    genre1_tv = genre1_tv[['tf_idf']]
    print(genre1_tv.sort_values(by='tf_idf', ascending=0))

def get_term_vector(df_genome_tags, df_movies_genre, df_tags_with_min_max):
    merged = pd.merge(df_movies_genre, df_tags_with_min_max.iloc[:, [1, 2, 6]],
                      on='movieid')
    merged['COUNTER'] = 1
    merged['COUNTER'] = pd.to_numeric(merged['COUNTER'])
    merged['ts_rank'] = pd.to_numeric(merged['ts_rank'])
    merged['COUNTER'] = merged.COUNTER * merged.ts_rank
    merged = pd.merge(merged, df_genome_tags, left_on='tagid', right_on='tagId')
    group_data = pd.DataFrame(merged.groupby(['movieid', 'tag'])['COUNTER'].sum())
    term_vector = group_data.pivot_table('COUNTER', ['movieid'], 'tag')
    return term_vector;

def p1_diff(genre1, genre2):
    print("Generating P1 diff model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_genome_tags = readcsv(DATASET_ROOT_PATH + 'genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')
    df_movies = readcsv(DATASET_ROOT_PATH + 'mlmovies.csv')
    df_movies = pd.concat([pd.Series(row['movieid'], row['genres'].split('|'))
                           for _, row in df_movies.iterrows()]).reset_index()
    df_movies.columns = ['genres', 'movieid']
    df_movies_genre1 = df_movies[df_movies['genres'] == genre1]
    df_movies_genre2 = df_movies[df_movies['genres'] == genre2]
    df_movies_genre1_or_genre2 = df_movies[(df_movies['genres'] == genre1) | (df_movies['genres'] == genre2) ]
    df_movies_genre1_or_genre2 = df_movies_genre1_or_genre2.drop_duplicates(subset='movieid',keep = 'first')
    R=len(df_movies_genre1.index)
    M=len(df_movies_genre1_or_genre2.index)
    genre1_tv = get_term_vector(df_genome_tags, df_movies_genre1, df_tags_with_min_max)
    genre2_tv = get_term_vector(df_genome_tags, df_movies_genre2, df_tags_with_min_max)
    count_df_genre1 = get_column_count(genre1_tv)
    count_df_genre2 = get_column_count(genre2_tv)
    sum_g1 = genre1_tv.sum(axis='index')
    sum_g1_df = sum_g1.to_frame()
    sum_g1_df.columns =['term_count']
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre1, left_index=True, right_index=True, how='outer')
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre2, left_index=True, right_index=True, how='outer')
    sum_g1_df.columns = ['term_count','r_1j','m_1j']
    sum_g1_df['r_1j'] = pd.to_numeric(sum_g1_df['r_1j'])
    sum_g1_df['m_1j'] = pd.to_numeric(sum_g1_df['m_1j'])
    sum_g1_df = clean_data_frame(sum_g1_df)
    sum_g1_df['weight'] = sum_g1_df.apply(lambda row: get_weight(row,M ,R),
                                          axis=1)
    sum_g1_df = sum_g1_df[sum_g1_df['weight'] != 'inf']
    sum_g1_df['weight'] = pd.to_numeric(sum_g1_df['weight'])
    sum_g1_df['diff'] = sum_g1_df.term_count * sum_g1_df.weight
    sum_g1_df = sum_g1_df[['diff']]
    print(sum_g1_df.sort_values(by='diff', ascending=0))

def get_weight(row, M ,R, is_p1_diff = True):
    if is_p1_diff and (row['m_1j'] - row['r_1j']) == 0:
        return 'inf'
    if not is_p1_diff and (M + row['r_1j'] - row['m_1j'] - R) == 0:
        return 0
    #if not is_p1_diff and row['m_1j'] - row['r_1j'] == 0:
     #   return 4000 #will never be zero

    if not is_p1_diff and (R - row['r_1j']) == 0:
        return 'inf'

    return np.log(2+((row['r_1j'] / (R - row['r_1j'])) /
             (np.abs((row['m_1j'] - row['r_1j'])) / (M + row['r_1j'] - row['m_1j'] - R))))*(np.abs(row['r_1j'] / R - ((row['m_1j'] - row['r_1j']) / (M - R))))


def clean_data_frame(data_frame):
    data_frame = data_frame.fillna(0)
    data_frame = data_frame[data_frame.term_count != 0]
    return data_frame


def get_column_count(term_vector):
    count = term_vector.count(axis='index')
    count_df = count.to_frame()
    return count_df


def p2_diff(genre1, genre2):
    print("Generating P2 diff model")
    df_tags = readcsv(DATASET_ROOT_PATH + 'mltags.csv')
    df_genome_tags = readcsv(DATASET_ROOT_PATH + 'genome-tags.csv')
    df_tags['timestamp'] = pd.to_datetime(df_tags['timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_tags_with_min_max = scale_down_ts_to_between_two_one(df_tags, 'timestamp', 'movieid',
                                                            'timestamp_min', 'timestamp_max', 'ts_rank')
    df_movies = readcsv(DATASET_ROOT_PATH + 'mlmovies.csv')
    df_movies = pd.concat([pd.Series(row['movieid'], row['genres'].split('|'))
                           for _, row in df_movies.iterrows()]).reset_index()
    df_movies.columns = ['genres', 'movieid']
    df_movies_genre1 = df_movies[df_movies['genres'] == genre1]
    df_movies_genre2 = df_movies[df_movies['genres'] == genre2]
    df_movies_genre1_or_genre2 = df_movies[(df_movies['genres'] == genre1) | (df_movies['genres'] == genre2)]
    df_movies_genre1_or_genre2 = df_movies_genre1_or_genre2.drop_duplicates(subset='movieid', keep='first')
    R = len(df_movies_genre2.index)
    M = len(df_movies_genre1_or_genre2.index)

    genre1_tv = get_term_vector(df_genome_tags, df_movies_genre1, df_tags_with_min_max)
    genre2_tv = get_term_vector(df_genome_tags, df_movies_genre2, df_tags_with_min_max)
    genre1_genre_2 = pd.concat([genre1_tv, genre2_tv])
    genre1_genre_2 = genre1_genre_2.drop_duplicates(keep='first')
    count_df_genre1_and_genre2 = get_column_count(genre1_genre_2)
    count_df_genre1_and_genre2.columns = ['m_1j']
    count_df_genre1_and_genre2['m_1j'] = len(df_movies_genre1_or_genre2) - count_df_genre1_and_genre2.m_1j
    count_df_genre2 = get_column_count(genre2_tv)
    count_df_genre2.columns = ['r_1j']
    count_df_genre2['r_1j'] = len(df_movies_genre2) - count_df_genre2.r_1j
    sum_g1 = genre1_tv.sum(axis='index')
    sum_g1_df = sum_g1.to_frame()
    sum_g1_df.columns = ['term_count']
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre2, left_index=True, right_index=True, how='outer')
    sum_g1_df = pd.merge(sum_g1_df, count_df_genre1_and_genre2, left_index=True, right_index=True, how='outer')
    sum_g1_df['r_1j'] = pd.to_numeric(sum_g1_df['r_1j'])
    sum_g1_df['m_1j'] = pd.to_numeric(sum_g1_df['m_1j'])
    sum_g1_df[['r_1j']] = sum_g1_df[['r_1j']].fillna(value=R)
    sum_g1_df = clean_data_frame(sum_g1_df)
    sum_g1_df['weight'] = sum_g1_df.apply(lambda row: get_weight(row, M, R, False),
                                          axis=1)
    with_out_inf = sum_g1_df[sum_g1_df['weight'] != 'inf']
    max = with_out_inf['weight'].max()
    sum_g1_df.ix[sum_g1_df['weight'] == 'inf', ['weight']] = [max+1]
    sum_g1_df['weight'] = pd.to_numeric(sum_g1_df['weight'])
    sum_g1_df['diff'] = sum_g1_df.term_count * sum_g1_df.weight
    sum_g1_df =  sum_g1_df[sum_g1_df['weight'] != 0]
    sum_g1_df = sum_g1_df[['diff']]
    print(sum_g1_df.sort_values(by='diff', ascending=0))


def get_model_for_id(id, term_vector):
    term_vector.index.name = 'label'
    term_vector.reset_index(inplace=True)
    tv = term_vector[term_vector['label'] == id]
    data = tv.transpose().iloc[1:]
    data.columns = ['term_vector']
    data = data[(data.T != 0).any()]
    return data


def generate_TF(term_vector):
    term_vector['total_freq'] = term_vector.sum(axis=1)
    columns = term_vector.columns.tolist()
    columns.remove('total_freq')
    term_vector = term_vector[columns].div(term_vector.total_freq, axis=0)
    return term_vector


def generate_idf(term_vector):
    count = term_vector.count(axis='index')
    count_df = count.to_frame()
    count_df['total_docs'] = len(term_vector.index)
    count_df.columns = ['tag_count', 'total_docs']
    count_df['total_docs'] = pd.to_numeric(count_df['total_docs'])
    count_df['tag_count'] = pd.to_numeric(count_df['tag_count'])
    count_df['idf'] = np.log(count_df.total_docs / count_df.tag_count)
    count_df = count_df.T
    return count_df


def scale_down_to_between_two_one(df, field, group_field, new_min_label
                                  , new_max_label, new_field_label):
    min_df = df[field].groupby(df[group_field]).min()
    min_df = min_df.to_frame()
    min_df = min_df.reset_index(level=[group_field])
    max_df = df[field].groupby(df[group_field]).max()
    max_df = max_df.to_frame()
    max_df = max_df.reset_index(level=[group_field])
    min_df.columns = [group_field, new_min_label]
    max_df.columns = [group_field, new_max_label]
    min_max_df = pd.merge(min_df, max_df, on=group_field)
    df_with_min_max = pd.merge(df, min_max_df, on=group_field)
    df_with_min_max[new_field_label] = df_with_min_max.apply(
        lambda row: ((row[field] - row[new_min_label])
                     / (row[new_max_label] - row[new_min_label])
                     if row[new_max_label] != row[new_min_label] else 1)
                    * (RANGE_MAX - RANGE_MIN) + RANGE_MIN, axis=1)
    return df_with_min_max

def scale_down_ts_to_between_two_one(df, field, group_field, new_min_label
                                  , new_max_label, new_field_label):
    min_df = df[field].groupby(df[group_field]).min()
    min_df = min_df.to_frame()
    min_df = min_df.reset_index(level=[group_field])
    max_df = df[field].groupby(df[group_field]).max()
    max_df = max_df.to_frame()
    max_df = max_df.reset_index(level=[group_field])
    min_df.columns = [group_field, new_min_label]
    max_df.columns = [group_field, new_max_label]
    min_max_df = pd.merge(min_df, max_df, on=group_field)
    df_with_min_max = pd.merge(df, min_max_df, on=group_field)
    df_with_min_max[new_field_label] = df_with_min_max.apply(
        lambda row: ((row[field].timestamp() - row[new_min_label].timestamp())
                     / (row[new_max_label].timestamp() - row[new_min_label].timestamp())
                     if row[new_max_label].timestamp() != row[new_min_label].timestamp() else 1)
                    * (RANGE_MAX - RANGE_MIN) + RANGE_MIN, axis=1)
    return df_with_min_max



#generate_actor_model(5)
#generate_genre_vector('Comedy')
#generate_user_vector(146)
#tf_idf_diff('Comedy','Action')
#p1_diff('Comedy','Action')
#p2_diff('Comedy','Action')

def main(argv):
    print(argv)
    if argv[1] == 'diff':
        tf_idf_diff('Comedy', 'Action')
    else:
        print_term_vector(argv[1],argv[2],int(argv[3]))

import sys
if __name__ == '__main__':
    main(sys.argv)
