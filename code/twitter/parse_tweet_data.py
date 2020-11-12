# ______________________________________________________________________________________________________________________
# this code is **************************************(U) UNCLASSIFIED***************************************************
# ______________________________________________________________________________________________________________________
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------------------
# program name:         parse_tweet_data
# major version:        1.1
# program purpose:      This program converts the updated ~10M tweet dataset and uses various functions to parse,
#                       inventory, and prep the data for topic modelling and metadata analysis.
# python version:       3.6
#
# Author:               Emily Parrish
# major version created:20200602
# last modification:    20200602 Created all major functions for inventoring and parsing
#                       20200612 Adjusted directory structure for outputs

# ----------------------------------------------------------------------------------------------------------------------

import os
import sys
import string
import csv
import operator
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import collections
import re
from helpers import *

# global paths
path = r'E:\Twitter\Russia\Russia_1906'
path_split = path.split('\\')
# current date
today = '20' + datetime.now().strftime('%m%d')

def convert_source(infile):
    ''' Imports a CSV file into Python Pandas and outputs a Pandas Data Frame Object.  It then saves this object to a
     pkl file for more efficient import of other processes.

     Inputs: *.csv file
     Outputs: *.pkl file in the input directory
     '''

    # construct absolute path
    filepath = os.path.join(path, infile)

    # import *.csv file to data frame
    df = pd.read_csv(filepath)

    outfile = path_split[1] + '_' + path_split[3] + '_' + today + '.pkl'

    # save data frame to *.pkl file of same name
    ext_path = os.path.join(path, '1_DataFrames')
    df.to_pickle(os.path.join(ext_path, outfile))

def convert_comb(files):
    '''Takes a list of *.csv file inputs and creates a single data frames from each of them.

    Inputs: Input *.csv files to be combined into a single data frame
    Outputs: A data frame for each input and a single output combined data frame
    '''

    dfs = list()
    for file in files:
        ext_path = os.path.join(path, '1_DataFrames')
        df = pd.read_pickle(os.path.join(ext_path, file))
        dfs.append(df)

    new_df = merge_df(dfs)

    outfile = path_split[1] + '_' + path_split[3] + '_' + today + '.pkl'

    # save new data frame under *.pkl file
    ext_path = os.path.join(path, '1_DataFrames')
    new_df.to_pickle(os.path.join(ext_path, outfile))

    return new_df

def sort_df(df, field='tweet_time'):
    ''' Takes a Pandas data frame of Twitter data and sorts by a specified field to prep for data frame parsing steps. It
    also adds a column called "unique_id_ida" with formatted ID numbers for each tweet.

    Inputs: Pandas data frame imported from *.csv or *.pkl file
            Field in data frame to be sorted (OPTIONAL: tweet_time, aka sort by date, is the default)
    Outputs: *.pkl file in the input directory with "sorted" label sorted, containing additional column "unique_id_ida"
                with 7-digit ID numbers beginning with 0000000 (i.e. 1234567, 0000134)
    '''

    # turn any time fields into datetime objects
    if field == 'tweet_time':
        df['tweet_time'] = pd.to_datetime(df.tweet_time)

    # sort data frame by field column
    df = df.sort_values(by=field)

    # generate list of unique ID numbers with 7-digits (leading zeros for smaller numbers)
    in_list = list()
    for i in range(0, len(df.index)):
        i = str(i)
        while len(i) != 7:
            i = "0" + i
        in_list.append(i)

    # add column ID numbers to data frame
    df['unique_id_ida'] = np.array(in_list)

    outfile = path_split[1] + '_' + path_split[3] + '_sorted_' + today + '.pkl'

    # save new data frame under *.pkl file
    ext_path = os.path.join(path, '1_DataFrames')
    df.to_pickle(os.path.join(ext_path, outfile))

    return df

def split_df(df, num=30):
    '''Takes an input data frame and creates inventories for that data frame, sorted by date.  First automatically calls
     the sort function to sort the data frame by the default (date).

     Inputs: Pandas data frame imported from *.csv or *.pkl file
             Number of inventories to split into.  Users discresction depending on size of the data set.
     Outputs: Directory of inventories containing Tweet content and metadata.  These inventories are divided in such a
            way to keep them between 130 and 160 MB, labelled with alphabetical characters to order them.  They are
            sorted by date with the ranges of dates in each inventory in the file name
            (i.e. AA_Twitter10M_090509_130214.csv)
     '''

    df = sort_df(df)

    alphabets = string.ascii_lowercase
    a_list = list()
    for i in alphabets:
        for j in alphabets:
            a_list.append(i.upper() + j.upper())

    # splits data set into 30 different data frames of equal size, which will each represent an individual inventory.
    df_split = np.array_split(df, num) 
    subpath = os.path.join(path, '2_Inventories')

    alpha_index = 0
    last_i = len(df_split) - 1

    for item in df_split:

        df_sub = pd.concat([item.head(1), item.tail(1)])
        date_bounds = pd.Series(df_sub['tweet_time'].tolist())
        date_bounds_format = (date_bounds.dt.strftime('%Y%m%d')).tolist()

        to_file = item[item.tweet_time.dt.strftime('%Y%m%d') != date_bounds[1].strftime('%Y%m%d')]

        if alpha_index == 0:
            comb_df = to_file
        elif alpha_index == last_i:
            comb_df = pd.concat([extra_rows, item], axis=0)
        else:
            comb_df = pd.concat([extra_rows, to_file], axis=0)

        prevdate = str(int(date_bounds_format[1]) - 1)
        filename = a_list[alpha_index] + '_' + path_split[1] + '_' + path_split[3] + '_' + date_bounds_format[0][2:] + '_' + prevdate[2:] + '.csv'
        print(filename)

        filepath = os.path.join(subpath,filename)
        comb_df.to_csv(filepath)

        extra_rows = item[item.tweet_time.dt.strftime('%Y%m%d') == date_bounds[1].strftime('%Y%m%d')]
        alpha_index += 1


def get_lang(df, lang):
    '''Takes an input data frame and generates a data frame with only a specific language's tweets (user specified).

    Inputs: Pandas data frame imported from *.csv or *.pkl file
               Language code for language of interest
    Outputs: Pandas data frame with a subset of tweets from that specific language
    '''
    lang_df = df.loc[df['tweet_language'] == lang]

    outfile = path_split[1] + '_' + path_split[3] + '_sorted_' + lang + '_' + today + '.pkl'

    # save new data frame under *.pkl file
    ext_path = os.path.join(path, '1_DataFrames')
    lang_df.to_pickle(os.path.join(ext_path, outfile))

    return lang_df


def strip_formatting(df, lim, lang='allLang'):
    '''Takes an imput data frame and removes emojis, punctuation, HTML entities like &amp;, links, handles, and emojis.
    Then based on a user specified character limit, it removes the tweets that are below that limit and returns the sub-
    data frame.

    Inputs: Pandas data frame imported from *.csv or *.pkl file
            Character limit for parsing after strip functionality implemented
            Language label user provides for file naming (if it is a data frame describing a particular language)
    Outputs: Pandas data frame with subset of tweets that satisfied the character limit after removind entities
             of interest
    '''

    tweets = df['tweet_text'].to_list()

    edit_tweets = list()
    include = list()

    for tweet in tweets:

        strip_tweet = strip_accounts(remove_punctuation(strip_html_entities(strip_links(strip_emoji(tweet)))))

        edit_tweets.append(strip_tweet)

        if is_length(strip_tweet, lim):
            include.append('1')
        else:
            include.append('0')

    df['stripped_tweet'] = edit_tweets
    df['tweet_length'] = df['tweet_text'].str.len()
    df['include_topic_model'] = include
    df['stripped_tweet_length'] = df['include_topic_model'].str.len()

    sub_df = df.loc[df['include_topic_model'] == '1']

    outfile = path_split[1] + '_' + path_split[3] + '_sorted_strip_' + lang + '_' + today + '.pkl'

    # save new data frame under *.pkl file
    ext_path = os.path.join(path, '1_DataFrames')
    sub_df.to_pickle(os.path.join(ext_path, outfile))

    return sub_df

def extract_content(df, label='All_Languages'):
    '''Takes an input data frame and extracts the individual Tweets and places it in chronological directories
    incremented by intervals based on a month.

    Inputs: Pandas data frame imported from *.csv or *.pkl file
            Language label user provides for file naming (if it is a data frame describing a particular language)
    Outputs: Directories of binned tweets by month.  Each tweet is in its own text file with the stripped tweet content
            only in the file. Each file is named accordingly like the following example:
    '''

    date_bounds = pd.Series(df['tweet_time'].tolist())
    date_bounds_ymd = (date_bounds.dt.strftime('%Y%m%d')).tolist()
    date_bounds_hms = (date_bounds.dt.strftime('%H%M')).tolist()
    content = pd.Series(df['stripped_tweet'].tolist())
    unid = pd.Series(df['unique_id_ida'].tolist())
    print('Total Files to process: ' + str(len(date_bounds_ymd)))

    parentdir = os.path.join(path, label)
    os.mkdir(parentdir)

    for i in range(0, len(date_bounds_ymd)):
        dir = date_bounds_ymd[i][:4] + '-' + date_bounds_ymd[i][4:6]
        fulldir = os.path.join(parentdir, dir)
        filename = str(unid[i]) + '_' + path_split[1] + '_' + path_split[3] + '_' + date_bounds_ymd[i][2:] + '_' + date_bounds_hms[i][:4] + '.txt'

        outpath = os.path.join(fulldir, filename)

        if os.path.exists(outpath):
            pass
        else:
            if os.path.isdir(fulldir):
                pass
            else:
                os.mkdir(fulldir)

            if int(i) % 10000 == 0:

                print('Files up to ' + str(i) + ' processed.')

            f = open(outpath, 'w', encoding='utf-8')
            f.write(content[i])
            f.close()

def generate_freq(df):
    '''Takes an input data frame and generates a histogram of number of tweets binned by month.

    Inputs: Pandas data frame imported from *.csv or *.pkl file
            Input parameter called "increment", which determined by what time interval the tweets are organized
    Outputs: Histogram
    '''

    date_bounds = pd.Series(df['tweet_time'].tolist())
    date_bounds_ym = (date_bounds.dt.strftime('%Y-%m')).tolist()
    df['date_md'] = np.array(date_bounds_ym)

    sort = df.sort_values(by=['date_md'])

    frq = sort['date_md'].value_counts().to_dict()
    frq_df = sort['date_md'].value_counts()

    od = collections.OrderedDict(sorted(frq.items()))
    rf_dates = list()

    for item in list(od.keys()):
        date_rf = date_reformat(item)
        rf_dates.append(date_rf)

    data = {"Date": rf_dates, "Freq": list(od.values())}
    graph_frame = pd.dataframe(data=data)
    frq_df.to_csv(os.path.join(path, 'tweet_freq_' + today + '.csv'))

    ax = graph_frame.plot.bar(x="Date", y="Freq", rot=45)

    plt.show()


def main():
    print('Start time: ' + str(datetime.now()))

    infile = 'Twitter_Russia_1906_sorted_strip_en_200929.pkl'
    inpath = os.path.join(path, '1_DataFrames')
    infilepath = os.path.join(inpath, infile)

    stripped_en = pd.read_pickle(infilepath)
    print(stripped_en.head()['unique_id_ida'])

    # sorted = sort_df(df)
    # split_df(sorted, 1)
    #
    # en_df = get_lang(sorted, 'en')
    # stripped_en = strip_formatting(en_df, 10, 'en')
    #
    # ru_df = get_lang(sorted, 'ru')
    # stripped_ru = strip_formatting(ru_df, 12, 'ru')
    #
    # zh_df = get_lang(sorted, 'zh')
    # stripped_zh = strip_formatting(zh_df, 2, 'zh')

    extract_content(stripped_en, 'English')

    print('End time: ' + str(datetime.now()))

if __name__ == '__main__':
    main()

# this code is **************************************(U) UNCLASSIFIED***************************************************