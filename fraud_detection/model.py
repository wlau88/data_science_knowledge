'''
Since the data is sensitive, I can only show the script used to produce the model.
'''

import pandas as pd
import numpy as np
from datetime import datetime
import cPickle

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import pdb

from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup


def clean_data(df, train=True):
    '''
    feature engineering and data cleaning
    '''

    # print "cleaning data..."

    df = df.copy()
    
    if train:
        ## adding target Fraud column
        mask = df['acct_type'].isin(['fraudster_event', 'fraudster', 'fraudster_att'])                           
        df['Fraud'] = mask.astype(int)
        del df['acct_type']
    
    df['event_published_Yes'] = (~np.isnan(df['event_published'])).astype(int)
    ## convert unix time to datetime
    date_cols = ['approx_payout_date', 'event_created', 'event_end', 'event_published', 'event_start', 'user_created']
    for d in date_cols:
#         df[d] = df[d].fillna(0).apply(datetime.utcfromtimestamp)
        df[d] = df[d].fillna(0)
    
    ## time differences
    df['u_create-e_create'] = df['event_created'] - df['user_created']
    df['e_create-e_pub'] = df['event_published'] - df['event_created']
    df['e_create-e_start'] = df['event_start'] - df['event_created']
    df['e_start-e_end'] = df['event_end'] - df['event_start']
    df['payout-e_end'] = df['approx_payout_date'] - df['event_end']
    
    ## clean payout
    mask = df['payout_type'] == ''
    df.loc[mask, 'payout_type'] = 'unknown'
    
    df['venue=country'] = (df['venue_country'] == df['country']).astype(int)
    
    df['num_previous_payouts'] = df['previous_payouts'].apply(len)
    
    df['num_ticket_types'] = df['ticket_types'].apply(len)
    
    def sum_tickets(ticket_list):
        if type(ticket_list) == dict:
            return ticket_list['quantity_total']
        quant_tickets = [int(ticket['quantity_total']) for ticket in ticket_list]
        if len(quant_tickets) == 0:
            return 0
        return np.sum( quant_tickets )
    df['quant_tickets_available'] = df['ticket_types'].apply(sum_tickets)
    
    def min_max_price(ticket_list):
        if type(ticket_list) == dict:
            return ticket_list['cost']
        ticket_prices = [int(ticket['cost']) for ticket in ticket_list]
        if len(ticket_prices) == 0:
            return 0
        return max(ticket_prices)
    df['min_max_ticket_price'] = df['ticket_types'].apply(min_max_price)
    
    df['sale_dur_diff'] = df['sale_duration2'] - df['sale_duration']
    
    if train:
        ## payout_type to dummy variables
        df = pd.get_dummies(df, columns=['payout_type'])
        
        df = df.drop(['payout_type_unknown'], axis=1)
    else:
        ach = 0
        check = 0
        if df.loc[0, 'payout_type'] == 'ACH':
            ach = 1
        elif df.loc[0, 'payout_type'] == 'CHECK':
            check = 1

        df['payout_type_ACH'] = ach
        df['payout_type_CHECK'] = check
        del df['payout_type']

    return df
        

def get_numeric_cols(df):
    '''
    removes non-numeric colums
    '''
    text_cols = ['country', 'currency', 'description', 'email_domain', 'listed', 'name', 'org_desc', 
                 'org_name', 'payee_name', 'previous_payouts', 'ticket_types', 'venue_address', 
                 'venue_country', 'venue_name', 'venue_state', 'object_id']
    return df.drop(text_cols, axis=1)


def nlp_description(df):
    '''
    performs tfidf on description columns and fits a random forest
    '''
    # Vectorizer
    y = df['Fraud'].values
    desc_text_series = df.description.apply(lambda x: BeautifulSoup(x).text)
    df['desc_text'] = desc_text_series

    vectorizer = TfidfVectorizer(stop_words='english')
    Xw = vectorizer.fit_transform(df.desc_text)
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, y, 
                                                           test_size=0.3,
                                                           random_state=1)

    n_trees = 100
    clf_rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1).fit(Xw_train.toarray(),
                                                             yw_train)

    pred_prob = clf_rf.predict_proba(Xw_test.toarray())

    df['pred_prob'] = pred_prob


def fit_random_forest(df):
    print "fitting random forest model..."
    y = df['Fraud'].values
    X = df.drop(['Fraud'], axis=1).fillna(0).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                       test_size=0.05,
                                                       random_state=2)
    rf = RandomForestClassifier(300, n_jobs=-1)

    rf.fit(X_train, y_train)
    
    return rf, X_train, y_train, X_test, y_test

def plot_important_features(rf, df):
    '''
    given a fit random forest model (rf), plots the feature feature_importances
    as a bar plot in sorted order
    '''
    features = df.drop(['Fraud'], axis=1).columns
    rf.feature_importances_
    feature_df = pd.DataFrame(rf.feature_importances_, index=features)
    feature_df.sort(0, inplace=True)
    feature_df.plot(kind="barh", figsize=(12,8))
    plt.show()


if __name__ == "__main__":

    ## import training data and make a copy
    df_original = pd.read_json('./data/train_new.json')
    df = df_original.copy()

    df = clean_data(df)

    df = get_numeric_cols(df)

    rf, X_train, y_train, X_test, y_test = fit_random_forest(df)
    
    ## pickle model and final df
    with open('./data/random_forest_model_2.pkl','w') as f:
        cPickle.dump(rf, f, -1) 

    y_pred = rf.predict(X_test)

    print np.c_[rf.predict_proba(X_test), y_test]

    print classification_report(y_test, y_pred)
    print 'Accuracy:', accuracy_score(y_test, y_pred)
    print 'Recall:', recall_score(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)

    
    # with open('./data/random_forest_model.pkl','r') as f:
    #     rf = cPickle.load(f)

    df.to_pickle('./data/clean_df_1.pkl')

    # plot_important_features(rf, df)



