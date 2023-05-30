import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# import seaborn as sn
# import matplotlib.pyplot as plt
# import numpy as np
import pymysql

from .log import log


def get_conn():
    ''' Get connection to the local ema database
    '''
    return pymysql.connect(host='localhost', user='root', password='', db='ema')


def read_data():
    ''' Read in data from ema tables
    Input Params:   N/A
    Output Params:
        success:    whether the database is successfully accessed.
        df:         reward_data table in the local ema database.
    '''
    df = None           # reward_data table, type: pandas dataframe
    success = False
    try:
        db = get_conn()
        cursor = db.cursor()
        # read entire reward data ema table
        query = "SELECT * FROM reward_data"
        num_data = cursor.execute(query)
        log('read_data() read {} lines of reward_data table'.format(num_data))
        cursor_fetch = cursor.fetchall()    # fetch all rows
        df = pd.DataFrame(
            list(cursor_fetch),
            columns=['speakerID', 'empathid', 'TimeSent', 'suid',
                     'TimeReceived', 'Response', 'Question', 'QuestionType',
                     'QuestionName', 'Reactive', 'SentTimes', 'ConnectionError',
                     'Uploaded'])
        success = True
    except Exception as err:
        log('read_data() failed to read reward_data table in proactive_model.py. Error msg: ', str(err))
        db.rollback()
    finally:
        # ensure db closed, exception is raised to upper layer to handle
        db.close()

        return success, df


def read_startDate():
    ''' Start of deployment date
    Input Params:   N/A
    Output Params:
        success:    whether the database is successfully accessed.
        startDate:  what date to start reading data from table.
    '''
    startDate = None        # start date of the deployment. Type: datetime
    success = False
    try:
        db = get_conn()
        cursor = db.cursor()
        # Get start date from the recomm_saved_memory table
        query = "SELECT FirstStartDate FROM recomm_saved_memory"
        num_data = cursor.execute(query)
        log('read_startDate() read {} lines from recomm_saved_memory table'.format(num_data))
        mystoredData = cursor.fetchone()    # fetch only the first row
        # save information from table
        startDate = mystoredData[0]         # startDate is the first column
        # change to datetime
        startDate = datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')
        success = True
    except Exception as err:
        log('read_startDate() failed to read recomm_saved_memory table in proactive_model.py', str(err))
        db.rollback()
    finally:
        # ensure db closed, exception is raised to upper layer to handle
        db.close()

        return success, startDate


def generate_proactive_models():
    '''
        Uses the baseline time and response to train a polynomial regression model
        return a regression model or None if unsuccessful

        This is run once after the baseline period in proactive_recomm()

        and every evening in scheduled_evts()
    '''

    saved_model = None

    try:
        # read the reward_data table and also the start date
        data_success, df = read_data()
        time_success, startdate = read_startDate()

        if (data_success is True) and (time_success is True):
            # change time stored in reward_data to datetime
            df['TimeSent'] = pd.to_datetime(df['TimeSent'], format='%Y-%m-%d %H:%M:%S')
            # use rows only after deployment startdate
            df = df.loc[df['TimeSent'] >= startdate]

            # [baseline phase] get the time and response of the baseline check-in msg
            # response is an integer from 0-10 (likert score)
            baseline_actions_timesent = df.loc[
                df['QuestionName'] == 'baseline:recomm:likertconfirm:1',
                'TimeSent'].tolist()
            baseline_actions_reponse = df.loc[
                df['QuestionName'] == 'baseline:recomm:likertconfirm:1',
                'Response'].tolist()

            # [2nd phase] get the time and response of post-recommendation helpfulness msg
            # response is an integer from 0-10
            post_recomm_time = df.loc[
                df['QuestionName'] == 'daytime:postrecomm:helpfulyes:1',
                'TimeSent'].tolist()
            post_recomm_reward = df.loc[
                df['QuestionName'] == 'daytime:postrecomm:helpfulyes:1',
                'Response'].tolist()

            # subtract 30 minutes from post_recomm_time because
            # a recommendation takes at least 30 minutes pause (?)
            for time_idex in range(0, len(post_recomm_time)):
                post_recomm_time[time_idex] = post_recomm_time[time_idex] - timedelta(minutes=30)

            # combine the send time and response of both baseline and 2nd phases
            timessent_lst = baseline_actions_timesent + post_recomm_time
            angry_helpful_lst = baseline_actions_reponse + post_recomm_reward

            fnl_bline_act_timesent_lst = []
            fnl_bline_act_reponse_lst = []
            for i in range(0, len(timessent_lst)):
                # change reponses to int. not angry: 0.0 to angry: 10.0, no response: -1.0
                angry_helpful_lst[i] = int(float(angry_helpful_lst[i]))
                # remove the datapoint if the response is -1.0 (i.e. no response)
                if angry_helpful_lst[i] != -1.0:
                    fnl_bline_act_reponse_lst.append(angry_helpful_lst[i])
                    # only store the hour of the msg
                    fnl_bline_act_timesent_lst.append(timessent_lst[i].hour)

            # We should have enough data to train the model
            # Otherwise, we will use the saved model (default: None)
            # TODO: the if condition here is duplicated
            if (len(fnl_bline_act_reponse_lst) < 30) or (len(fnl_bline_act_timesent_lst) < 30):
                return saved_model

            data_dict = {'hour': fnl_bline_act_timesent_lst,
                         'angry': fnl_bline_act_reponse_lst}

            # convert dictionary to pandas dataframe
            df_for_model = pd.DataFrame(data_dict, columns=['hour', 'angry'])

            # set independent and dependent variables
            X = df_for_model[['hour']]
            y = df_for_model['angry']

            # test train split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=0
            )

            # use polynomial regression
            poly_reg = PolynomialFeatures(degree=4)
            # transform independent variable to polynomial
            X_poly = poly_reg.fit_transform(X_train)
            # initialize linear regression
            lin_reg = LinearRegression()
            # input the transformed independent variable and train model
            saved_model = lin_reg.fit(X_poly, y_train)
        else:
            log('generate_proactive_models() failed, no data')
    except Exception as err:
        log('Error in generate_proactive_models()', str(err))
    finally:
        # either the model or None
        return saved_model


def get_proactive_prediction(hour, model):
    ''' Predict whether should we send recommendation given a pretrained model
    Input params:
        hour: time (only at the granularity of hour)
        model: pretrained proactive recommendation model, Polynomial Regression
    Output params:
        success:                whether successfully generate the prediction
        send_proact_recomm:     whether send the proactive recommendation
                                0: don't send recomm, 1: send recomm
    '''

    send_proact_recomm = None
    success = False
    threshold_angry = 4         # threshold of angry score to trigger recommendation

    try:
        # check if we have a valid proactive recommendation model
        if model is None:
            return success, send_proact_recomm

        # initilaize polynomial regression for transforming hour
        poly_reg = PolynomialFeatures(degree=4)
        log('Proactive model predicting...')
        # predict angry score of the input hour
        y_pred = model.predict(poly_reg.fit_transform([[hour]]))
        y_pred = float(y_pred[0])

        if y_pred >= threshold_angry:
            send_proact_recomm = 1
        else:
            send_proact_recomm = 0

        log('Proactive model predicts:', send_proact_recomm)
        success = True
    except Exception as err:
        log('Error in get_proactive_prediction', str(err))
    finally:
        # return value could be (False, None) or (True, 0/1)
        return success, send_proact_recomm

# my_model = generate_proactive_models()
# print(my_model)
# print(get_proactive_prediction(17,my_model))

# # plot
# mymodel = np.poly1d(np.polyfit(fnl_bline_act_timesent_lst, fnl_bline_act_reponse_lst, 4))
# myline = np.linspace(0, 23, 100)
# plt.scatter(fnl_bline_act_timesent_lst,fnl_bline_act_reponse_lst)
# plt.plot(myline,mymodel(myline))
# plt.show()

# fnl_bline_act_reponse_lst = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 0, 0, 0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# fnl_bline_act_timesent_lst = [2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

# # change to categorical
# df_for_model['hour'] = df_for_model.hour.astype('category')
# df_for_model['angry'] = df_for_model.angry.astype('category')

# # predict
# Y_pred = lin_reg.predict(poly_reg.fit_transform(X_test))
# since polynomial regression use
# mean_ab_err = metrics.mean_absolute_error(Y_pred,Y_test) #.14
# print(mean_ab_err)
# print(X_test)
# print(Y_pred)
# print(type(Y_pred))
# print(Y_pred[0],type(int(Y_pred[0])))

# #use logistic regression
# logistic_regression = LinearRegression() #LogisticRegression()
# #train the model
# logistic_regression.fit(X_train,y_train)
# y_pred=logistic_regression.predict(np.array([3]).reshape(1,-1))
# # y_pred=logistic_regression.predict(X_test)
# acc = metrics.accuracy_score(y_test,y_pred)
# print(acc)
