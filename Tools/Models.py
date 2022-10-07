import pickle
from typing import List
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics, ensemble
from IPython.display import clear_output


class ModelBase:
    """
    Class giving basic methods to Model class (inheritance)
    """

    def __init__(
            self, df_data: pd.DataFrame, feature_names: List[str], name_to_predict: str,
            idx_train: List[int], idx_test: List[int], idx_validation: List[int]):
        """
        :param df_data: dataframe containing the features and the Y. Train and test dataset  would be extract from it
        :param feature_names: dataframe used for model validation
        :param name_to_predict: name of the Ys
        :param idx_test: names of the features used to train
        :param idx_train: names of the features used to train
        :param idx_validation: percentage of df_data used for training
        """
        self.feature_names = feature_names
        self.df_data = df_data.copy().reset_index(drop=True)
        self.name_to_predict = name_to_predict

        self.x_train = self.df_data.iloc[idx_train][self.feature_names].to_numpy()
        self.x_test = self.df_data.iloc[idx_test][self.feature_names].to_numpy()
        self.x_validation = self.df_data.iloc[idx_validation][self.feature_names].to_numpy()

        self.y_train = self.df_data.iloc[idx_train][name_to_predict].values
        self.y_test = self.df_data.iloc[idx_test][name_to_predict].values
        self.y_validation = self.df_data.iloc[idx_validation][name_to_predict].values

        # filled by the inheritor model class
        self.y_train_predict = None
        self.y_test_predict = None
        self.y_validation_predict = None
        self.model = None

    def score(self, validation: bool = False):
        """
        compute the train, test and validation score, which is the distance between the reality and the prediction
        :param validation: if True compute validation score instead of train and test score
        :return: train and test score or validation score
        """
        if validation is True:
            validation_score = np.round(
                metrics.mean_absolute_error(self.y_validation, self.y_validation_predict), 3)
            return validation_score
        else:
            train_score = np.round(metrics.mean_absolute_error(self.y_train, self.y_train_predict), 3)
            test_score = np.round(metrics.mean_absolute_error(self.y_test, self.y_test_predict), 3)
            return train_score, test_score

    def predict(self):
        """
        Compute the prediction of the train, test and validation Y.
        """

        self.y_train_predict = self.model.predict(self.x_train)
        self.y_test_predict = self.model.predict(self.x_test)
        self.y_validation_predict = self.model.predict(self.x_validation)


class LinearRegressionModelClass(ModelBase):
    """
    Logistic regression model class
    """

    def __init__(
            self, df_data: pd.DataFrame, feature_names: List[str], name_to_predict: str,
            idx_train: List[int], idx_test: List[int], idx_validation: List[int]):
        """
        :param df_data: dataframe containing the features and the Y. Train and test dataset  would be extract from it
        :param feature_names: dataframe used for model validation
        :param name_to_predict: name of the Ys
        :param idx_test: names of the features used to train
        :param idx_train: names of the features used to train
        :param idx_validation: percentage of df_data used for training
        """

        self.name = 'LinearRegression'

        ModelBase.__init__(
            self, df_data=df_data, feature_names=feature_names, name_to_predict=name_to_predict,
            idx_test=idx_test, idx_train=idx_train, idx_validation=idx_validation)
        self.model = linear_model.LinearRegression(fit_intercept=False)
        self.model.fit(self.x_train, self.y_train)
        self.predict()


class RandomForestClass(ModelBase):

    def __init__(
            self, max_depth: float, n_estimators: float,
            df_data: pd.DataFrame, name_to_predict: str, feature_names: List[str],
            idx_train: List[int], idx_test: List[int], idx_validation: List[int]):
        """
        :param max_depth: maximum depth of the trees
        :param n_estimators: number of trees
        :param df_data: inherited from ModelBase
        :param name_to_predict: inherited from ModelBase
        :param feature_names: inherited from ModelBase
        :param idx_test: names of the features used to train
        :param idx_train: names of the features used to train
        :param idx_validation: percentage of df_data used for training
        """

        self.name = 'RandomForest'

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        ModelBase.__init__(
            self, df_data=df_data, feature_names=feature_names, name_to_predict=name_to_predict,
            idx_test=idx_test, idx_train=idx_train, idx_validation=idx_validation)

        self.model = ensemble.RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators)
        self.model.fit(self.x_train, self.y_train)
        self.predict()


class HyperparameterExplorationClass:
    """
    Class exploring hyperparameters of a model
    """

    def __init__(
            self, model_class, load_root: str, feature_names_list: List[list],
            para_tuple_list: List[tuple], para_names: List[str],
            df_data: pd.DataFrame, name_to_predict: str,
            idx_train: List[int], idx_test: List[int], idx_validation: List[int],
            load: bool = False):
        """
        :param model_class: model class trained on each hyperparameter
        :param feature_names_list: list of feature name list used from training
        :param para_tuple_list: list of the hyperparameter tuple
        :param para_names: names of the hyperparameters
        :param df_data: dataframe containing the features and the Y. Train and test dataset  would be extract from it
        :param name_to_predict: name of the Ys
        """

        self.model_class = model_class
        self.feature_names_list = feature_names_list
        self.para_tuple_list = para_tuple_list
        self.para_names = para_names
        self.n_para = len(self.para_names)
        self.idx_test = idx_test
        self.idx_train = idx_train
        self.idx_validation = idx_validation

        self.name_to_predict = name_to_predict

        self.df_data = df_data

        self.exploration_dict = None

        self.load = load
        self.load_root = load_root

        self.df_scores = pd.DataFrame(columns=['train_score', 'test_score'])

    def exploration(self):
        """
        Will train model instance on each parameter tuple
        """
        if self.load is True:
            with open(self.load_root+'_dict.pkl', 'br') as f:
                self.exploration_dict = pickle.load(f)

            self.df_scores = pd.read_csv(self.load_root+'_df.csv')
            self.df_scores.columns = ['train_score', 'test_score']
        else:
            self.exploration_dict = dict()

            k = 0
            if self.n_para == 0:
                nb_paras = len(self.feature_names_list)+1

                for feature_names in self.feature_names_list:
                    print('%i/%i' % (k, nb_paras))
                    model = self.model_class(
                        df_data=self.df_data, feature_names=feature_names, name_to_predict=self.name_to_predict,
                        idx_test=self.idx_test, idx_train=self.idx_train, idx_validation=self.idx_validation)

                    model_dict = {k: model.__dict__[k] for k in ['name', 'feature_names', 'name_to_predict']}
                    model_dict['train_score'], model_dict['test_score'] = model.score()
                    model_dict['validation_score'] = model.score(validation=True)

                    self.df_scores.loc[k] = (model_dict['train_score'], model_dict['test_score'])
                    model_dict['id'] = k
                    self.exploration_dict[k] = model_dict
                    k += 1
                    clear_output()
            else:
                nb_paras = len(self.feature_names_list)*len(self.para_tuple_list)+1
                for feature_names in self.feature_names_list:
                    for paras in self.para_tuple_list:
                        print('%i/%i' % (k, nb_paras))
                        para_dict = {self.para_names[i]: paras[i] for i in range(self.n_para)}

                        model = self.model_class(
                            df_data=self.df_data, feature_names=feature_names, name_to_predict=self.name_to_predict,
                            idx_test=self.idx_test, idx_train=self.idx_train, idx_validation=self.idx_validation,
                            **para_dict)

                        model_dict = {k: model.__dict__[k] for k in ['name', 'feature_names', 'name_to_predict']}
                        model_dict['train_score'], model_dict['test_score'] = model.score()
                        model_dict['validation_score'] = model.score(validation=True)

                        model_dict['id'] = k
                        model_dict['para_dict'] = para_dict

                        self.df_scores.loc[k] = (model_dict['train_score'], model_dict['test_score'])

                        self.exploration_dict[k] = model_dict
                        k += 1

                        clear_output()
            with open(self.load_root + '_dict.pkl', 'bw') as pickle_file:
                pickle.dump(self.exploration_dict, pickle_file)
            self.df_scores.to_csv(self.load_root + '_df.csv', index=False)

    def get_best_model_id(self):
        """
        :return: id of the model object with the highest test score
        """
        if self.exploration_dict is None:
            self.exploration()
        best_model_id = self.df_scores.sort_values(['test_score']).index[0]
        return best_model_id

    def get_best_model(self):
        """
        :return: model object with the highest test score
        """
        best_model_id = self.get_best_model_id()
        return self.exploration_dict[best_model_id]

    def print_model(self, i):
        """
        print model characteristics
        :param i: id of the model to print
        """
        model_dict = self.exploration_dict[i]
        print(model_dict['name'])
        print('features: ', model_dict['feature_names'])
        if 'para_dict' in model_dict:
            print('parameters: ', model_dict['para_dict'])
        print()

        print('## Train score:', model_dict['train_score'])

        print('## Test score:', model_dict['test_score'])
        print()

    def print_best_model(self):
        """
        print characteristics of the model with the highest test score
        """

        print('#### Best model ####')

        best_model_id = self.get_best_model_id()
        self.print_model(best_model_id)

    def print_all_models(self):
        """
        print characteristics of all models
        """
        print('#### All models ####')

        for i in range(len(self.exploration_dict)):
            print('##')
            self.print_model(i)

    def print_best_model_validation(self):
        """
        print validation score of the model with the highest test score
        """
        print('#### Best model validation####')
        best_model_id = self.get_best_model_id()
        model_dict = self.exploration_dict[best_model_id]
        print('##', model_dict['name'])

        print('## Validation score:', model_dict['validation_score'])
        print()
        print()
