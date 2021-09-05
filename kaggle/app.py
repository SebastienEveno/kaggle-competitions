from utils.data import DatasetManager
from models.classification import ClassificationManager

if __name__ == "__main__":

    # load data
    datasetManager = DatasetManager("titanic")
    df_train, df_test = datasetManager.load_data()

    # preprocess training data
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    columns_to_encode = ['Sex']
    columns_dummies = ['Embarked']

    df_train = datasetManager.preprocess_data(df_train, 
        drop_na=True, 
        columns_to_drop=columns_to_drop,
        columns_to_encode=columns_to_encode, 
        columns_dummies=columns_dummies)

    # train the different models on training set
    features = df_train[df_train.columns[1:]]
    target_variable = df_train[df_train.columns[0]]

    classification_manager = ClassificationManager(features, target_variable)
    results = classification_manager.build_all_models()

    # preprocess test data
    columns_to_fillna_with_mean = ['Age', 'Fare']
    passenger_id = df_test['PassengerId']

    df_test = datasetManager.preprocess_data(df_test,
    is_training_data=False, 
    drop_index=False, 
    reshuffle=False, 
    columns_to_drop=columns_to_drop,
    columns_to_fillna_with_mean=columns_to_fillna_with_mean, 
    columns_to_encode=columns_to_encode,
    columns_dummies=columns_dummies)

    # TODO: pick the best model