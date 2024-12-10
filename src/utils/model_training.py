from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_evaluate_conditions(conditions, x_backtest, y_backtest, x_after, y_after):
    accuracies = []
    for condition in conditions:
        condition = "not (" + condition + ")"
        filtered_x_backtest = x_backtest.query(condition)
        filtered_y_backtest = y_backtest[filtered_x_backtest.index]
        model = LogisticRegression(max_iter=1000)
        model.fit(filtered_x_backtest, filtered_y_backtest)
        y_pred = model.predict(x_after)
        accuracies.append(accuracy_score(y_after, y_pred))
    return accuracies
