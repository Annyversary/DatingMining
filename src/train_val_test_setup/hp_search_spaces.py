from sklearn.linear_model import LogisticRegression
import numpy as np

search_spaces = {
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'param_distributions': [        
            {
                'solver': ['liblinear'],
                'penalty': ['l1', 'l2'],
                'C': np.logspace(-3, 3, 10)
            },        
            {
                'solver': ['lbfgs'],
                'penalty': ['l2'],
                'C': np.logspace(-3, 3, 10)
            },
            {
                'solver': ['saga'],
                'penalty': ['l1', 'l2'],
                'C': np.logspace(-3, 3, 10)
            }
        ]
    }
}
