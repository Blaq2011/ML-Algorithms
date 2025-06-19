import numpy as np

# simple/polynomial regression
class GENERAL_REGRESSION:

    def __init__(self):
        self.coeff = None
        self.y_col = None

    def fit(self, X, y, deg):
        self.x_col = np.array(X)
        self.y_col = np.array(y)
        
        try:
            self.coeff = np.polyfit(self.x_col, self.y_col, deg)
            model = np.poly1d(self.coeff)
            y_i  = np.polyval(self.coeff, self.x_col)
            stats = self.stats(self.y_col, y_i)
            return model, stats
            
        except Exception as e:
            return {"error": str(e) }, {"error": str(e)}
            
    def predict(self,x):
        try:
            y_pred  = np.polyval(self.coeff, np.array(x))
            err = self.y_col - y_pred
            return y_pred , err
           
        except Exception as e:
            return {"error": str(e) }

    def stats(self, y, y_pred):
        sst = np.sum((y-np.mean(y))**2)
        ssr = np.sum((y-y_pred)**2)
        n = len(y)
        k = 1
        
        R_squared = 1 - ssr/sst
        R_squared_adj = 1 - ((1- R_squared)*(n - 1)/ (n-k-1))

        return {"r_squared": np.round(R_squared, 5), "adj_r_squared": np.round(R_squared_adj, 5)}


#Simple/Multuiple Linear regression
class LINEAR_REGRESSION:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weight = None
        self.y = None
        self.n_features = None
        

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        print("X shape:", X.shape)
        print("X", X)
        print("y shape:", y.shape)
        self.y = y
        self.n_samples, self.n_features = X.shape

        self.weight = np.zeros(self.n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias

            #weight update
            dw = (1/self.n_samples) * np.dot(X.T, (y_pred - y)) #weight gradient function
            self.weight = self.weight - self.lr*dw #update

            #bias update
            db = (1/self.n_samples) * np.sum((y_pred -y)) #bias gradient function
            self.bias = self.bias - self.lr*db #update
            
        # print("passed")
        try: 
            stats = self.stats(self.y, y_pred)
            return stats
        except Exception as e:
            return {"error": str(e)}
            
    def predict(self,x):
        x = np.array(x)

        try: 
            y_pred = np.dot(x, self.weight) + self.bias
            # print(self.weight, self.bias)
            coeffs = {"x-term": self.weight.tolist(), "bias":self.bias.item()}
            err = self.y - y_pred
            # print("y_pred", y_pred)
            # print("err", err)
            return y_pred, coeffs, err
        except Exception as e:
            return {"error": str(e)}, {"error": str(e)}, {"error": str(e)}
        
    def stats(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)
        
        sst = np.sum((y - np.mean(y)) ** 2)  # Total Sum of Squares
        ssr = np.sum((y - y_pred) ** 2)      # Residual Sum of Squares
        n = len(y)                           # Number of samples
        k = self.n_features                  # Number of predictors (X features)

        if sst == 0 or n - k - 1 == 0:
            return {
                "r_squared": None,
                "adj_r_squared": None,
                "note": "Insufficient variance or too few samples to calculate adjusted R²."
            }

        r_squared = 1 - ssr / sst
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - k - 1)

        return {
            "r_squared": np.round(r_squared, 5),
            "adj_r_squared": np.round(adj_r_squared, 5)
        }

    # def generate_partial_dependence_data(self, X: np.ndarray, feature_names: list[str], num_points: int = 20):
    #     """
    #     Generate predictions by varying each feature individually across its range,
    #     while holding other features constant at their mean values.
    #     """
    #     X = np.array(X)
    #     means = np.mean(X, axis=0)
    #     min_vals = np.min(X, axis=0)
    #     max_vals = np.max(X, axis=0)

    #     viz_data = {}

    #     for i, name in enumerate(feature_names):
    #         # Vary only one feature
    #         x_range = np.linspace(min_vals[i], max_vals[i], num_points)
    #         X_input = np.tile(means, (num_points, 1))  # shape: (num_points, n_features)
    #         X_input[:, i] = x_range  # modify only the i-th feature

    #         try:
    #             y_pred = np.dot(X_input, self.weight) + self.bias
    #         except Exception as e:
    #             return {"error": str(e)}

    #         viz_data[name] = {
    #             "x_values": x_range.tolist(),
    #             "y_pred": y_pred.tolist(),
    #             "held_constants": {
    #                 feature_names[j]: round(float(means[j]), 4)
    #                 for j in range(len(feature_names)) if j != i
    #             }
    #         }

    #     return viz_data
