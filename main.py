from fastapi import FastAPI
from Regression import GENERAL_REGRESSION, LINEAR_REGRESSION
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Will add specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#reqest body model for fit_data endpoint
class UniFitDataRequest(BaseModel):
    x: List[float]
    y: List[float]
    deg: int


@app.get("/")
async def root():
    return {"message": "ML ALGORITHMS API running!"}


@app.post("/uni_fit_data")
async def uni_fit_data(data: UniFitDataRequest):
    x = data.x
    y = data.y
    deg = data.deg
    
    try:
        regressor = GENERAL_REGRESSION()  # Initialize the regression model	 
        model, stats = regressor.fit(x, y, deg)  # Fit the model with the provided data

        # Predict
        y_pred, err = regressor.predict(x)
        

        # Build coefficient table
        coeff_list = model.coefficients.tolist()
        coeff_table = {}
        for i, coef in enumerate(coeff_list):
            degree = deg - i
            if degree == 0:
                coeff_table["intercept"] =  np.round(coef, 4)
            else:
                coeff_table[f"X^{degree}"] = np.round(coef, 4)

        return sanitize({
            "coefficients": coeff_table,
            "y_pred": y_pred,
            "err": err,
            "stats": stats
        })
    except Exception as e:
        return {"error": str(e)}


class MultiFitDataRequest(BaseModel):
    X: List[List[float]]  # Accepts 2D data: each inner list is a feature vector
    y: List[float]
    lr: float
    niters: int

@app.post("/multi_fit_data")
async def multi_fit_data(data: MultiFitDataRequest):
    X = data.X
    y = data.y
    lr = data.lr
    niters = data.niters

    try:
        regressor = LINEAR_REGRESSION(lr=lr, n_iters=niters)
        stats = regressor.fit(X, y)
        # print(len(X))
        # for i in range(len(X)):
        #     print(X[i][0])

        # print(X)
        
        y_pred, model, err = regressor.predict(X)

        # Catch model errors
        for item in (model, y_pred, err):
            if isinstance(item, dict) and "error" in item:
                return {"error": item["error"]}


                # Build coefficient table
        coeff_list = model["x-term"]
        coeff_table = {}
        for i, coef in enumerate(coeff_list):
                coeff_table[f"X{i+1}"] = np.round(coef, 4)

        coeff_table["bias"] = np.round(model["bias"], 4)
        
        print("passed run")

        return sanitize({
            "coefficients": coeff_table,
            "y_pred": y_pred,
            "err": err,
            "stats": stats
        })

    except Exception as e:
        return {"error": str(e)}

def sanitize(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None  # or str(obj) if you want "inf", "-inf", "nan"
    elif isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    else:
        return obj

import os           

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)