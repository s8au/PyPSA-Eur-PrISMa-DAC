import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def linreg(var):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    df_ct = df.set_index("country")
    df_test = df_ct.loc[["DK","AL","ES","NO"]]

    X = np.column_stack((df_test["T_feed"], df_test["RH"]))
    model = LinearRegression().fit(X, df_test[var])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X)
    print("poly features",
           poly_features.shape, df_test[var].shape)
    model = LinearRegression().fit(poly_features, df_test[var])

    # Coefficients and intercept
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    def insamplersqur():
        X_full = df_ct[["T_feed", "RH"]].values      # shape (n_samples, 2)
        y_full = df_ct[var].values    # shape (n_samples,)
        X_full_poly = poly.transform(X_full)

        r2_in_sample = r2_score(y_full, model.predict(X_full_poly)) 
        print(f"In-sample R²: {r2_in_sample:.3f}")
    insamplersqur()
    

    def crossval(var):
        from sklearn.model_selection import cross_val_score

        X = df_ct[["T_feed", "RH"]].values
        y = df_ct["compression-electricity-input"].values + df_ct["electricity-input"].values 

        cv_r2 = cross_val_score(LinearRegression(), X, y,
                                cv=5, scoring="r2")   # 5-fold CV
        print("Fold R²s :", cv_r2)
        print("Mean R²  :", cv_r2.mean())

    def testrsqu(var):
        from sklearn.model_selection import train_test_split
        

        X = df_ct[["T_feed", "RH"]].values
        y = df_ct[var].values 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, shuffle=True)

        model = LinearRegression().fit(X_train, y_train)

        print("Test R²:", model.score(X_test, y_test))          # same as r2_score(y_test, model.predict(X_test))
        print("MAE     :", mean_absolute_error(y_test, model.predict(X_test)))
        print("RMSE    :", mean_squared_error(y_test, model.predict(X_test)))
        def predicted_plot(var):
            y_pred = model.predict(X_test)
            plt.figure()
            plt.scatter(y_test, y_pred)
            plt.xlabel(f"Actual {var}")
            plt.ylabel(f"Predicted {var}")
            plt.title(f"Predicted vs Actual")
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.grid(True)
            plt.figtext(
                    0.05,
                    -0.1,
                    f" R²:{model.score(X_test, y_test)}""\n"
                    f"{model.intercept_:9.4f}+{model.coef_[0]:9.4f}T+{model.coef_[1]:9.4f}RH",
                )
            plt.savefig(f"{folder}{material}pred{var}.png", bbox_inches="tight")
        predicted_plot(var)
    testrsqu(var)

def scatter(var): 
    fig, ax = plt.subplots()
    scatter = plt.scatter(df["T_feed"], df[var], c=df["RH"])
    legend1 = ax.legend(*scatter.legend_elements(num=6),
                        loc="best", title="relative humidity [%]")
    ax.add_artist(legend1)
    ax.set_ylabel(f"{var} consumption [MWh]")
    ax.set_xlabel("Temperature [degC]")
    fig.savefig(f"{folder}{material}{var}scatterplot_temp.png")

    fig, ax = plt.subplots()
    scatter = plt.scatter(df["RH"], df[var] , c=df["T_feed"])
    legend1 = ax.legend(*scatter.legend_elements(num=6),
                        loc="best", title="Temperature [degC]")
    ax.add_artist(legend1)
    ax.set_ylabel(f"{var} consumption [MWh]")
    ax.set_xlabel("relative humidity [%]")
    fig.savefig(f"{folder}{material}{var}scatterplot_hum.png")

material = "Lewatit"
folder = "mofs_lewatit/"
df = pd.read_csv("mofs_lewatit.csv").set_index("scenario")
df["country"] = df.index.str.split("_").str[2]
df["temp"] = df.index.str.split("_").str[5]
df = df.sort_values(["country","temp"])
df["total-electricity-input"] = df["compression-electricity-input"]+df["electricity-input"]
depV = ["compression-electricity-input","electricity-input", "total-electricity-input", "heat-input"]
for var in depV:
    scatter(var)
    linreg(var)

    