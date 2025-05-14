from flask import Flask, render_template, request
import numpy as np
from scipy.stats import binom, poisson, norm, expon, t, ttest_1samp, ttest_ind, chi2_contingency, linregress, pearsonr
import statistics
import matplotlib.pyplot as plt
import uuid
import os
import seaborn as sns

from database import init_db, insert_history


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/basic-stats", methods=["GET", "POST"])
def basic_stats():
    if request.method == "POST":
        raw_data = request.form["data"]
        try:
            numbers = sorted([float(x) for x in raw_data.split(",")])
        except ValueError:
            return render_template("basic_stats.html", error="Invalid input. Use comma-separated numbers.")

        mean = round(np.mean(numbers), 2)
        median = round(np.median(numbers), 2)
        try:
            mode = statistics.mode(numbers)
        except statistics.StatisticsError:
            mode = "No unique mode"
        std = round(np.std(numbers), 2)
        var = round(np.var(numbers), 2)

        insert_history(raw_data, mean, median, str(mode), std, var)

        data_range = round(max(numbers) - min(numbers), 2)
        q1 = round(np.percentile(numbers, 25), 2)
        q3 = round(np.percentile(numbers, 75), 2)
        iqr = round(q3 - q1, 2)

        # Generate unique ID for this session
        plot_id = str(uuid.uuid4())

        # Create boxplot
        plt.figure(figsize=(4, 6))
        plt.boxplot(numbers, vert=True)
        plt.title("Boxplot of Input Data")
        plt.ylabel("Value")
        boxplot_path = f"static/boxplot_{plot_id}.png"
        plt.savefig(boxplot_path)
        plt.close()

        # Create histogram
        plt.figure(figsize=(6, 4))
        plt.hist(numbers, bins='auto', edgecolor='black', color='skyblue')
        plt.title("Histogram of Input Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        hist_path = f"static/hist_{plot_id}.png"
        plt.savefig(hist_path)
        plt.close()

        return render_template("basic_stats_result.html",
                               numbers=numbers,
                               mean=mean,
                               median=median,
                               mode=mode,
                               std=std,
                               var=var,
                               data_range=data_range,
                               q1=q1,
                               q3=q3,
                               iqr=iqr,
                               boxplot=boxplot_path,
                               histplot=hist_path)

    return render_template("basic_stats.html")

@app.route("/distributions", methods=["GET", "POST"])
def distributions():
    if request.method == "POST":
        dist_type = request.form["dist_type"]
        x = float(request.form["x"])

        result = {}
        if dist_type == "binomial":
            n = int(request.form["n"])
            p = float(request.form["p"])
            result["pdf"] = round(binom.pmf(x, n, p), 5)
            result["cdf"] = round(binom.cdf(x, n, p), 5)

        elif dist_type == "poisson":
            lam = float(request.form["lam"])
            result["pdf"] = round(poisson.pmf(x, lam), 5)
            result["cdf"] = round(poisson.cdf(x, lam), 5)

        elif dist_type == "normal":
            mu = float(request.form["mu"])
            sigma = float(request.form["sigma"])
            result["pdf"] = round(norm.pdf(x, mu, sigma), 5)
            result["cdf"] = round(norm.cdf(x, mu, sigma), 5)

        elif dist_type == "exponential":
            scale = float(request.form["scale"])
            result["pdf"] = round(expon.pdf(x, scale=scale), 5)
            result["cdf"] = round(expon.cdf(x, scale=scale), 5)

        plot_id = str(uuid.uuid4())
        x_values = None
        y_values = None

        # Generate plot based on distribution type
        if dist_type == "binomial":
            x_values = np.arange(0, int(request.form["n"]) + 1)
            n = int(request.form["n"])
            p = float(request.form["p"])
            y_values = binom.pmf(x_values, n, p)
            plt.bar(x_values, y_values, color='skyblue')
            plt.title(f"Binomial PMF (n={n}, p={p})")

        elif dist_type == "poisson":
            lam = float(request.form["lam"])
            x_values = np.arange(0, 30)
            y_values = poisson.pmf(x_values, lam)
            plt.bar(x_values, y_values, color='lightgreen')
            plt.title(f"Poisson PMF (λ={lam})")

        elif dist_type == "normal":
            mu = float(request.form["mu"])
            sigma = float(request.form["sigma"])
            x_values = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
            y_values = norm.pdf(x_values, mu, sigma)
            plt.plot(x_values, y_values, color='orange')
            plt.fill_between(x_values, y_values, alpha=0.2)
            plt.title(f"Normal PDF (μ={mu}, σ={sigma})")

        elif dist_type == "exponential":
            scale = float(request.form["scale"])
            x_values = np.linspace(0, 5 * scale, 100)
            y_values = expon.pdf(x_values, scale=scale)
            plt.plot(x_values, y_values, color='purple')
            plt.fill_between(x_values, y_values, alpha=0.2)
            plt.title(f"Exponential PDF (scale={scale})")

        plt.xlabel("x")
        plt.ylabel("Probability")
        plt.grid(True)

        os.makedirs("static", exist_ok=True)
        plot_path = f"static/dist_plot_{plot_id}.png"
        plt.savefig(plot_path)
        plt.close()

        return render_template("distributions_result.html",
                               dist=dist_type, x=x, result=result,
                               plot_path=plot_path)

    return render_template("distributions.html")

@app.route("/inferential")
def inferential():
    return render_template("inferential.html")

@app.route("/confidence-interval", methods=["GET", "POST"])
def confidence_interval():
    if request.method == "POST":
        try:
            method = request.form["method"]
            mean = float(request.form["mean"])
            std_dev = float(request.form["std_dev"])
            n = int(request.form["n"])
            confidence = float(request.form["confidence"])

            alpha = 1 - (confidence / 100)

            if method == "z":
                z_value = norm.ppf(1 - alpha / 2)
                margin = z_value * (std_dev / (n ** 0.5))
            elif method == "t":
                t_value = t.ppf(1 - alpha / 2, df=n - 1)
                margin = t_value * (std_dev / (n ** 0.5))
            else:
                return "Invalid method"

            lower = round(mean - margin, 4)
            upper = round(mean + margin, 4)

            # Create bell curve plot
            plot_id = str(uuid.uuid4())
            x_vals = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 500)
            y_vals = norm.pdf(x_vals, mean, std_dev)

            plt.figure(figsize=(8, 4))
            plt.plot(x_vals, y_vals, label="Normal Distribution")
            plt.fill_between(x_vals, y_vals, where=(x_vals >= lower) & (x_vals <= upper),
                             color='skyblue', alpha=0.6, label="Confidence Interval")

            plt.axvline(lower, color='red', linestyle='--', label=f"Lower = {lower}")
            plt.axvline(upper, color='green', linestyle='--', label=f"Upper = {upper}")
            plt.title(f"{confidence}% Confidence Interval")
            plt.xlabel("x")
            plt.ylabel("Probability Density")
            plt.legend()
            plt.tight_layout()

            # Save plot
            os.makedirs("static", exist_ok=True)
            plot_path = f"static/conf_interval_{plot_id}.png"
            plt.savefig(plot_path)
            plt.close()

            return render_template("confidence_interval_result.html",
                                   method=method,
                                   confidence=confidence,
                                   mean=mean,
                                   std_dev=std_dev,
                                   n=n,
                                   lower=lower,
                                   upper=upper,
                                   plot_path=plot_path)

        except Exception as e:
            return f"Error: {e}"

    return render_template("confidence_interval.html")

@app.route("/hypothesis-test", methods=["GET", "POST"])
def hypothesis_test():
    if request.method == "POST":
        test_type = request.form["test_type"]
        alpha = float(request.form["alpha"])

        try:
            if test_type == "one":
                raw_sample = request.form["sample1"]
                mu_0 = float(request.form["mu_0"])
                sample = [float(x.strip()) for x in raw_sample.split(",")]
                t_stat, p_val = ttest_1samp(sample, mu_0)

            elif test_type == "two":
                raw_1 = request.form["sample1"]
                raw_2 = request.form["sample2"]
                sample1 = [float(x.strip()) for x in raw_1.split(",")]
                sample2 = [float(x.strip()) for x in raw_2.split(",")]
                t_stat, p_val = ttest_ind(sample1, sample2, equal_var=False)

            else:
                return "Invalid test type"

            decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"

            x_vals = np.linspace(-5, 5, 500)
            y_vals = t.pdf(x_vals, df=20)  # df=20 as a generic curve

            plt.figure(figsize=(8, 4))
            plt.plot(x_vals, y_vals, label="t-distribution")
            plt.axvline(t_stat, color='red', linestyle='--', label=f"t = {round(t_stat, 2)}")
            plt.fill_between(x_vals, y_vals, where=(x_vals <= -abs(t_stat)) | (x_vals >= abs(t_stat)),
                             color='lightcoral', alpha=0.3, label="Rejection region")

            plt.title("t-Test Visualization")
            plt.xlabel("t")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()

            plot_id = str(uuid.uuid4())
            t_plot_path = f"static/t_plot_{plot_id}.png"
            plt.savefig(t_plot_path)
            plt.close()

            return render_template("hypothesis_test_result.html",
                                   test_type=test_type,
                                   t_stat=round(t_stat, 4),
                                   p_val=round(p_val, 4),
                                   alpha=alpha,
                                   decision=decision,
                                   t_plot_path=t_plot_path)


        except Exception as e:
            return f"Error: {e}"

    return render_template("hypothesis_test.html")

@app.route("/chi-square", methods=["GET", "POST"])
def chi_square():
    if request.method == "POST":
        try:
            raw_table = request.form["table"]
            rows = raw_table.strip().split("\n")
            table = [list(map(int, row.strip().split(","))) for row in rows]

            chi2, p_val, dof, expected = chi2_contingency(table)
            alpha = float(request.form["alpha"])
            decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"

            # Generate heatmap side-by-side
            plot_id = str(uuid.uuid4())

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(np.array(table), annot=True, fmt="d", cmap="Blues", ax=axes[0])
            axes[0].set_title("Observed")

            sns.heatmap(np.array(expected), annot=True, fmt=".2f", cmap="Oranges", ax=axes[1])
            axes[1].set_title("Expected")

            plt.tight_layout()
            heatmap_path = f"static/chi_heatmap_{plot_id}.png"
            plt.savefig(heatmap_path)
            plt.close()

            return render_template("chi_square_result.html",
                                   chi2=round(chi2, 4),
                                   p_val=round(p_val, 4),
                                   dof=dof,
                                   alpha=alpha,
                                   decision=decision,
                                   table=table,
                                   expected=expected,
                                   heatmap_path=heatmap_path)

        except Exception as e:
            return f"Error: {e}"

    return render_template("chi_square.html")

@app.route("/regression", methods=["GET", "POST"])
def regression():
    if request.method == "POST":
        try:
            x_values = [float(x) for x in request.form["x"].split(",")]
            y_values = [float(y) for y in request.form["y"].split(",")]

            if len(x_values) != len(y_values):
                return "X and Y must have the same number of values."

            slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

            # Generate plot
            import matplotlib.pyplot as plt
            import uuid
            import os
            import numpy as np

            x_array = np.array(x_values)
            y_array = np.array(y_values)
            y_pred = intercept + slope * x_array

            plt.figure(figsize=(6, 4))
            plt.scatter(x_array, y_array, label="Data Points")
            plt.plot(x_array, y_pred, color="orange", label=f"y = {intercept:.2f} + {slope:.2f}x")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Linear Regression")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_id = str(uuid.uuid4())
            plot_path = f"static/regression_{plot_id}.png"
            plt.savefig(plot_path)
            plt.close()

            return render_template("regression_result.html",
                                   slope=round(slope, 4),
                                   intercept=round(intercept, 4),
                                   r_value=round(r_value, 4),
                                   p_value=round(p_value, 4),
                                   std_err=round(std_err, 4),
                                   plot_path=plot_path)
        except Exception as e:
            return f"Error: {e}"

    return render_template("regression.html")

init_db()

if __name__ == "__main__":
    app.run(debug=True)