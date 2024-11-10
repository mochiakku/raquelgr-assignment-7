from flask import Flask, render_template, request, session, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from scipy import stats

app = Flask(__name__)
app.secret_key = 'your_secret_key'

if not os.path.exists("static"):
    os.makedirs("static")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    N = int(request.form['N'])
    mu = float(request.form['mu'])
    sigma2 = float(request.form['sigma2'])
    beta0 = float(request.form['beta0'])
    beta1 = float(request.form['beta1'])
    S = int(request.form['S'])

    slopes = []
    intercepts = []
    for _ in range(S):
        X = np.random.normal(mu, np.sqrt(sigma2), N)
        error = np.random.normal(0, np.sqrt(sigma2), N)
        Y = beta0 + beta1 * X + error
        
        slope, intercept, _, _, _ = stats.linregress(X, Y)
        slopes.append(slope)
        intercepts.append(intercept)

    session['slopes'] = slopes
    session['intercepts'] = intercepts
    session['beta0'] = beta0
    session['beta1'] = beta1

    plt.scatter(X, Y, alpha=0.5)
    plt.plot(X, beta0 + beta1 * X, color='red')
    plt.title("Scatter Plot with Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig('static/plot1.png')
    plt.close()

    plt.hist(slopes, bins=15, alpha=0.7, label='Slopes')
    plt.axvline(beta1, color='blue', linestyle='-', linewidth=2, label='True Slope')
    plt.title("Histogram of Slopes")
    plt.legend()
    plt.savefig('static/plot2.png')
    plt.close()

    return render_template('index.html', plot1=True)


@app.route('/hypothesis_test', methods=['POST'])
def hypothesis_test():
    parameter = request.form['parameter']
    test_type = request.form['test_type']

    observed_slope = session.get('beta1')
    observed_intercept = session.get('beta0')
    slopes = session.get('slopes')
    intercepts = session.get('intercepts')
    
    if parameter == 'slope':
        observed_stat = observed_slope
        simulated_stats = slopes
    else:
        observed_stat = observed_intercept
        simulated_stats = intercepts

    if test_type == '>':
        p_value = np.mean([s >= observed_stat for s in simulated_stats])
    elif test_type == '<':
        p_value = np.mean([s <= observed_stat for s in simulated_stats])
    elif test_type == '!=':
        p_value = np.mean([abs(s - observed_stat) >= abs(observed_stat) for s in simulated_stats])

    plt.hist(simulated_stats, bins=15, alpha=0.7, label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--', label=f'Observed {parameter.capitalize()}: {observed_stat:.4f}')
    plt.axvline(np.mean(simulated_stats), color='blue', linestyle='-', label=f'Hypothesized {parameter.capitalize()}')
    plt.legend()
    plt.title(f"Hypothesis Test for {parameter.capitalize()}")
    plt.savefig('static/plot3.png')
    plt.close()

    fun_message = None
    if p_value <= 0.0001:
        fun_message = "Rare event detected! p-value is extremely small."

    return render_template('index.html', plot1=True, plot3=True, parameter=parameter, observed_stat=observed_stat, p_value=p_value, fun_message=fun_message)



@app.route('/confidence_interval', methods=['POST'])
def confidence_interval():
    parameter = request.form['parameter']
    confidence_level = float(request.form['confidence_level']) / 100

    if parameter == 'slope':
        simulated_stats = session.get('slopes')
        true_value = session.get('beta1')
    else:
        simulated_stats = session.get('intercepts')
        true_value = session.get('beta0')

    mean_estimate = np.mean(simulated_stats)
    se = np.std(simulated_stats) / np.sqrt(len(simulated_stats))

    margin = stats.t.ppf(1 - (1 - confidence_level) / 2, len(simulated_stats) - 1) * se
    ci_lower = mean_estimate - margin
    ci_upper = mean_estimate + margin
    includes_true = ci_lower <= true_value <= ci_upper

    plt.scatter(simulated_stats, np.zeros_like(simulated_stats), color='gray', alpha=0.5)
    plt.plot([ci_lower, ci_upper], [0, 0], color='blue', linewidth=5, label=f"{confidence_level*100}% CI")
    plt.axvline(true_value, color='green', linestyle='--', label='True Value')
    plt.title(f"{confidence_level*100}% Confidence Interval for {parameter.capitalize()}")
    plt.legend()
    plt.savefig('static/plot4.png')
    plt.close()

    return render_template(
        'index.html',
        plot1=True,
        plot4=True,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_estimate=mean_estimate,
        confidence_level=confidence_level * 100,
        includes_true=includes_true,
        parameter=parameter  
    )

if __name__ == '__main__':
    app.run(debug=True)
